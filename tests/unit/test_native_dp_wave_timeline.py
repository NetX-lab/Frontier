"""Synthetic wave timelines: classification, joins, missing data and attribution."""
import json

import pytest

from native_dp_wave_timeline import Records, analyze, candidate_intervals, classification


def span(t, end=None, **fields):
    return dict(t=t, mono=t, end_t=t if end is None else end,
                end_mono=t if end is None else end, **fields)


@pytest.mark.parametrize('gap,duration,expected', [
    (.2,.30,'served_at_once'), (.2,.75,'held'), (.26,.58,'260_760_ms'),
    (.76,.4,'after_760_ms'), (2.1,.8,'ungrouped')])
def test_historical_classification(gap, duration, expected):
    assert classification(gap,gap+duration,0)==expected


def test_iteration_spans_are_indexed():
    rows = [dict(event='busy_iteration', t=10, mono=20, calls={
        'engine_dummy_batch':dict(mono=20.1,end_mono=20.2)})]
    assert Records(rows).between('engine_dummy_batch',10,11)==[
        dict(event='engine_dummy_batch', t=10+(20.1-20), end_t=10+(20.2-20), mono=20.1,end_mono=20.2)]


@pytest.mark.parametrize('values,expected', [
    ((.3,.301,.302,.303,.304),'start_wave_message'),
    ((.01,.3,.301,.302,.303),'start_wave_message'),
    ((.01,.02,.021,.3,.301),'idle_engine'),
    ((.01,.02,.021,.022,.3),'worker_join')])
def test_delay_candidates(values, expected):
    enqueue, handle, handled, dummy, reduce = values
    lanes = [dict(lane=1, enqueue=span(enqueue), handle=span(handle,handled),
                  dummy=span(dummy), reduce=span(reduce,reduce+.01))]
    intervals, candidate = candidate_intervals(0,reduce+.31,lanes,span(.02,reduce+.01),300)
    assert candidate==expected
    assert intervals['critical_lane']==1


@pytest.mark.parametrize('reduce_end,update_end,expected', [(.5,.8,'all_reduce'),(.03,.8,'after_all_reduce')])
def test_reduce_and_after_reduce(reduce_end,update_end,expected):
    lanes = [dict(lane=1,enqueue=span(.001),handle=span(.002,.003),
                  dummy=span(.004),reduce=span(.005,reduce_end))]
    _, candidate = candidate_intervals(0,update_end,lanes,span(.002,reduce_end),300)
    assert candidate==expected


def test_real_lane_can_join_last():
    peer = dict(lane=1,enqueue=span(.001),handle=span(.002,.003),dummy=span(.004),reduce=span(.005,.41))
    real = dict(lane=0,enqueue=span(-.01),handle=span(-.009,-.008),dummy=span(0),reduce=span(.4,.41))
    intervals, candidate = candidate_intervals(0,.72,[peer],real['reduce'],300,real)
    assert intervals['critical_lane']==0 and candidate=='worker_join'


def test_missing_and_clock_order():
    assert candidate_intervals(0,1,[dict(reduce=None)],None,300)==(None,'unavailable')
    lane = dict(lane=1,enqueue=span(.1),handle=span(.05,.06),dummy=span(.07),reduce=span(.08,.09))
    assert candidate_intervals(0,1,[lane],span(.08,.09),300)==(None,'clock_order_unverified')


def write(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


@pytest.mark.parametrize('fault', [None, 'reduced_values', 'clock_order', 'missing_lane'])
def test_full_four_lane_run(tmp_path, fault):
    request='cmpl-formal-r001-repeat01'
    wave=1
    for rank in range(4):
        if fault == 'missing_lane' and rank == 3:
            continue
        rows=[dict(event='start',role='prefill',dp_rank=rank,t=0)]
        if rank==0:
            rows.extend([
                dict(event='wave_complete',wave=0,t=1),
                dict(event='add_enqueue',request_id=request,request_wave=0,**span(1.998)),
                dict(event='engine_add',request_id=request,current_wave_after=wave,**span(1.999,2)),
                dict(event='add',request_id=request,arrival_time=1.99,t=1.999),
                dict(event='schedule',step=1,t=2,tokens=6144,new=[[request,0,6144]]),
                dict(event='busy_iteration',t=2,mono=2,calls={'process_engine_step':dict(mono=2,end_mono=2.7)}),
                dict(event='update',step=1,t=2.7,dur_ms=1,first_token=[request])])
        else:
            rows.extend([
                dict(event='start_dp_wave_enqueue',wave=wave,excluded_engine_index=0,**span(2.001)),
                dict(event='handle_client_request',request_type='START_DP_WAVE',wave=wave,**span(2.002,2.003)),
                dict(event='busy_iteration',t=2,mono=2,calls={'engine_dummy_batch':dict(mono=2.004,end_mono=2.7)})])
        write(tmp_path/f'nodes/h{rank}/scheduler_lifecycle/dp{rank}.jsonl',rows)
        worker=[dict(event='worker_wave_start',role='prefill',dp_rank=rank,t=0)]
        calls = {
            'dp_padding_sync':dict(forward=1,stage='serving',dp_size=4,num_tokens=6144 if rank==0 else 1,mono=2.01,end_mono=2.4),
            'dp_cpu_all_reduce':dict(forward=1,reduced_values=[[6144,1,1,1]],mono=2.02+rank*.01,end_mono=2.4)}
        worker.append(dict(event='worker_execute_model' if rank==0 else 'worker_dummy_batch',
            request_ids=[request] if rank==0 else [],calls=calls,**span(2.003,2.69)))
        if rank == 3 and fault == 'reduced_values':
            calls['dp_cpu_all_reduce']['reduced_values'] = [[1,6144,1,1]]
        if rank == 3 and fault == 'clock_order':
            rows[-1]['calls']['engine_dummy_batch']['mono'] = 2.06
            write(tmp_path/f'nodes/h{rank}/scheduler_lifecycle/dp{rank}.jsonl',rows)
        write(tmp_path/f'nodes/h{rank}/worker_dp_wave/dp{rank}.jsonl',worker)
    write(tmp_path/'run/proxy_requests.jsonl',[dict(proxy_arrival_time=1.99,
        kv_transfer_params=dict(remote_request_id=request))])
    rows, summary=analyze(tmp_path)
    assert summary['formal_prefills']==1
    assert summary['complete_timelines'] == (1 if fault is None else 0)
    assert rows[0]['arrival']==1.99
    assert len(rows[0]['other_lanes']) == (2 if fault == 'missing_lane' else 3)
    expected = {None:'all_reduce', 'reduced_values':'rendezvous_unverified',
        'clock_order':'clock_order_unverified', 'missing_lane':'rendezvous_unverified'}
    assert rows[0]['delay_candidate']==expected[fault]
    assert summary['errors']==[]


def test_historical_records_have_unavailable_intervals(tmp_path):
    write(tmp_path/'nodes/h0/scheduler_lifecycle/dp0.jsonl',[
        dict(event='start',role='prefill',dp_rank=0,t=0),
        dict(event='add',request_id='formal',arrival_time=1,t=1),
        dict(event='schedule',step=1,t=1,tokens=6144,new=[['formal',0,6144]]),
        dict(event='update',step=1,t=1.5,first_token=['formal'])])
    rows,summary=analyze(tmp_path)
    assert rows[0]['intervals'] is None and rows[0]['delay_candidate']=='unavailable'
    assert summary['complete_timelines']==0
