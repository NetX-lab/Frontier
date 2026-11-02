"""Retained P2 H200 job inventory. No platform query; never equate queue time to GPU time."""
import csv
import datetime
import json
from pathlib import Path
from assess_inputs import HERE, TASK, H3O, load, record, save

rows = []

def seconds(a,b):
    return (datetime.datetime.fromisoformat(b.replace('Z','+00:00')) - datetime.datetime.fromisoformat(a.replace('Z','+00:00'))).total_seconds()

def terminated(d):
    if isinstance(d,dict):
        if 'startedAt' in d and 'finishedAt' in d: yield d['startedAt'], d['finishedAt']
        elif 'started_at' in d and 'finished_at' in d: yield d['started_at'], d['finished_at']
        else:
            for v in d.values(): yield from terminated(v)
    elif isinstance(d,list):
        for v in d: yield from terminated(v)

def add(job,purpose,gpus,source,hours=None,basis='UNKNOWN',note='',reported=None,preemptible=False):
    source=Path(source).resolve()
    rows.append(dict(job_id=job,purpose=purpose,gpus=gpus,resource_class='H200 step_main '+('wholly preemptible' if preemptible else 'non-preemptible'),
                     gpu_hours=hours if hours is not None else 'UNKNOWN',basis=basis,reported_or_bound_gpu_hours=reported if reported is not None else '',
                     source=str(source),source_sha256=record(source)['sha256'],limitation=note))

def platform(path,purpose,gpus,per_gpu=None):
    d=load(path);job=d.get('job',d.get('name',d.get('requested')))
    spans=list(terminated(d.get('replicas',[])))
    assert spans
    if per_gpu is None: per_gpu=gpus/len(spans)
    hours=sum(seconds(a,b)*per_gpu/3600 for a,b in spans)
    if 'container_gpu_hours' in d: assert abs(hours-d['container_gpu_hours'])<1e-9
    add(job,purpose,gpus,path,hours,'container intervals',f'{len(spans)} containers; sum duration seconds x {per_gpu:g} GPUs / 3600; includes initialization, compile, warmup and teardown. Not a billing receipt.')

for p in sorted((TASK/'h200_mainline').glob('p2_*/package_*/case/runs/groundtruth_clean/*/platform_terminal_receipt.json')):
    case=p.parts[p.parts.index('h200_mainline')+1];gpus=32 if '32' in case else 64
    platform(p,case,gpus)
platform(TASK/'h200_dsv4_g0b_20261021/platform_terminal_receipt.json','Earlier DSV4 H200 full-model G0-B prerequisite (shared, not canonical P2 reference)',32)
p=TASK/'h200_dsv4_compute_20261021/observed/root_platform_terminal.json'
for i,j in enumerate(load(p)['jobs']):
    spans=list(terminated(j['replicas']));hours=sum(seconds(a,b)/3600 for a,b in spans)
    add(j['job'],'Earlier DSV4 H200 '+('shared FP8 operators' if i==0 else 'MARLIN MoE operators')+' (shared prerequisite; old tables not consumed by final P2)',1,p,hours,'container intervals')

# Old one-H200 jobs retain submission/phase clocks, not per-container allocation intervals.
items=[
 ('exp-1028-024316-296035','P1h graph bridge attempt1 FAILED private pool reuse','test_report_2026-10-27_b1_graph_replay_integration.md',.166,'submission-to-finish estimate'),
 ('exp-1028-031147-398843','P1h bridge attempt2','test_report_2026-10-27_b1_graph_replay_integration.md',.162,'submission-to-finish estimate'),
 ('exp-1028-132243-707076','P1h bridge package03','test_report_2026-10-28_p1h_package03.md',879/3600,'job wall window'),
 ('exp-1029-010007-750668','G1h01 FAILED fasteners import, all three engines','test_report_2026-10-28_g1h_worker_import_fix.md',.03,'reported approximate job GPUh'),
 ('exp-1029-020008-235508','G1h02 attention original grid','test_report_2026-10-28_g1h_worker_import_fix.md',3055/3600,'job wall window'),
 ('exp-1029-010008-493772','G3h MoE rank classes','test_report_2026-10-28_b12_g3h_moe_rank_classes.md',132.8/3600,'phase-only lower bound'),
 ('exp-1029-035012-764213','G2h standalone operators','test_report_2026-10-28_b13_v4_standalone_operators.md',407/3600,'job wall window'),
 ('exp-1029-113750-923871','G2h runner01','test_report_2026-10-29_g2h_runner_01.md',3089/3600,'create-to-stop window'),
 ('exp-1029-135035-301114','G2h replay02','test_report_2026-10-29_g2h_runner_02.md',3049/3600,'job wall window'),
 ('exp-1029-135037-016686','G2h block02 FAILED bootstrap','test_report_2026-10-29_g2h_runner_02.md',214/3600,'job wall window'),
 ('exp-1029-141735-456588','G2h block03','test_report_2026-10-29_g2h_runner_02.md',709/3600,'job wall window'),
 ('exp-1029-154446-382293','G2h runner04','test_report_2026-10-29_g2h_runner_04.md',3380/3600,'first Starting poll-to-finish upper bound'),
 ('exp-1030-021110-248027','G1h supplement 450 extra attention coordinates','test_report_2026-10-29_b22_g1h_writer_supplement.md',(1538.1+403.4)/3600,'phase-only lower bound'),
]
for job,purpose,report,hours,basis in items:
    add(job,purpose,1,TASK/report,None,'UNKNOWN full-container hours',basis+'; initialization/queue separation not fully witnessed. Reported value is not promoted to exact GPU-hours.',hours)

for name,gpus in [('dsv4-comm-g5h16-01',16),('dsv4-comm-g5h16-02',16),('dsv4-comm-g5h32-01',32),('dsv4-comm-g5h16-iso-01',16),('dsv4-comm-g5h32-iso-01',32)]:
    p=H3O/'g5h/launch/receipts'/f'{name}_exit.json'; d=load(p); h=(d['end']-d['start'])*gpus/3600
    add(d['rjob_name'],name,gpus,p,None,'UNKNOWN full-container hours','Launcher envelope includes startup, final barriers and possible queue; 32 isolated report explicitly includes ~5.5min queued. Do not sum envelope as measured spend.',h)
platform(H3O/'a6_repeatability_20261101/platform_03.json','A6 independent attention repeatability',1)
platform(H3O/'a6_three_run_20261101/platform_final.json','A6 three independent attention passes',1)
platform(H3O/'b30b/hardware_20261031/exp-1031-233253-628432_platform_20261031T153924Z.json','B30b linksKV01 FAILED UUID validation, all phases skipped',16,8)
platform(H3O/'b30b/hardware_20261031/exp-1031-233305-902815_platform_20261031T153924Z.json','B30b launch depth01',1,1)
platform(H3O/'b30b/execution_20261101/platform_terminal.json','B30b linksKV02 collective phases FAILED; NIXL02 PASS (fit rejected)',16,8)
platform(H3O/'b30b/execution_links03_20261101/platform_terminal.json','B30b links03 collectives',16,8)
p=H3O/'input_qualification_20261101/nixl6144_run/platform_terminal.json'; d=load(p)
add(d['name'],'Q2 exact T6144 NIXL timing',16,p,None,'UNKNOWN full-container hours','Pod read HTTP403; running window excludes container startup before job Running. The source labels it container_gpu_hours but computes job-running seconds x16/3600; scheduling-to-finish upper bound is 1.100310 GPUh; queue excluded.',d['running_seconds']*16/3600,True)
assert len(rows)==len({r['job_id'] for r in rows})
rows.sort(key=lambda x:x['job_id'])
for row in rows:
    row['accounting_scope']='earlier_shared_DSV4_prerequisite' if row['purpose'].startswith('Earlier DSV4') else 'P2_H200_campaign'
with (HERE/'gpu_hours.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
known=sum(r['gpu_hours'] for r in rows if isinstance(r['gpu_hours'],(float,int)))
summary=dict(job_count=len(rows),known_container_gpu_hours=known,unknown_jobs=[r['job_id'] for r in rows if r['gpu_hours']=='UNKNOWN'],
             scope_subtotals={scope:dict(job_count=sum(r['accounting_scope']==scope for r in rows),known_container_gpu_hours=sum(r['gpu_hours'] for r in rows if r['accounting_scope']==scope and isinstance(r['gpu_hours'],(float,int))),unknown_job_count=sum(r['accounting_scope']==scope and r['gpu_hours']=='UNKNOWN' for r in rows)) for scope in ['P2_H200_campaign','earlier_shared_DSV4_prerequisite']},
             total_actual_gpu_hours='UNKNOWN; '+str(known)+' container GPUh witnessed plus '+str(sum(r['gpu_hours']=='UNKNOWN' for r in rows))+' jobs with incomplete allocation clocks',
             scope='P2 H200 native, profiles, diagnostic/profiling/collective/NIXL jobs and earlier reused DSV4 H200 prerequisites. Each job counted once. GLM/H800 jobs, CPU analyses and unused package builds excluded.',
             reported_values_not_summed='Different timing bases (queue envelopes, phase-only, coarse upper bounds) cannot form an exact actual-spend total.', jobs=rows)
save('scratch/gpu_hours_detail.json',summary)
print('jobs',len(rows),'known container GPUh',known,'unknown',len(summary['unknown_jobs']))
