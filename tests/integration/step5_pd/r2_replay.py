"""Run frozen native R2 waves, direct-D independent references and soft idle diagnostics."""

import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'comparison/calibration'))
import pd_replay
from serving_metrics import parse_prometheus, transfer_delta
try:
    from . import canonical_replay
except ImportError:
    import canonical_replay


async def run(args):
    from aiohttp import web

    stage = args.trace_dir.parent
    config = json.loads((stage / 'config.json').read_text())
    definition = json.loads((stage / 'r2_run.json').read_text())
    settings = config['canonical_g3']
    model = canonical_replay.local_model_config(args.model_dir, 'step5')
    urls = {'prefill': args.prefill_url, 'decode': args.decode_url}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'diagnostic_only': True, 'clean_timing_eligible': False, 'waves': {},
              'errors': [], 'ready_for_formal': False}

    def persist():
        (args.output_dir / 'r2_replay_summary.json').write_text(json.dumps(report, indent=1))

    async def idle():
        try:
            return await canonical_replay.observe_idle(urls, settings['dp_size'],
                settings['idle_timeout_s'], settings['idle_poll_s'])
        except Exception as error:
            return {'verified': False, 'error': repr(error)}

    app = canonical_replay.build_app(args.prefill_url, args.decode_url, '', '',
        args.output_dir / 'proxy_requests.jsonl', vllm_version='step5')
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    try:
        await web.TCPSite(runner, '127.0.0.1', args.proxy_port).start()
        for wave in definition['waves']:
            entry = report['waves'][wave['name']] = {'local_prefill_reference': bool(wave.get('local_prefill_reference'))}
            # First-start gauges can be absent. Subsequent/reference boundaries
            # retain their full observations and are required by the consumer.
            if wave['name'] == 'warmup16':
                entry['before'] = {'verified': False, 'diagnostic_only': True}
            else:
                entry['before'] = await idle()
            records = []
            try:
                rows = json.loads((args.trace_dir / wave['name'] / 'request_ids.json').read_text())['rows']
                direct = args.decode_url+'/v1/completions' if wave.get('local_prefill_reference') else None
                entry['dispatch'] = 'direct_native_D_no_proxy_or_transfer_metadata' if direct else 'native_PD_proxy'
                entry['summary'], _ = await canonical_replay.replay(rows, args.proxy_port, model['vocab_size'],
                    args.origin_lead_s, vllm_version='step5', records=records, completion_url=direct)
                invalid = {}
                for record in records:
                    if not record.get('cancel_after_first_content') and not record.get('cancel_after_decode_submit_s'):
                        errors = canonical_replay.validate_record(record)
                        if errors:
                            invalid[record['request_id']] = errors
                if invalid:
                    entry['request_errors'] = invalid
                    raise ValueError('Invalid native request records: '+json.dumps(invalid))
            except Exception as error:
                entry['error'] = repr(error)
                report['errors'].append(wave['name']+': '+repr(error))
                print('NATIVE_PD_AUXILIARY_UNVERIFIED', report['errors'][-1], flush=True)
            destination = args.output_dir / wave['name']
            destination.mkdir(parents=True, exist_ok=True)
            (destination / 'client_requests.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
            entry['after'] = await idle()
            try:
                if entry['before']['verified'] and entry['after']['verified']:
                    entry['transfer_delta'] = {role: transfer_delta(
                        parse_prometheus(entry['before']['observations'][-1]['pages'][role]),
                        parse_prometheus(entry['after']['observations'][-1]['pages'][role])) for role in urls}
            except Exception as error:
                entry['metrics_error'] = repr(error)
            persist()
    finally:
        await runner.cleanup()
    return 0  # All admission/auxiliary checks are soft, including reference failures.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('engine-config', 'trace-dir', 'output-dir', 'model-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('mode', 'prefill-url', 'decode-url'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--proxy-port', type=int, default=8000)
    parser.add_argument('--origin-lead-s', type=float, default=1)
    parser.add_argument('--startup-timeout-s', type=float)
    parser.add_argument('--drain-s', type=float)
    args = parser.parse_args(argv)
    if args.mode != 'clean':
        parser.error('Canonical native diagnostics require the unchanged clean serving adapter')
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
