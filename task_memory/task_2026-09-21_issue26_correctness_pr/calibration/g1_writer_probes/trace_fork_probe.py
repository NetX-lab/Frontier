import importlib.util, json, multiprocessing, os, sys, time
from pathlib import Path
log_dir = Path(sys.argv[1]); os.environ["VLLM_FRONTIER_DP_PLACEMENT_LOG_DIR"] = str(log_dir)
spec = importlib.util.spec_from_file_location("frontier_trace", "/data/ycfeng/Frontier/.real-engine/vLLM-BS/vllm/v1/frontier_trace.py")
trace = importlib.util.module_from_spec(spec); spec.loader.exec_module(trace)

def child():
    for i in range(3):
        trace.log_dp_placement_record("engine_iteration", step=i)
    time.sleep(30)

if __name__ == "__main__":
    trace.log_dp_placement_record("frontend_route", request_id="r0")
    p = multiprocessing.get_context("fork").Process(target=child); p.start()
    time.sleep(1.0); p.terminate(); p.join()
    for f in sorted(log_dir.iterdir()):
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        print(f.name == f"dp_placement_{rows[0]['pid']}.jsonl", [(r["kind"], r["seq"], r.get("step")) for r in rows])
    print("child exitcode", p.exitcode)
