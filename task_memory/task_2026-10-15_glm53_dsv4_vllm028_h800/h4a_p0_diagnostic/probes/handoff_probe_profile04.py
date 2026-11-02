"""Read-only cross-check of h4a5_attribution.profile02_handoff on profile04: place each traced D k=1 step
(TP0 rank of each decode lane) inside its request's proxy handoff, using wall clocks."""
import glob, gzip, json, re
R = "/data/ycfeng/tmp/glm53_dsv4/profile04_results_1u00cy05/artifacts/case/runs/groundtruth_clean/dsv4-h800-p0-profile-20261024-04"
STEP = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
proxy = [json.loads(line) for line in open(f"{R}/run/proxy_requests.jsonl")]
for dp in ("dp0", "dp1"):
    [path] = glob.glob(f"{R}/torch_traces/decode/{dp}/{dp}_pp0_tp0_*")
    trace = json.load(gzip.open(path))
    base = trace["baseTimeNanoseconds"] / 1e9
    ann = {cat: sorted((base + e["ts"] / 1e6, e["name"]) for e in trace["traceEvents"]
                       if e.get("cat") == cat and e["name"].startswith("execute_context"))
           for cat in ("gpu_user_annotation", "user_annotation")}
    gpu = ann["gpu_user_annotation"]
    for (start, step), (following, _) in zip(gpu, gpu[1:]):
        if STEP.fullmatch(step).group(1) == "0":
            continue
        for p in [p for p in proxy if p["prefill_response_time"] < start < p["decode_first_byte_time"]]:
            cpu = max(t for t, n in ann["user_annotation"] if n == step and t <= start)
            pauses = [b - a for (a, _), (b, _) in zip(gpu, gpu[1:]) if a < start and b > p["prefill_response_time"]]
            print(dp, p["request_id"][-24:], step,
                  "handoff %.2f" % ((p["decode_first_byte_time"] - p["prefill_response_time"]) * 1e3),
                  "resp->cpu %.2f" % ((cpu - p["prefill_response_time"]) * 1e3),
                  "cpu->gpu %.2f" % ((start - cpu) * 1e3),
                  "step %.2f" % ((following - start) * 1e3),
                  "next->byte %.2f" % ((p["decode_first_byte_time"] - following) * 1e3),
                  "pause %s" % (round(max(pauses) * 1e3, 2) if pauses else None))
