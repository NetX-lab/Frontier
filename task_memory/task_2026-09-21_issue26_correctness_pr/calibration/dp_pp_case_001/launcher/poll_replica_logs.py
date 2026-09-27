"""Keep the longest replica-log snapshot of one RJob on local disk.

logs_rjob returns an empty string for these jobs; the replica endpoint returns
a JSON envelope whose rows carry the container's lines. The platform keeps only
a tail and reclaims it with the pod, so the poller runs while the job lives and
for a while after it ends (the full tail appears some minutes later).
"""

import json
import os
import signal
import sys
import time

from steptron.utils.stepmind import get_rjob_client

job = os.environ["STEPMIND_JOB"]
out = os.environ["WORKER_LOG_PATH"]
deadline = time.time() + float(os.environ["POLL_SECONDS"])
settle_s = float(os.environ.get("SETTLE_SECONDS", "420"))

call_timeout_s = 60


def call_timed_out(signum, frame):
    raise TimeoutError(f"log query exceeded {call_timeout_s} s")


# A log query to a stopped job can block without returning; the alarm bounds
# each query so the settle deadline is still checked.
signal.signal(signal.SIGALRM, call_timed_out)
client = get_rjob_client()
best, done_at = "", None
while time.time() < deadline:
    signal.alarm(call_timeout_s)
    try:
        lines = []
        for rows in client.get_rjob_infos(job).values():
            for replica in rows:
                response = client.logs_replica(replica["name"], tail_lines=3000)
                text = response.text if hasattr(response, "text") else str(response)
                lines += [row.get("message", "") for row in json.loads(text)["data"]]
        text = "\n".join(lines)
    except Exception as exc:
        text = ""
        print(f"[poll] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        signal.alarm(0)
    if len(text) > len(best):
        best = text
        with open(out, "w") as handle:
            handle.write(best + "\n")
        print(f"[poll] captured {len(best)} bytes", flush=True)
    if done_at is None and "WORKER_STATUS=" in best:
        done_at = time.time()
    if done_at is not None and time.time() - done_at > settle_s:
        break
    time.sleep(20)
print(f"[poll] final {len(best)} bytes -> {out}", flush=True)
