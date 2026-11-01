"""Frontier-price run-ahead FIFO model of the P2-64 prefill side, with fixed QPS-1 arrivals.

Same rule as a10/wave/fifo.py (lockstep lanes, K forwards in flight, 32-forward DP sync, wave
ends at a sync with no waiting request), with Frontier's zero host dummy cost. Arrivals are
i seconds plus the 10.3 ms ingress delay. Reports the per-request wait before the prefill
starts and the queue depth seen at formation.
"""
import statistics, sys

SYNC = 32


def simulate(g, P, K=19, c=0.0, ingress=10.3, n=128):
    arrivals = [i + ingress / 1000 for i in range(n)]
    i, t, gpu_free, running, step = 0, None, 0.0, False, 0
    done, pending, out, ends = [], [], [], []
    while i < n or pending:
        if not running:
            if not pending:
                pending.append(i); t = max(t or arrivals[i], arrivals[i]); i += 1
            running, step, done = True, 0, []
        while i < n and arrivals[i] <= t:
            pending.append(i); i += 1
        if pending:
            ahead = sum(1 for e in ends if e > t)
            start = max(t, gpu_free); end = start + P / 1000
            for r in pending:
                out.append(dict(request=r, wait_ms=(start - arrivals[r]) * 1000, ahead=ahead))
            pending = []
            gpu_free = end; done.append(end); ends.append(end); t = end
        else:
            issue = t if len([e for e in done[-K:] if e > t]) < K else max(t, done[-K])
            start = max(issue, gpu_free); end = start + g / 1000
            gpu_free = end; done.append(end); ends.append(end); t = issue + c / 1000
        step += 1
        if step % SYNC == 0:
            while i < n and arrivals[i] <= t:
                pending.append(i); i += 1
            if not pending:
                running = False
    return out


def summary(out):
    return statistics.mean(o["wait_ms"] for o in out), statistics.mean(o["ahead"] for o in out)


if __name__ == "__main__":
    # Frontier observed wait = (arrival->formed + formed->start) - 10.3 ms ingress, from prefill_wait.py.
    for name, g, P, obs_wait, obs_ahead in (("B29", 24.3801, 279.8667, 11.044 + 239.484 - 10.3, 10.219),
                                           ("pass1_all", 24.3575, 279.427, 12.010 + 254.180 - 10.3, 10.875),
                                           ("pass2_all", 24.3448, 279.7662, 11.939 + 253.828 - 10.3, 10.812)):
        out = simulate(g, P)
        w, a = summary(out)
        print(name, "model wait %.3f ahead %.3f" % (w, a), "| Frontier arrival->start %.3f ahead %.3f" % (obs_wait, obs_ahead))
        print("   model ahead sequence head:", [o["ahead"] for o in out[:34]])
