"""CPU-only sensitivity model, adapted from a10/wave/fifo.py and scratch frontier_fifo.py.

No model parameters are fitted here. The caller supplies observed prices and
uses this model only after checking it against the existing replay waits.
"""
import statistics


def simulate(g, p, k=19, c=0.0, ingress=10.3, n=128):
    arrivals = [i + ingress / 1000 for i in range(n)]
    i, t, gpu_free, running, step = 0, None, 0.0, False, 0
    done, pending, out, ends = [], [], [], []
    while i < n or pending:
        if not running:
            if not pending:
                pending.append(i)
                t = max(t or arrivals[i], arrivals[i])
                i += 1
            running, step, done = True, 0, []
        while i < n and arrivals[i] <= t:
            pending.append(i)
            i += 1
        if pending:
            ahead = sum(e > t for e in ends[-k:])
            start = max(t, gpu_free)
            end = start + p / 1000
            for r in pending:
                out.append(dict(request=r, wait_ms=(start - arrivals[r]) * 1000, ahead=ahead))
            pending = []
            gpu_free = end
            done.append(end)
            ends.append(end)
            t = end
        else:
            issue = t if sum(e > t for e in done[-k:]) < k else max(t, done[-k])
            start = max(issue, gpu_free)
            end = start + g / 1000
            gpu_free = end
            done.append(end)
            ends.append(end)
            t = issue + c / 1000
        step += 1
        if step % 32 == 0:
            while i < n and arrivals[i] <= t:
                pending.append(i)
                i += 1
            if not pending:
                running = False
    return out


def mean_wait(rows):
    return statistics.mean(r["wait_ms"] for r in rows)


def phase_wait(g, p, **kwargs):
    """Long-window mean, not a gate reading or a warmed Frontier run."""
    return mean_wait(simulate(g, p, n=8192, **kwargs)[128:])
