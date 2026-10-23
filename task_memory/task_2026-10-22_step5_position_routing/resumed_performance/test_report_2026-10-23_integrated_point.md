# Integrated performance source and actual-point check

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Verify reviewed79 source,unchanged full pricing payload and live-read/decode counts;launch main08. |

Workdir:/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06.
Source79f1a3fd35d42b751cd1520c764ddac717b923d9 and independent3dac53c submodule
are clean. All seven changed source/test files match their reviewed owner commits.
Parent component checks and both independent review axes closed before this freeze.

## Actual-point execution

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 PYTHONPATH="$PWD:$PWD/tests/unit" TMPDIR=/data/ycfeng/tmp \
  /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_perf_parent_loop.py \
  --source-commit 79f1a3fd35d42b751cd1520c764ddac717b923d9 --execution --no-profile \
  --iterations 5 --output /data/ycfeng/tmp/d29_performance_integrated_parent01 \
  > /data/ycfeng/tmp/d29_performance_integrated_parent01.log 2>&1
```

Python3.10.6. Audited exact_text_loop.py SHA
0dfebf96ac7234f496c1ffb181528a0ba4397b7f8c1cbc424b61c206be7d8d61;
launcher SHA b12dc2a85fb83046eae68ca4de396b7fc49790e033e4e8d5cdf34e298b7a08b2.
Launcher validates clean source/owner bytes and changes only the probe's source
directory literal. Both diagnostic reuse switches are OFF. Probe RLIMIT_AS30GiB
remains;actual manager/cache setup is excluded from timing,as in the baseline.

Expected:one88-layer resolution across5calls,all20current artifact reads,one
large decode,and original full cost/proof/count/plan bytes. Observed PASS,exit0:
88descriptions,20reads,1decode;times0.695308/0.128664/0.254753/0.128842/0.132253s.
Last-three mean0.171949546s;retain the observed variation rather than declaring
a fixed latency. Text/parsed residency18,617,675B;peak RSS1,216,764KiB.

Parent compares complete_payload.json bytes against exact_baseline01 directly.
Full-file SHA c2aa250d2103db50140e3f19c3579e9221dcfaf6251df1b44ed5f580982c5582;
canonical extended3888b075b5af3fd63fac1c7c0ef9ef27a4f65732a874742875c48f02cd3cd1a7;
originald2c141f088b52063edb539094fe3e0967f35e53303d70ebe2e8fbdbe4c86c952.
All unchanged. Result JSON SHA a9acebdd46a5056068dcceb0ca79c1c588e868132cabf5551945c2e70e78d9d8.
This is the originalP6128+idle1 CPU/unadmitted point,not complete workflow timing.

## Final-source main08 launch

```bash
timeout 5100s env -u FRONTIER_TEST_CPU_WRITER_FILE -u FRONTIER_TEST_CPU_WRITER_SHA256 \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PWD:$PWD/tests/unit" TMPDIR=/data/ycfeng/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp \
  /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/run_step5_measured_main.py \
  --fixture-manifest /data/ycfeng/tmp/d29-parent-position-fixture03/fixture_manifest.json \
  --output-root /data/ycfeng/tmp/d29-parent-position-main08 --mode all \
  > /data/ycfeng/tmp/d29-parent-position-main08.log 2>&1
```

Original6144/1024 workload,2400s per mode;running/unverified when recorded.
Its nine launch input SHA256 values equal main07;only the output path of the
identical synthetic_read_cost.json is normalized. New input receipt SHA
b094262a4199ea2a369538bec772674a093622b3ad779aedbb0027da64dedd8f.
Fresh79 strict77/full-unit/supplements and isolated T2 rebinding run separately.
No full workflow or native/GPU/G3/E2E verdict is inferred from this launch.
