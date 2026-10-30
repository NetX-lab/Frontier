#!/bin/bash
# One-at-a-time swaps of the FIFO model parameters (a10/wave/fifo.py) from the native set to
# Frontier's, at K = 19. Run from a directory that holds fifo.py and explore_out.json (explore.py).
# Usage: run_swaps.sh <wave-run artifacts> <output dir>
A=$1; O=$2; P=/data/ycfeng/envs/frontier-py310/bin/python
ulimit -v 31457280
for spec in "native 23.8 4.68 19 300 6.5 3" "g_frontier 24.38 4.68 19 300 6.5 3" "c_frontier 23.8 0 19 300 6.5 3" \
  "P_frontier 23.8 4.68 19 275.2 6.5 3" "dsched_frontier 23.8 4.68 19 300 3.3 3" "dpost_frontier 23.8 4.68 19 300 6.5 0" \
  "frontier 24.38 0 19 275.2 3.3 0" "frontier_P300 24.38 0 19 300 3.3 0" "frontier_P290 24.38 0 19 290 3.3 0"; do
  set -- $spec; n=$1; shift
  timeout 120 $P fifo.py $A "$@" > $O/fifo_$n.txt 2>&1
done
