#!/bin/bash
# GEARS on both datasets, Adamson benchmark, scaling (after queue 2).
set -e
cd "$(dirname "$0")"
export PYTHONPATH=.
while pgrep -f run_queue2.sh > /dev/null; do sleep 60; done
python3 -m gcpm.run_gears --seeds 1 2 3 4 5
python3 -m gcpm.benchmark --dataset adamson --seeds 1 2 3 4 5 --models all
python3 -m gcpm.run_gears --dataset adamson --seeds 1 2 3 4 5
[ -f results/scaling.csv ] || python3 -m gcpm.scaling
echo QUEUE3_DONE
