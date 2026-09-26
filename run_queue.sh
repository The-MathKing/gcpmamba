#!/bin/bash
# Runs every experiment in the paper, in order. Each step skips work already written to
# results/, so the script can simply be re-run after an interruption.
set -e
cd "$(dirname "$0")"
export PYTHONPATH=.
SEEDS="1 2 3 4 5"
python3 -m gcpm.benchmark --seeds $SEEDS --models all
python3 -m gcpm.run_gears --seeds $SEEDS
python3 -m gcpm.benchmark --dataset adamson --seeds $SEEDS --models all
python3 -m gcpm.run_gears --dataset adamson --seeds $SEEDS
[ -f results/scaling.csv ] || python3 -m gcpm.scaling
echo QUEUE_DONE
