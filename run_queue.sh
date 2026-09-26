#!/bin/bash
# Runs every experiment in the paper, in order. Each step skips work already written to
# results/, so the script can simply be re-run after an interruption.
set -e
cd "$(dirname "$0")"
export PYTHONPATH=.
SEEDS="1 2 3 4 5"
python3 -m gcpm.benchmark --seeds $SEEDS --models all
# two further initialisation replicates of the key models
python3 -m gcpm.benchmark --seeds $SEEDS --reps 1 2 --models "GCP-Mamba" "GCP-Mamba (perm. graph)" \
    "Mamba (no graph)" "Graph-MLP (no scan)" "GCP-Mamba (additive prior)"
python3 -m gcpm.benchmark --dataset adamson --seeds $SEEDS --models all
python3 -m gcpm.run_gears --dataset adamson --seeds $SEEDS
python3 -m gcpm.run_gears --seeds $SEEDS
[ -f results/scaling.csv ] || python3 -m gcpm.scaling
echo QUEUE_DONE
