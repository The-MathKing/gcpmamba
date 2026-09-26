#!/bin/bash
# Runs every experiment in the paper, in order. Each step skips work already written to
# results/, so the script can simply be re-run after an interruption.
set -e
cd "$(dirname "$0")"
export PYTHONPATH=.
SEEDS="1 2 3 4 5"

# wait for any benchmark already running (e.g. started by hand)
while pgrep -f "gcpm.benchmark --seeds" > /dev/null; do sleep 60; done

python3 -m gcpm.benchmark --seeds $SEEDS --models all
# baselines: drop once and recompute so every split uses the validation-selected ridge penalty
if [ ! -f results/.baselines_recomputed ]; then
python3 - <<'PY'
import pandas as pd
f = 'results/norman_per_condition.csv'
d = pd.read_csv(f)
d[~d.model.isin(['No change', 'Train mean', 'Additive', 'Linear (PCA)', 'Linear (ctrl emb.)'])].to_csv(f, index=False)
PY
touch results/.baselines_recomputed
fi
python3 -m gcpm.benchmark --seeds $SEEDS --models baselines
python3 -m gcpm.run_gears --seeds $SEEDS
python3 -m gcpm.benchmark --dataset adamson --seeds $SEEDS --models all
python3 -m gcpm.run_gears --dataset adamson --seeds $SEEDS
[ -f results/scaling.csv ] || python3 -m gcpm.scaling
echo QUEUE_DONE
