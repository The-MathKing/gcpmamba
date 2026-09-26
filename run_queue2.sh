#!/bin/bash
# Re-run Norman deep models that stopped before the min_epochs rule was introduced.
set -e
cd "$(dirname "$0")"
export PYTHONPATH=.
while pgrep -f run_queue.sh > /dev/null; do sleep 60; done
if [ ! -f results/.norman_pruned ]; then python3 -m gcpm.prune_short_runs norman; touch results/.norman_pruned; fi
python3 -m gcpm.benchmark --seeds 1 2 3 4 5 --models all
echo QUEUE2_DONE
