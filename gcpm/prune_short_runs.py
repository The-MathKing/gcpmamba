"""Remove results of deep-model runs that stopped before TRAIN['min_epochs'].

Training is deterministic, so a run that already lasted >= min_epochs is identical under
the min_epochs rule; shorter runs are deleted here and recomputed by gcpm.benchmark.
Usage: python -m gcpm.prune_short_runs norman
"""
import os
import sys
import pandas as pd
from gcpm.benchmark import TRAIN


def main(dataset):
    base = f'results/{dataset}_per_condition'
    curves = pd.read_csv(base + '_curves.csv')
    n = curves.groupby(['seed', 'model']).size()
    short = set(n[n < TRAIN['min_epochs']].index)
    print('rerunning', sorted(short))
    for path in (base + '.csv', base + '_curves.csv', base + '_wc_norms.csv'):
        if os.path.exists(path):
            d = pd.read_csv(path)
            keep = [(s, m) not in short for s, m in zip(d.seed, d.model)]
            d[keep].to_csv(path, index=False)


if __name__ == '__main__':
    main(sys.argv[1])
