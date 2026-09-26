"""Head-to-head GEARS (Roohani et al., 2024) on the same GEARS simulation splits.

Official cell-gears implementation and default hyperparameters; only the number
of epochs is set here. Predictions are scored with the same metrics as every
other model (gcpm.benchmark.condition_metrics).

Usage: python -m gcpm.run_gears --seeds 1 2 3 4 5 --epochs 20
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from gears import PertData, GEARS

# cell-gears indexes a scipy sparse matrix with a boolean pandas Series, which recent scipy
# handles by calling Series.nonzero() (removed in pandas 2); restore it for GEARS only
if not hasattr(pd.Series, 'nonzero'):
    pd.Series.nonzero = lambda self: self.to_numpy().nonzero()

from gcpm.data import NormanData, load_split, targets
from gcpm.benchmark import condition_metrics, OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--dataset', default='norman')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    args.out = args.out or OUT.format(args.dataset)
    torch.set_num_threads(os.cpu_count())
    d = NormanData(args.dataset)
    pert = PertData('gears_data')
    pert.load(data_path=f'gears_data/{args.dataset}')
    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out, usecols=['seed', 'model'])
        done = set(prev.seed[prev.model == 'GEARS'])
    for seed in args.seeds:
        if seed in done:
            continue
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        pert.prepare_split(split='simulation', seed=seed)
        pert.get_dataloader(batch_size=32, test_batch_size=128)
        model = GEARS(pert, device='cpu')
        model.model_initialize(hidden_size=64)
        model.train(epochs=args.epochs)
        _, sub = load_split(seed, args.dataset)
        rows = []
        for c in pert.set2conditions['test']:
            t = targets(c)
            p = model.predict([t])['_'.join(t)]
            p = np.asarray(p).ravel() - d.ctrl_mean
            rows.append(dict(seed=seed, model='GEARS', condition=c, subgroup=sub[c],
                             **condition_metrics(d, c, p)))
        pd.DataFrame(rows).to_csv(args.out, mode='a', header=not os.path.exists(args.out), index=False)
        print(f'[split {seed}] GEARS done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
