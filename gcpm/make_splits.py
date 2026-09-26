"""Generate the official GEARS 'simulation' splits (seeds 1-5) with cell-gears.

Usage: python -m gcpm.make_splits norman|adamson
Writes splits/gears_simulation_splits[_<dataset>].json.
"""
import json
import os
import sys
from gears import PertData


def main(dataset):
    os.makedirs(f'gears_data/{dataset}', exist_ok=True)
    for f in ['perturb_processed.h5ad', 'go.csv']:
        dst = f'gears_data/{dataset}/{f}'
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(f'data/{dataset}/{f}'), dst)
    pert = PertData('gears_data')
    pert.load(data_path=f'gears_data/{dataset}')
    out = {}
    for s in range(1, 6):
        pert.prepare_split(split='simulation', seed=s)
        out[s] = {'set2conditions': pert.set2conditions, 'subgroup': pert.subgroup}
    suffix = '' if dataset == 'norman' else '_' + dataset
    json.dump(out, open(f'splits/gears_simulation_splits{suffix}.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
