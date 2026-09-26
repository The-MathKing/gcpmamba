"""Quantify the information lost by an HVG-restricted perturbation encoding.

The first version of this project (legacy data_loader.py) encoded a
perturbation as a multi-hot vector over the 500 highly variable genes and parsed
single perturbations with cond.split('+')[0], so 'ctrl+GENE' mapped to 'ctrl'.
Any condition whose targets are not HVGs, or are written second, receives an
all-zero input and is indistinguishable from every other such condition.

Usage: python -m gcpm.audit_hvg_encoding  -> results/hvg_encoding_audit.json
"""
import json
import os
import numpy as np

from data_loader import DataEngine


def main():
    out = {}
    for k in range(10):
        de = DataEngine(h5ad_path='data/norman/perturb_processed.h5ad',
                        splits_path=f'splits/splits_manifest_{k}.json')
        de.prepare_data()
        hv = set(de.hugo_names)
        man = json.load(open(f'splits/splits_manifest_{k}.json'))
        tg = {g for key in ('train_singles', 'test_singles', 'seen2_doubles', 'seen1_doubles', 'seen0_doubles')
              for c in man[key] for g in c.split('+') if g != 'ctrl'}
        split = {'targets': len(tg), 'targets_in_hvg': len(tg & hv)}
        for name, ld in [('train', de.train_loader), ('seen1', de.seen1_loader),
                         ('seen0', de.seen0_loader), ('test_singles', de.test_singles_loader)]:
            X = ld.dataset.X.numpy()
            split[name] = dict(n=len(X), all_zero=int((X.sum(1) == 0).sum()),
                               distinct_inputs=int(len(np.unique(X, axis=0))) if len(X) else 0)
        out[k] = split
        print(k, split, flush=True)
    os.makedirs('results', exist_ok=True)
    json.dump(out, open('results/hvg_encoding_audit.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
