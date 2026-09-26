"""Tables and figures for the manuscript, computed only from results/*.csv.

Usage: python -m gcpm.report
Writes results/summary_*.csv, manuscript/tables.tex and manuscript/fig_*.pdf.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = 'results/norman_per_condition.csv'
SUBGROUPS = ['unseen_single', 'combo_seen0', 'combo_seen1', 'combo_seen2']
SUB_LABEL = {'unseen_single': 'Unseen single', 'combo_seen0': 'Double, 0/2 seen',
             'combo_seen1': 'Double, 1/2 seen', 'combo_seen2': 'Double, 2/2 seen', 'all': 'All test'}
BASELINES = ['No change', 'Train mean', 'Additive', 'Linear (PCA)', 'Linear (ctrl emb.)']
MAIN = ['GCP-Mamba', 'GEARS']
ABLATIONS = ['GCP-Mamba (Delta only)', 'GCP-Mamba (perm. graph)', 'GCP-Mamba (random order)',
             'Mamba (no graph)', 'Graph-MLP (no scan)']
ORDER = BASELINES + ['GEARS', 'GCP-Mamba'] + ABLATIONS
# validated categorical slots (dataviz reference palette, light mode): 1 blue, 2 orange; neutrals for baselines
COLOR = {m: '#8a8984' for m in BASELINES}
COLOR.update({m: '#2a78d6' for m in ['GCP-Mamba'] + ABLATIONS}, GEARS='#eb6834')
INK, INK2 = '#0b0b0b', '#52514e'


def split_means(df, metric, by_sub=True):
    keys = ['model', 'seed'] + (['subgroup'] if by_sub else [])
    return df.groupby(keys)[metric].mean().reset_index()


def summary(df, metric):
    """mean and s.d. over the five splits of the per-split mean, per subgroup and overall."""
    s = split_means(df, metric)
    a = split_means(df, metric, by_sub=False).assign(subgroup='all')
    s = pd.concat([s, a])
    g = s.groupby(['model', 'subgroup'])[metric].agg(['mean', 'std']).reset_index()
    return g


def paired_test(df, a, b, metric):
    """Wilcoxon signed-rank on per-condition differences (a - b), pooled over splits."""
    x = df[df.model == a].set_index(['seed', 'condition'])[metric]
    y = df[df.model == b].set_index(['seed', 'condition'])[metric]
    j = pd.concat([x, y], axis=1, keys=['a', 'b']).dropna()
    if len(j) < 5 or np.allclose(j.a, j.b):
        return len(j), float(np.median(j.a - j.b)) if len(j) else np.nan, np.nan
    return len(j), float(np.median(j.a - j.b)), float(wilcoxon(j.a, j.b).pvalue)


def holm(p):
    p = np.asarray(p, float)
    idx = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(idx):
        running = max(running, (len(p) - r) * p[i])
        out[i] = min(1.0, running)
    return out


def fmt(m, s, digits=3):
    return f'{m:.{digits}f}\\,$\\pm$\\,{s:.{digits}f}' if np.isfinite(m) else '---'


def table_main(df, models, metric, digits, caption, label):
    g = summary(df, metric).set_index(['model', 'subgroup'])
    cols = SUBGROUPS + ['all']
    lines = [r'\begin{table*}[!t]', r'\processtable{' + caption + r'\label{' + label + '}}{',
             r'\begin{tabular*}{\hsize}{@{\extracolsep{\fill}}l' + 'c' * len(cols) + '@{}}', r'\toprule',
             'Model & ' + ' & '.join(SUB_LABEL[c] for c in cols) + r' \\', r'\midrule']
    best = {c: None for c in cols}
    for c in cols:
        vals = {m: g.loc[(m, c), 'mean'] for m in models if (m, c) in g.index}
        vals = {m: v for m, v in vals.items() if np.isfinite(v)}
        if vals:
            best[c] = (min if metric.startswith('mse') else max)(vals, key=vals.get)
    for m in models:
        cells = []
        for c in cols:
            if (m, c) in g.index:
                cell = fmt(g.loc[(m, c), 'mean'], g.loc[(m, c), 'std'], digits)
                cells.append(r'\textbf{' + cell + '}' if best[c] == m else cell)
            else:
                cells.append('---')
        lines.append(m.replace('Delta', r'$\Delta$') + ' & ' + ' & '.join(cells) + r' \\')
        if m == BASELINES[-1] or m == 'GCP-Mamba':
            lines.append(r'\midrule')
    lines += [r'\botrule', r'\end{tabular*}}{}', r'\end{table*}']
    return '\n'.join(lines)


def comparisons(df):
    rows = []
    pairs = [('GCP-Mamba', m) for m in BASELINES + ['GEARS'] + ABLATIONS]
    for metric in ['mse_de20', 'pearson_delta_de20', 'gi_mse_de20']:
        res = [(a, b) + paired_test(df, a, b, metric) for a, b in pairs]
        padj = holm([r[4] if np.isfinite(r[4]) else 1.0 for r in res])
        for r, q in zip(res, padj):
            rows.append(dict(metric=metric, model_a=r[0], model_b=r[1], n=r[2],
                             median_diff=r[3], p=r[4], p_holm=q))
    return pd.DataFrame(rows)


def fig_benchmark(df, path):
    metrics = [('mse_de20', 'MSE, top-20 DE genes (lower is better)'),
               ('pearson_delta_de20', 'Pearson $r$ of $\\delta$, top-20 DE genes')]
    models = [m for m in ORDER if m in df.model.unique()]
    fig, axes = plt.subplots(len(metrics), 4, figsize=(7.1, 5.4), sharey=True)
    for r, (metric, title) in enumerate(metrics):
        s = split_means(df, metric)
        for c, sg in enumerate(SUBGROUPS):
            ax = axes[r, c]
            for k, m in enumerate(models):
                v = s[(s.model == m) & (s.subgroup == sg)][metric].to_numpy()
                v = v[np.isfinite(v)]
                if not len(v):
                    continue
                y = len(models) - 1 - k
                ax.scatter(v, np.full(len(v), y), s=9, color=COLOR[m], alpha=0.45, lw=0, zorder=2)
                ax.scatter([v.mean()], [y], s=34, color=COLOR[m], edgecolor='white', lw=1.2, zorder=3)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([m.replace('Delta', r'$\Delta$') for m in models[::-1]], fontsize=6.5, color=INK)
            ax.tick_params(axis='x', labelsize=6.5, colors=INK2, length=2)
            ax.tick_params(axis='y', length=0)
            ax.grid(axis='x', color='#e6e5e0', lw=0.6, zorder=0)
            for sp in ['top', 'right', 'left']:
                ax.spines[sp].set_visible(False)
            ax.spines['bottom'].set_color('#c3c2b7')
            if r == 0:
                ax.set_title(SUB_LABEL[sg], fontsize=7.5, color=INK)
        # row title above the first panel (the first row also carries subgroup titles)
        axes[r, 0].annotate(title, xy=(0, 1.0), xycoords='axes fraction', xytext=(-2, 24 if r == 0 else 14),
                            textcoords='offset points', fontsize=7.5, color=INK, fontweight='bold', ha='left')
    fig.tight_layout(h_pad=2.2, w_pad=0.6)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_gi(df, path):
    """GI residual Pearson vs GI residual MSE: shows that the Pearson metric rewards 'No change'."""
    models = [m for m in ORDER if m in df.model.unique() and m not in ('Train mean',)]
    d = df[df.gi_pearson_de20.notna() | df.gi_mse_de20.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    for ax, (metric, title) in zip(axes, [('gi_pearson_de20', 'GI residual Pearson $r$ (higher is better)'),
                                          ('gi_mse_de20', 'GI residual MSE (lower is better)')]):
        s = split_means(d, metric, by_sub=False)
        for k, m in enumerate(models):
            v = s[s.model == m][metric].to_numpy()
            v = v[np.isfinite(v)]
            y = len(models) - 1 - k
            if len(v):
                ax.scatter(v, np.full(len(v), y), s=9, color=COLOR[m], alpha=0.45, lw=0, zorder=2)
                ax.scatter([v.mean()], [y], s=34, color=COLOR[m], edgecolor='white', lw=1.2, zorder=3)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m.replace('Delta', r'$\Delta$') for m in models[::-1]], fontsize=6.5, color=INK)
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.tick_params(axis='x', labelsize=6.5, colors=INK2, length=2)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#e6e5e0', lw=0.6, zorder=0)
        for sp in ['top', 'right', 'left']:
            ax.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    df = pd.read_csv(RES)
    models = [m for m in ORDER if m in df.model.unique()]
    for metric in ['mse_de20', 'pearson_delta_de20', 'direction_de20', 'pearson_delta', 'gi_pearson_de20', 'gi_mse_de20']:
        summary(df, metric).to_csv(f'results/summary_{metric}.csv', index=False)
    comp = comparisons(df)
    comp.to_csv('results/comparisons.csv', index=False)
    tex = [table_main(df, [m for m in BASELINES + MAIN if m in models], 'mse_de20', 3,
                      r'MSE of predicted versus observed expression change on the top-20 DE genes '
                      r'(mean\,$\pm$\,s.d.\ over the five GEARS simulation splits; lower is better; best in bold)',
                      'tab:main'),
           table_main(df, [m for m in BASELINES + MAIN if m in models], 'pearson_delta_de20', 2,
                      r'Pearson correlation of predicted and observed expression change on the top-20 DE genes '
                      r'(mean\,$\pm$\,s.d.\ over five splits; higher is better)', 'tab:pearson'),
           table_main(df, [m for m in ['GCP-Mamba'] + ABLATIONS if m in models], 'mse_de20', 3,
                      r'Ablations: MSE on the top-20 DE genes (mean\,$\pm$\,s.d.\ over five splits)', 'tab:ablation')]
    open('manuscript/tables.tex', 'w').write('\n\n'.join(tex) + '\n')
    fig_benchmark(df, 'manuscript/fig_benchmark.pdf')
    fig_gi(df, 'manuscript/fig_gi.pdf')
    print(summary(df, 'mse_de20').pivot(index='model', columns='subgroup', values='mean').round(4).to_string())
    print(comp.round(4).to_string())


if __name__ == '__main__':
    main()
