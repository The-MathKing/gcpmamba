"""Tables and figures for the manuscript, computed only from results/*.csv.

Usage: python -m gcpm.report
Writes results/summary_*.csv, manuscript/tables.tex and manuscript/fig_*.pdf.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

RES = 'results/norman_per_condition.csv'
SUBGROUPS = ['unseen_single', 'combo_seen0', 'combo_seen1', 'combo_seen2']
SUB_LABEL = {'unseen_single': 'Unseen single', 'combo_seen0': 'Double, 0/2 seen',
             'combo_seen1': 'Double, 1/2 seen', 'combo_seen2': 'Double, 2/2 seen', 'all': 'All test'}
BASELINES = ['No change', 'Train mean', 'Additive', 'Linear (PCA)', 'Linear (ctrl emb.)']
MAIN = ['GCP-Mamba', 'GEARS']
ABLATIONS = ['GCP-Mamba (Delta only)', 'GCP-Mamba (Delta only, w_c~N(0,1))', 'GCP-Mamba (perm. graph)',
             'GCP-Mamba (random order)', 'Mamba (no graph)', 'Graph-MLP (no scan)']
RESIDUAL = ['GCP-Mamba (additive prior)', 'Mamba (additive prior)']
ORDER = BASELINES + ['GEARS', 'GCP-Mamba'] + ABLATIONS + RESIDUAL
# validated categorical slots (dataviz reference palette, light mode): 1 blue, 2 orange; neutrals for baselines
COLOR = {m: '#8a8984' for m in BASELINES}
COLOR.update({m: '#2a78d6' for m in ['GCP-Mamba'] + ABLATIONS + RESIDUAL}, GEARS='#eb6834')
INK, INK2 = '#0b0b0b', '#52514e'


def gi_r2(df):
    """R^2 of the GI residual per (model, split), pooled over doubles whose two singles were both
    in training (combo_seen2), where the additive baseline equals the sum of measured singles:
    1 - sum MSE / sum mean(eps^2). 0 = exact additive prediction; > 0 = captures interactions."""
    d = df[df.gi_var_de20.notna() & (df.subgroup == 'combo_seen2')]
    g = d.groupby(['model', 'seed']).apply(
        lambda x: 1 - x.mse_de20.sum() / x.gi_var_de20.sum(), include_groups=False)
    return g.rename('gi_r2').reset_index()


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


def tex_name(m):
    return (m.replace('Delta', r'$\Delta$').replace('w_c~N(0,1)', r'$\mathbf{w}_c\sim\mathcal{N}(0,1)$')
            .replace('perm.', 'permuted'))


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
        lines.append(tex_name(m) + ' & ' + ' & '.join(cells) + r' \\')
        if m in (BASELINES[-1], 'GCP-Mamba', ABLATIONS[-1]) and m != models[-1]:
            lines.append(r'\midrule')
    lines += [r'\botrule', r'\end{tabular*}}{}', r'\end{table*}']
    return '\n'.join(lines)


def comparisons(df):
    """GCP-Mamba vs every other model, per metric and test subgroup; Holm within each family."""
    rows = []
    others = [m for m in BASELINES + ['GEARS'] + ABLATIONS + RESIDUAL if m in df.model.unique()]
    for metric in ['mse_de20', 'pearson_delta_de20']:
        for sg in ['all'] + SUBGROUPS:
            sub = df if sg == 'all' else df[df.subgroup == sg]
            res = [(m,) + paired_test(sub, 'GCP-Mamba', m, metric) for m in others]
            res = [r for r in res if r[1] > 0]
            padj = holm([r[3] if np.isfinite(r[3]) else 1.0 for r in res])
            for r, q in zip(res, padj):
                rows.append(dict(metric=metric, subgroup=sg, model_a='GCP-Mamba', model_b=r[0], n=r[1],
                                 median_diff=r[2], p=r[3], p_holm=q))
    return pd.DataFrame(rows)


def fig_benchmark(df, path):
    metrics = [('mse_de20', 'MSE, top-20 DE genes (lower is better)'),
               ('pearson_delta_de20', 'Pearson $r$ of $\\delta$, top-20 DE genes')]
    models = [m for m in FIG_MODELS if m in df.model.unique()]
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
            ax.set_yticklabels([tex_name(m) for m in models[::-1]], fontsize=6.5, color=INK)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(axis='x', labelsize=6.5, colors=INK2, length=2)
            ax.tick_params(axis='y', length=0)
            ax.grid(axis='x', color='#e6e5e0', lw=0.6, zorder=0)
            for sp in ['top', 'right', 'left']:
                ax.spines[sp].set_visible(False)
            ax.spines['bottom'].set_color('#c3c2b7')
            if r == 0:
                ax.set_title(SUB_LABEL[sg], fontsize=7.5, color=INK)
    fig.tight_layout(h_pad=3.0, w_pad=0.6, rect=(0, 0, 1, 0.97))
    for r, (_, title) in enumerate(metrics):   # row titles, left-aligned above each row
        top = axes[r, 0].get_position().y1
        fig.text(0.01, top + (0.045 if r == 0 else 0.012), title, fontsize=7.5, color=INK, fontweight='bold')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_gi(df, path):
    """GI Pearson r vs R^2_GI per split: the Pearson metric rewards trivial predictors."""
    models = list(dict.fromkeys([m for m in FIG_MODELS + ['No change', 'Train mean'] if m in df.model.unique()]))
    d = df[df.gi_var_de20.notna() & (df.subgroup == 'combo_seen2')]
    panels = [(split_means(d, 'gi_pearson_de20', by_sub=False), 'gi_pearson_de20',
               'GI residual Pearson $r$ (higher is better)'),
              (gi_r2(d), 'gi_r2', '$R^2_{\\mathrm{GI}}$ (higher is better; additive = 0)')]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    for ax, (s, metric, title) in zip(axes, panels):
        for k, m in enumerate(models):
            v = s[s.model == m][metric].to_numpy()
            v = v[np.isfinite(v)]
            y = len(models) - 1 - k
            if len(v):
                ax.scatter(v, np.full(len(v), y), s=9, color=COLOR[m], alpha=0.45, lw=0, zorder=2)
                ax.scatter([v.mean()], [y], s=34, color=COLOR[m], edgecolor='white', lw=1.2, zorder=3)
        if metric == 'gi_r2':
            ax.axvline(0, color=INK2, lw=0.7, ls='--', zorder=1)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([tex_name(m) for m in models[::-1]], fontsize=6.5, color=INK)
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(axis='x', labelsize=6.5, colors=INK2, length=2)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#e6e5e0', lw=0.6, zorder=0)
        for sp in ['top', 'right', 'left']:
            ax.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


FIG_MODELS = BASELINES[:4] + ['GEARS', 'GCP-Mamba', 'GCP-Mamba (perm. graph)', 'Mamba (no graph)',
                                'GCP-Mamba (additive prior)']


def table_gi(df, models):
    d = df[df.gi_var_de20.notna() & (df.subgroup == 'combo_seen2')]
    g1 = gi_r2(d).groupby('model').gi_r2.agg(['mean', 'std'])
    g2 = summary(d, 'gi_pearson_de20')
    g2 = g2[g2.subgroup == 'all'].set_index('model')
    lines = [r'\begin{table}[!t]', r'\processtable{Prediction of genetic interactions (GI) in double perturbations '
             r'(top-20 DE genes; mean\,$\pm$\,s.d.\ over five splits). $R^2_{\mathrm{GI}}$ is the fraction of '
             r'the variance of the GI residual $\boldsymbol{\epsilon}$ that a prediction explains; an exactly '
             r'additive prediction scores 0. The GI Pearson $r$ gives the trivial \emph{No change} and '
             r'\emph{Train mean} predictors scores close to those of the deep models.\label{tab:gi}}{',
             r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcc@{}}', r'\toprule',
             r'Model & $R^2_{\mathrm{GI}}$ $\uparrow$ & GI Pearson $r$ $\uparrow$ \\', r'\midrule']
    for m in models:
        if m in g1.index:
            lines.append(f"{tex_name(m)} & {fmt(g1.loc[m, 'mean'], g1.loc[m, 'std'], 2)} & "
                         f"{fmt(g2.loc[m, 'mean'], g2.loc[m, 'std'], 2)} \\\\")
    lines += [r'\botrule', r'\end{tabular*}}{}', r'\end{table}']
    return '\n'.join(lines)


def table_adamson(path):
    df = pd.read_csv(path)
    rows = []
    for metric, dig in (('mse_de20', 3), ('pearson_delta_de20', 2)):
        g = summary(df, metric)
        rows.append(g[g.subgroup == 'all'].set_index('model'))
    models = [m for m in ORDER if m in df.model.unique()]
    lines = [r'\begin{table}[!t]', r'\processtable{Adamson et al.\ CRISPRi screen, unseen single perturbations '
             r'(mean\,$\pm$\,s.d.\ over five GEARS simulation splits).\label{tab:adamson}}{',
             r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcc@{}}', r'\toprule',
             r'Model & MSE (top-20 DE) $\downarrow$ & Pearson $r$ (top-20 DE) $\uparrow$ \\', r'\midrule']
    for m in models:
        lines.append(f"{tex_name(m)} & {fmt(rows[0].loc[m, 'mean'], rows[0].loc[m, 'std'])} & "
                     f"{fmt(rows[1].loc[m, 'mean'], rows[1].loc[m, 'std'], 2)} \\\\")
    lines += [r'\botrule', r'\end{tabular*}}{}', r'\end{table}']
    return '\n'.join(lines)


def main():
    df = pd.read_csv(RES)
    models = [m for m in ORDER if m in df.model.unique()]
    gi_r2(df).to_csv('results/summary_gi_r2_per_split.csv', index=False)
    for metric in ['mse_de20', 'pearson_delta_de20', 'direction_de20', 'pearson_delta', 'gi_pearson_de20', 'gi_mse_de20']:
        summary(df, metric).to_csv(f'results/summary_{metric}.csv', index=False)
    comp = comparisons(df)
    comp.to_csv('results/comparisons.csv', index=False)
    main_models = [m for m in BASELINES + ['GEARS', 'GCP-Mamba'] if m in models]
    abl_models = [m for m in ['GCP-Mamba'] + ABLATIONS + RESIDUAL if m in models]
    tex = [table_main(df, main_models, 'mse_de20', 3,
                      r'Norman et al.\ CRISPRa screen: MSE of the predicted expression change on the top-20 DE genes '
                      r'of each test perturbation (mean\,$\pm$\,s.d.\ over the five GEARS simulation splits; lower is '
                      r'better; best in bold)', 'tab:main'),
           table_main(df, abl_models, 'mse_de20', 3,
                      r'Ablations and residual variants on the Norman data: MSE on the top-20 DE genes '
                      r'(mean\,$\pm$\,s.d.\ over five splits; best in bold)', 'tab:ablation'),
           table_gi(df, [m for m in BASELINES + ['GEARS', 'GCP-Mamba', 'Mamba (no graph)'] + RESIDUAL if m in models])]
    supp = [table_main(df, main_models + [m for m in ABLATIONS + RESIDUAL if m in models], 'pearson_delta_de20', 2,
                       r'Pearson correlation of predicted and observed expression change on the top-20 DE genes '
                       r'(mean\,$\pm$\,s.d.\ over five splits; higher is better)', 'tab:pearson')]
    if os.path.exists('results/adamson_per_condition.csv'):
        tex.append(table_adamson('results/adamson_per_condition.csv'))
    open('manuscript/tables.tex', 'w').write('\n\n'.join(tex) + '\n')
    open('manuscript/tables_supp.tex', 'w').write('\n\n'.join(supp) + '\n')
    fig_benchmark(df[df.model.isin(FIG_MODELS)], 'manuscript/fig_benchmark.pdf')
    fig_gi(df, 'manuscript/fig_gi.pdf')
    print(summary(df, 'mse_de20').pivot(index='model', columns='subgroup', values='mean').round(4).to_string())

if __name__ == '__main__':
    main()
