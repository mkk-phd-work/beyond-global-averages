import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

# Load verified extracted seed data
df = pd.read_csv('outputs/tables/02_seed_gap_stability_data.csv')

# Gap convention: conditional - global
# Positive gap: feature is MORE important for this group than globally
# Negative gap: feature is LESS important for this group than globally
df['gap'] = df['group_avg_importance'] - df['global_importance']

# Typography and aesthetics suitable for Springer LNCS proceedings
plt.rcParams.update({
    'font.size': 9.5,
    'axes.labelsize': 10.5,
    'axes.titlesize': 11,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.0,
    'font.family': 'sans-serif',
    'mathtext.fontset': 'dejavusans',
    'figure.autolayout': False,
})

fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=300)

palette = {
    'group_a': '#2b5c8f',  # Steel Navy
    'group_b': '#d95f02',  # Burnt Orange
}

features = ['exog_feature_1', 'exog_feature_2', 'exog_feature_3']
feature_display = [
    'exog_feature_1\n(Group A active)',
    'exog_feature_2\n(Group B active)',
    'exog_feature_3\n(Shared / uniform)'
]

offsets = {'group_a': -0.16, 'group_b': 0.16}
box_width = 0.24

for feat_idx, feat in enumerate(features):
    for grp in ['group_a', 'group_b']:
        sub = df[(df['feature'] == feat) & (df['group'] == grp)].sort_values('seed')
        vals = sub['gap'].values
        pos = feat_idx + offsets[grp]
        
        # Box plot showing quartiles, median, whiskers (NO duplicate flier glyphs)
        bp = ax.boxplot(
            vals,
            positions=[pos],
            widths=box_width,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            meanprops=dict(marker='^', markerfacecolor='white', markeredgecolor='black', markersize=5.5, zorder=5),
            medianprops=dict(color='black', linewidth=1.4, zorder=4),
            whiskerprops=dict(color=palette[grp], linewidth=1.2, alpha=0.9),
            capprops=dict(color=palette[grp], linewidth=1.2, alpha=0.9),
            boxprops=dict(facecolor=palette[grp], alpha=0.30, edgecolor=palette[grp], linewidth=1.4)
        )
        
        # Overlaid discrete seed points (deterministic jitter to prevent overlap)
        np.random.seed(100 + feat_idx * 10 + (0 if grp == 'group_a' else 1))
        jitter = np.random.uniform(-0.055, 0.055, size=len(vals))
        ax.scatter(
            pos + jitter,
            vals,
            color=palette[grp],
            s=26,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.6,
            zorder=6
        )

# Reference baseline: zero gap (global average)
ax.axhline(0, color='#555555', linestyle='--', linewidth=1.0, alpha=0.85, zorder=2)

# Styling and annotations
ax.set_xticks(range(len(features)))
ax.set_xticklabels(feature_display)
ax.set_ylabel('Importance Gap ($I_{\mathrm{cond}} - I_{\mathrm{global}}$)')
ax.grid(True, linestyle=':', alpha=0.5, axis='y', zorder=1)

# Set clean y-limits
ax.set_ylim(-0.40, 0.42)

# Legend
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=palette['group_a'], alpha=0.6, markersize=8, label='group_a ($n = 20$ seeds)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=palette['group_b'], alpha=0.6, markersize=8, label='group_b ($n = 20$ seeds)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='white', markeredgecolor='black', markersize=6, label='Mean across seeds'),
    Line2D([0], [0], color='#555555', linestyle='--', linewidth=1.0, label='Global baseline ($\Delta = 0$)')
]
ax.legend(handles=legend_elements, loc='upper right', framealpha=0.92, edgecolor='#cccccc')

plt.tight_layout()

# Save under both names for compatibility
out_paths = [
    'outputs/figures/02_seed_gap_stability_boxplot.pdf',
    'outputs/figures/02_seed_gap_stability_boxplot.png',
    'outputs/figures/02_seed_gap_stability_subplots.pdf',
    'outputs/figures/02_seed_gap_stability_subplots.png',
]

for p in out_paths:
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print(f'Saved {p}')

plt.close()
