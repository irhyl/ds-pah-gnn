import os
import pandas as pd
import matplotlib.pyplot as plt

inpath = os.path.join('artifacts', 'optimization_results.csv')
outdir = os.path.join('artifacts', 'figures')
os.makedirs(outdir, exist_ok=True)

df = pd.read_csv(inpath)
if 'Predicted_Loss' not in df.columns:
    raise SystemExit('Predicted_Loss column not found')

# Sort by predicted loss
df_sorted = df.sort_values('Predicted_Loss').reset_index(drop=True)
best_loss = df_sorted.loc[0, 'Predicted_Loss']
mean_loss = df['Predicted_Loss'].mean()
imp_vs_mean = (mean_loss - best_loss) / mean_loss * 100.0

# Minimal, Times New Roman styling
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'axes.grid': False,
})

fig, ax = plt.subplots(figsize=(8, 3.2))

# primary line
ax.plot(df_sorted.index + 1, df_sorted['Predicted_Loss'], color='#2c7fb8', linewidth=1.2, alpha=0.95)

# sparse markers to reduce clutter
step = max(1, len(df_sorted) // 120)
ax.scatter(df_sorted.index[::step] + 1, df_sorted['Predicted_Loss'][::step], color='#2c7fb8', s=10, alpha=0.9)

# highlight top-k
topk = 10
ax.scatter(df_sorted.index[:topk] + 1, df_sorted['Predicted_Loss'][:topk], color='#e34a33', s=50, edgecolors='white', linewidths=0.6, zorder=5, label=f'Top {topk}')

ax.set_xlabel('Rank (1 = best)')
ax.set_ylabel('Predicted Loss')

# subtle horizontal grid lines
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.25)
ax.set_axisbelow(True)

# tidy spines for minimal look
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(0.8)

# annotation box
txt = f'Best = {best_loss:.4f}\nMean = {mean_loss:.4f}\nImp vs mean = {imp_vs_mean:.1f}\%'
ax.text(0.02, 0.96, txt, transform=ax.transAxes, fontsize=10, va='top', bbox=dict(facecolor='white', edgecolor='#dddddd', boxstyle='round', alpha=0.9))

ax.set_xlim(1, len(df_sorted))
ax.tick_params(axis='both', which='both', length=4)
ax.legend(frameon=False)
plt.tight_layout()
outpath = os.path.join(outdir, 'optimization_ranking.png')
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print('Saved plot to', outpath)
