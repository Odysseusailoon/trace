"""Fig 5 - cost: naive FPA prefill vs RadixAttention shared trunk (real token accounting)."""
import json
from _style import *

R = '../data/reports'
c = json.load(open(f'{R}/cost_table.json'))
rows = c['rows']
eps = [r['episode'] for r in rows] + ['total']
naive = [r['naive_prefill'] for r in rows]
anch = [r['anchored_prefill'] for r in rows]
naive.append(sum(naive))
anch.append(sum(anch))
ratios = [n / a for n, a in zip(naive, anch)]
labels = ['t4 fail (2 steps)', 't4 success (2 steps)', 't7 fail (2 steps)', 't7 success (5 steps)', 'all four']

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_position([0.2, 0.2, 0.666, 0.333])

y = np.arange(len(eps))[::-1]
h = 0.36
ax.barh(y + h / 2 + 0.02, naive, height=h, color=C_NEUTRAL, zorder=2)
ax.barh(y - h / 2 - 0.02, anch, height=h, color=C_CORRECT, zorder=2)

for yi, n, a, r in zip(y, naive, anch, ratios):
    ax.annotate(f'{n/1000:.0f}K', xy=(n * 1.15, yi + h / 2 + 0.02), va='center', ha='left',
                fontproperties=font_properties_annotate, color=C_NEUTRAL)
    ax.annotate(f'{a/1000:.1f}K   {r:.0f}x less', xy=(a * 1.15, yi - h / 2 - 0.02),
                va='center', ha='left',
                fontproperties=font_properties_annotate, color=C_CORRECT)

ax.set_xscale('log')
ax.set_xlim(300, 3.3e6)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.legend(['naive prefill (no KV reuse)', 'RadixAttention, 98.5% hit (measured)'],
          frameon=False, prop=font_properties_legend, loc='lower left',
          bbox_to_anchor=(0.0, 1.02), ncol=2, columnspacing=1.6, handlelength=1.4)
format_axis_labels(ax, 'prefill tokens for K = 50 replays per step (log scale)', '')
despine(ax)
ax.set_title('Counterfactual replay is practically free on SGLang  ·  exact token accounting',
             fontproperties=font_properties_label, color=C_INK, pad=32)

save_plot('plot5_cost')
