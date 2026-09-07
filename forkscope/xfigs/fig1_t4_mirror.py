"""Fig 1 — t4 mirror: same state, first SQL locks the outcome (K=50 replays)."""
import json
from _style import *

R = '../data/reports'
fail = json.load(open(f'{R}/replay_t4_artist_album_ratio_sql_intdiv_10000.json'))
succ = json.load(open(f'{R}/replay_t4_artist_album_ratio_sql_avg_groupby_10015.json'))

d = [0, 1]
p_fail = [s['p_correct'] for s in fail['steps']]   # 0.22 -> 0.00
p_succ = [s['p_correct'] for s in succ['steps']]   # 0.22 -> 1.00

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_position([0.2, 0.2, 0.666, 0.333])

ax.plot(d, p_succ, color=C_CORRECT, linewidth=2, zorder=2)
ax.scatter(d, p_succ, color=C_CORRECT, s=100, marker='o', linewidth=1.5, zorder=3)
ax.plot(d, p_fail, color=C_ERROR, linewidth=2, zorder=2)
ax.scatter(d, p_fail, color=C_ERROR, s=100, marker='o', linewidth=1.5, zorder=3)

# shared starting state
ax.scatter([0], [0.22], color=C_INK, s=100, zorder=4)
ax.annotate('same state\nP(correct) = 0.22', xy=(0, 0.22), xytext=(-0.02, 0.47),
            ha='left', fontproperties=font_properties_annotate, color=C_INK,
            arrowprops=dict(arrowstyle='-', color=C_NEUTRAL, linewidth=0.8))

ax.annotate('SELECT AVG(n) FROM (... GROUP BY ArtistId)', xy=(0.42, 0.78),
            ha='center', fontproperties=font_properties_annotate, color=C_CORRECT)
ax.annotate('locked correct  1.00', xy=(1, 1.0), xytext=(0.97, 1.08), ha='right',
            fontproperties=font_properties_annotate, color=C_CORRECT)
ax.annotate('COUNT(*) / COUNT(DISTINCT ArtistId)\n(integer division, returns 1, no error)',
            xy=(0.42, -0.02), ha='center', va='top',
            fontproperties=font_properties_annotate, color=C_ERROR)
ax.annotate('locked wrong  0.00', xy=(1, 0.0), xytext=(0.97, 0.08), ha='right',
            fontproperties=font_properties_annotate, color=C_ERROR)

ax.set_xlim(-0.15, 1.15)
ax.set_ylim(-0.22, 1.18)
ax.set_xticks([0, 1])
ax.set_xticklabels(['d = 0\nbefore first SQL', 'd = 1\nfirst SQL in context'])
ax.set_yticks([0, 0.22, 0.5, 1.0])
format_axis_labels(ax, 'decision step', 'P(correct final answer)')
despine(ax)
ax.set_title('One SQL statement locks the outcome  ·  50 full replays per step, tools executed',
             fontproperties=font_properties_label, color=C_INK, pad=10)

save_plot('plot1_t4_mirror')
