"""Fig 2 — t7 localization: fork is mid-trajectory; same SQL, same result, only wording differs."""
import json
from _style import *

R = '../data/reports'
succ = json.load(open(f'{R}/replay_t7_avg_track_len_min_sql_avg_ms_raw_10003.json'))
fail = json.load(open(f'{R}/replay_t7_avg_track_len_min_sql_avg_ms_raw_10000.json'))

d_s = [s['d'] for s in succ['steps']][:3]          # 0,1,2
p_s = [s['p_correct'] for s in succ['steps']][:3]  # 0.08, 0.14, 1.00
d_f = [s['d'] for s in fail['steps']]
p_f = [s['p_correct'] for s in fail['steps']]      # 0.08, 0.00

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_position([0.2, 0.2, 0.666, 0.333])

ax.plot(d_s, p_s, color=C_CORRECT, linewidth=2, zorder=2)
ax.scatter(d_s, p_s, color=C_CORRECT, s=100, marker='o', linewidth=1.5, zorder=3)
ax.plot(d_f, p_f, color=C_ERROR, linewidth=2, zorder=2)
ax.scatter(d_f, p_f, color=C_ERROR, s=100, marker='o', linewidth=1.5, zorder=3)

# fork marker at d=1
ax.axvline(1, color=C_NEUTRAL, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
ax.annotate('fork: same SQL, same 393599 ms in context.\nOnly ~10% of continuations call calculator( / 60000)\n$-$ the rest answer in seconds',
            xy=(1, 0.14), xytext=(0.06, 0.78), ha='left',
            fontproperties=font_properties_annotate, color=C_INK,
            arrowprops=dict(arrowstyle='-', color=C_NEUTRAL, linewidth=0.8,
                            connectionstyle='arc3,rad=-0.15'))

ax.annotate('conversion lands in context\n$\\rightarrow$ locked correct', xy=(2, 1.0),
            xytext=(1.92, 1.02), ha='right', va='top',
            fontproperties=font_properties_annotate, color=C_CORRECT)
ax.annotate('answers 393.6 "minutes"\n(seconds, unconverted)', xy=(1, 0.0),
            xytext=(0.6, -0.16), ha='center',
            fontproperties=font_properties_annotate, color=C_ERROR)

ax.set_xlim(-0.15, 2.3)
ax.set_ylim(-0.28, 1.18)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['d = 0\nbefore SQL', 'd = 1\nSQL result in context', 'd = 2\nafter unit conversion'])
ax.set_yticks([0, 0.08, 0.5, 1.0])
format_axis_labels(ax, 'decision step', 'P(correct final answer)')
despine(ax)
ax.set_title('The fork can sit mid-trajectory  ·  t7, 50 replays per step',
             fontproperties=font_properties_label, color=C_INK, pad=10)

save_plot('plot2_t7_localization')
