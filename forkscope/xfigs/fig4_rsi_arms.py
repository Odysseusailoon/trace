"""Fig 4 — RSI arms: what the agent is told determines whether it can fix itself (t4, 200 vs 200)."""
from _style import *

arms = ['control\n(no report)', 'warning-only note\n(location, no mechanism)', 'measured fork report\n(mechanism + fix direction)']
vals = [27.0, 15.5, 100.0]
cols = [C_NEUTRAL, C_ERROR, C_CORRECT]

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_position([0.2, 0.2, 0.666, 0.333])

y = np.arange(len(arms))[::-1]
ax.barh(y, vals, color=cols, height=0.62, zorder=2)

for yi, v in zip(y, vals):
    ax.annotate(f'{v:.1f}%', xy=(v + 2, yi), va='center', ha='left',
                fontproperties=font_properties_tick, color=C_INK)

ax.annotate('error shifts to another wrong table\n(whack-a-mole), accuracy drops',
            xy=(28, y[1]), va='center', ha='left',
            fontproperties=font_properties_annotate, color=C_ERROR)

ax.set_yticks(y)
ax.set_yticklabels(arms)
ax.set_xlim(0, 118)
ax.set_xticks([0, 27, 50, 100])
format_axis_labels(ax, 'final accuracy on t4 (200 episodes per arm)', '')
despine(ax)
ax.set_title('Agent patches its own tool description  ·  same model, only the diagnosis differs',
             fontproperties=font_properties_label, color=C_INK, pad=10)

save_plot('plot4_rsi_arms')
