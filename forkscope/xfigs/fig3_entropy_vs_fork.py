"""Fig 3 — token entropy vs causal fork: the fatal tokens are not the entropy peak."""
import json
from _style import *

R = '../data/reports'
ep = json.load(open(f'{R}/entropy_agent_raw.json'))['t4_10000']
toks = ep['turns'][0]['tokens']  # turn 0: the SQL tool call that locks the outcome

labels = [t['tok'] for t in toks]
H = np.array([t['H'] for t in toks])
n = len(toks)

# causal tokens: the integer-division construction
causal_idx = {i for i, t in enumerate(labels) if t in (' COUNT', '(*)', ' /', '(D')}
# decorative peak: the alias naming choice
peak_idx = {i for i, t in enumerate(labels) if t == ' Average'}

colors = []
for i in range(n):
    if i in causal_idx:
        colors.append(C_ERROR)
    elif i in peak_idx:
        colors.append(C_NEUTRAL)
    else:
        colors.append('#DDDAD2')

fig, ax = plt.subplots(figsize=(11, 9))
ax.set_position([0.2, 0.2, 0.666, 0.333])

ax.bar(np.arange(n), H, color=colors, width=0.82, zorder=2)

i_avg = list(peak_idx)[0]
ax.annotate('highest entropy in the whole trajectory:\n" Average" — an alias naming choice (H = 0.91)',
            xy=(i_avg, H[i_avg]), xytext=(i_avg + 2.0, 1.02), ha='left',
            fontproperties=font_properties_annotate, color=C_INK,
            arrowprops=dict(arrowstyle='-', color=C_NEUTRAL, linewidth=0.8))

i_count = min(causal_idx)
ax.annotate('the tokens that lock the failure:\nCOUNT(*) / COUNT(DISTINCT ...)  (H = 0.31$-$0.63)',
            xy=(i_count + 1.5, 0.66), xytext=(i_count - 16.2, 0.86), ha='left',
            fontproperties=font_properties_annotate, color=C_ERROR,
            arrowprops=dict(arrowstyle='-', color=C_ERROR, linewidth=0.8))

ax.set_xlim(-1.2, n + 0.2)
ax.set_ylim(0, 1.25)
ax.set_yticks([0, 0.5, 1.0])
ax.set_xticks([0, 16, i_avg, n - 1])
ax.set_xticklabels(['<tool_call>', 'SELECT', 'Average', ';'], rotation=0)
ax.annotate('most tokens: H $\\approx$ 0\n(near-deterministic)',
            xy=(1.2, 0.16), ha='left',
            fontproperties=font_properties_annotate, color=C_NEUTRAL)
format_axis_labels(ax, 'token position (turn that writes the fatal SQL)', 'token entropy H (nats, top-20)')
despine(ax)
ax.set_title('Entropy is a poor proxy for causal forks  ·  t4 failure trajectory, Qwen3-8B',
             fontproperties=font_properties_label, color=C_INK, pad=10)

save_plot('plot3_entropy_vs_fork')
