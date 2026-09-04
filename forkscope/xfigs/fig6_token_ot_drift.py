"""Fig 6 - token-level o_t: the answer mix oscillates and never lands (drift-level failure).

Points are the measured fork windows (TVD > 0.1, S = 20 continuations each) from
report_virology_5.json / report_college_physics_4.md; Forking Fast showed o_t is
flat between sparse forks, so the segments between points are near-constant.
"""
import json
import re
from _style import *

R = '../data/reports'

vir = json.load(open(f'{R}/report_virology_5.json'))
v_pts = []
for f in vir['forks']:
    v_pts.append((f['t'], f['before'].get('B', 0), f['before'].get('C', 0)))
    v_pts.append((f['t_next'], f['after'].get('B', 0), f['after'].get('C', 0)))
v_pts.sort()

md = open(f'{R}/report_college_physics_4.md').read()
raw = re.findall(r"## Fork \d+: t = (\d+) -> (\d+).*?- before: ({[^}]+})\n- after:  ({[^}]+})", md, re.S)
c_pts = []
for t, tn, b, a in raw:
    b, a = eval(b), eval(a)
    c_pts.append((int(t), b.get('B', 0), b.get('Other', 0)))
    c_pts.append((int(tn), a.get('B', 0), a.get('Other', 0)))
c_pts.sort()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
panels = [
    (axes[0], v_pts, 'answer B', 'answer C', 'virology_5  ·  11 measured forks'),
    (axes[1], c_pts, 'answer B', 'other answers', 'college_physics_4  ·  14 measured forks'),
]
axes[0].set_position([0.2, 0.57, 0.666, 0.30])
axes[1].set_position([0.2, 0.155, 0.666, 0.30])

for ax, pts, l0, l1, name in panels:
    t = [p[0] for p in pts]
    s0 = [p[1] for p in pts]
    s1 = [p[2] for p in pts]
    ax.axhline(0.5, color=C_NEUTRAL, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.plot(t, s0, color=C_CORRECT, linewidth=1.5, zorder=2)
    ax.scatter(t, s0, color=C_CORRECT, s=24, zorder=3)
    ax.plot(t, s1, color=C_ERROR, linewidth=1.5, zorder=2)
    ax.scatter(t, s1, color=C_ERROR, s=24, zorder=3)
    ax.annotate(l0, xy=(t[-1] + 12, s0[-1]), va='center', ha='left',
                fontproperties=font_properties_annotate, color=C_CORRECT)
    ax.annotate(l1, xy=(t[-1] + 12, s1[-1]), va='center', ha='left',
                fontproperties=font_properties_annotate, color=C_ERROR)
    ax.annotate(name, xy=(0.01, 1.05), xycoords='axes fraction', ha='left',
                fontproperties=font_properties_annotate, color=C_INK)
    ax.set_xlim(0, 660)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    format_axis_labels(ax, '', 'P(final answer)')
    despine(ax)

axes[0].set_title('Token-level o$_t$: the answer mix never lands  ·  drift, not a decision',
                  fontproperties=font_properties_label, color=C_INK, pad=22)
format_axis_labels(axes[1], 'generation position (token)', 'P(final answer)')
fig.text(0.2, 0.055, 'measured at fork windows (TVD > 0.1), S = 20 continuations each  ·  '
         'no single step to blame: the fix is sampling/aggregation, not a patch',
         fontproperties=font_properties_annotate, color=C_NEUTRAL)

save_plot('plot6_token_ot_drift')
