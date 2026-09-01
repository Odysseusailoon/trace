# Shared prelude per matplotlib-scientific skill (copied into namespace via import *)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

matplotlib.rcParams['mathtext.default'] = 'regular'

fs, fss, fsss, fsl = 12, 12, 8, 24
font_properties_label = fm.FontProperties(family='Arial', size=fs)
font_properties_tick = fm.FontProperties(family='Arial', size=fss)
font_properties_annotate = fm.FontProperties(family='Arial', size=fsss)
font_properties_legend = fm.FontProperties(family='Arial', size=fss)

# validated palette (dataviz six-checks, light surface): correct / error + neutral context
C_CORRECT = '#1273A6'
C_ERROR = '#C96442'
C_NEUTRAL = '#8A8778'
C_INK = '#3D3D3A'

os.makedirs('./output', exist_ok=True)


def apply_font_styling(ax):
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font_properties_tick)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font_properties_tick)


def format_axis_labels(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontproperties=font_properties_label)
    ax.set_ylabel(ylabel, fontproperties=font_properties_label)
    apply_font_styling(ax)


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(C_INK)
    ax.spines['bottom'].set_color(C_INK)
    ax.tick_params(colors=C_INK)


def save_plot(filename, dpi=300):
    if not filename.startswith('plot'):
        filename = f'plot1_{filename}'
    base = filename.rsplit('.', 1)[0]
    plt.savefig(f'./output/{base}.svg', dpi=dpi, bbox_inches='tight', format='svg')
    plt.savefig(f'./output/{base}.png', dpi=dpi, bbox_inches='tight', format='png',
                facecolor='white')
