"""Stacked-area o_t visualization with fork point annotations."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def stacked_area(
    curves: dict[str, np.ndarray],
    positions: np.ndarray,
    categories: list[str],
    forks: list[int] | None = None,
    base_strs: list[str] | None = None,
    out_path: str = "o_t.png",
    title: str = "",
):
    """curves: {label: (T,K)} plotted as vertical panels sharing x."""
    n = len(curves)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    for ax, (label, curve) in zip(axes, curves.items()):
        ax.stackplot(positions, curve.T, labels=categories, colors=[cmap(i) for i in range(curve.shape[1])], alpha=0.85)
        ax.set_ylabel(label)
        ax.set_ylim(0, 1)
        if forks:
            for f in forks:
                ax.axvline(f, color="red", ls="--", lw=1, alpha=0.8)
        ax.legend(loc="upper right", ncol=len(categories), fontsize=8)
    if base_strs and forks:
        ax = axes[0]
        for f in forks:
            idx = int(np.searchsorted(positions, f))
            if idx < len(base_strs):
                ax.annotate(base_strs[idx][:20], (f, 1.0), rotation=90, fontsize=6, va="bottom")
    axes[-1].set_xlabel("generation position (token)")
    if title:
        axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
