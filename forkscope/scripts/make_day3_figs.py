"""Day-3 demo figures from replay JSONs (light theme, notebook use).

fig_t4_mirror.png       — t4: same o_0, one decision step, two destinies
fig_t7_localization.png — t7: fork is NOT at step 0; replay localizes it at d=1
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"

# dataviz reference palette (validated, fixed slot order)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CAT = {"correct": "#2a78d6", "intdiv_1.000": "#eb6834",
       "wrongtable_1.262": "#1baf7a", "wrong_other": "#eda100",
       "wrong_unit_seconds": "#eb6834", "no_answer": "#eda100"}
LABEL = {"correct": "correct (1.701)", "intdiv_1.000": "intdiv (1.000)",
         "wrongtable_1.262": "wrong table (1.262)", "wrong_other": "other wrong"}

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "PingFang SC", "Hiragino Sans GB"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 1.0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 200,
})


def load(name: str) -> dict:
    return json.load(open(REPORTS / name))


def fig_t4_mirror() -> None:
    runs = [
        (load("replay_t4_artist_album_ratio_sql_intdiv_10000.json"),
         "branch: COUNT(*) / COUNT(DISTINCT ArtistId)\n(integer-division trap — 46% of episodes)"),
        (load("replay_t4_artist_album_ratio_sql_avg_groupby_10015.json"),
         "branch: AVG(...) GROUP BY\n(correct — an 18% minority choice)"),
    ]
    order = ["correct", "intdiv_1.000", "wrongtable_1.262", "wrong_other"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=True)
    for ax, (run, subtitle) in zip(axes, runs):
        for x, step in enumerate(run["steps"][:2]):
            k = step["k_ok"]
            bottom = 0.0
            for cat in order:
                frac = step["o_d"].get(cat, 0) / k
                if frac == 0:
                    continue
                ax.bar(x, frac, width=0.56, bottom=bottom, color=CAT[cat],
                       edgecolor=SURFACE, linewidth=2)
                if frac >= 0.07:  # relief rule: direct labels
                    ax.text(x, bottom + frac / 2, f"{frac:.0%}",
                            ha="center", va="center", fontsize=10,
                            color=SURFACE if frac > 0.12 else INK)
                bottom += frac
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["before first SQL\n(d=0)", "after SQL + result\n(d=1)"],
                           fontsize=9.5)
        ax.set_title(subtitle, fontsize=9.5, color=INK2, pad=8)
        ax.set_ylim(0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axes[0].set_ylabel("outcome share over 50 resampled continuations", fontsize=9.5)
    fig.suptitle("t4 · one decision step, two destinies — "
                 "same starting distribution (22% correct) collapses to 0% or 100%",
                 fontsize=11.5, x=0.5, y=1.02)
    fig.legend(handles=[Patch(facecolor=CAT[c], label=LABEL[c]) for c in order],
               loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_t4_mirror.png", bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)


def fig_t7_localization() -> None:
    fail = load("replay_t7_avg_track_len_min_sql_avg_ms_raw_10000.json")
    ok = load("replay_t7_avg_track_len_min_sql_avg_ms_raw_10003.json")
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for run, color, label in [(ok, "#2a78d6", "episode that recovered (seed 10003)"),
                              (fail, "#eb6834", "episode that failed (seed 10000)")]:
        xs = [s["d"] for s in run["steps"]]
        ys = [s["p_correct"] for s in run["steps"]]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        ax.annotate(f"{ys[-1]:.0%}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(10, -4), fontsize=10, color=INK)
    ax.annotate("every resample writes the same SQL\n(persistence 1.00 — not a fork)",
                (0, 0.08), textcoords="offset points", xytext=(6, 52),
                fontsize=9, color=INK2)
    ax.annotate("THE fork: only ~10% call the calculator to convert ms → minutes;\n"
                "the rest answer seconds as if they were minutes",
                (1, 0.14), textcoords="offset points", xytext=(30, -26),
                fontsize=9, color=INK2)
    ax.annotate("conversion result in context\n→ outcome locked correct",
                (2, 1.0), textcoords="offset points", xytext=(14, -40),
                fontsize=9, color=INK2)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["d=0\nbefore SQL", "d=1\nafter SQL result",
                        "d=2\nafter ms→min calc", "d=3\nafter round()",
                        "d=4\nbefore answer"], fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("p(correct) over 50 resampled continuations", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title("t7 · the fork is not at step 0 — decision-step replay "
                 "localizes it at the unit-conversion choice (d=1)",
                 fontsize=11.5, pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_t7_localization.png", bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    fig_t4_mirror()
    fig_t7_localization()
    print("wrote", REPORTS / "fig_t4_mirror.png")
    print("wrote", REPORTS / "fig_t7_localization.png")
