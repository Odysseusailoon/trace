"""Token-level FPA results — the o_t curve over a real reasoning trace.

Curve anchors are the measured before/after distributions of the 11 forks in
data/reports/report_virology_5.json (linear between anchors, noted on screen).
College-physics cameo uses its largest measured fork (t=288, TVD 0.50).
Render: ~/.venvs/manim/bin/manim -qh token_forks_anim.py TokenForks
"""
import json
from pathlib import Path
import numpy as np
from manim import *

C_B = "#4ea8de"      # option B
C_C = "#e4572e"      # option C
C_O = "#5a5f68"      # other mass
C_ACCENT = "#ffd166"
CODE_FONT = "Menlo"

REPORT = Path(__file__).resolve().parents[1] / "data/reports/report_virology_5.json"


def load_anchors():
    d = json.loads(REPORT.read_text())
    anchors = []          # (t, pB, pC)
    for f in d["forks"]:
        anchors.append((f["t"], f["before"].get("B", 0), f["before"].get("C", 0)))
        anchors.append((f["t_next"], f["after"].get("B", 0), f["after"].get("C", 0)))
    anchors.sort()
    ts = np.array([a[0] for a in anchors], dtype=float)
    pb = np.array([a[1] for a in anchors])
    pc = np.array([a[2] for a in anchors])
    grid = np.arange(ts.min(), 625, 4.0)
    return grid, np.interp(grid, ts, pb), np.interp(grid, ts, pc), d["forks"]


class TokenForks(Scene):
    def construct(self):
        grid, pb, pc, forks = load_anchors()

        # ---------- Act 1: how to read o_t ----------
        title = Text("Token by token: o_t over a real reasoning trace",
                     font_size=32, color=C_ACCENT).to_edge(UP, buff=0.5)
        sub = Text("MMLU virology · Qwen3-8B · resampled at 51 positions",
                   font_size=21, color=GREY_B).next_to(title, DOWN, buff=0.2)
        self.play(Write(title), FadeIn(sub), run_time=1.2)

        ax = Axes(x_range=[24, 624, 100], y_range=[0, 1, 0.5],
                  x_length=11.0, y_length=3.4,
                  axis_config={"color": GREY_C, "include_ticks": False},
                  ).move_to(DOWN * 0.55)
        xl = Text("position t in the chain of thought", font_size=19, color=GREY_B)
        xl.next_to(ax, DOWN, buff=0.25)
        yl = Text("answer share", font_size=19, color=GREY_B)
        yl.rotate(PI / 2).next_to(ax, LEFT, buff=0.25)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=0.9)

        def band(y_lo, y_hi, color):
            pts_top = [ax.c2p(t, hi) for t, hi in zip(grid, y_hi)]
            pts_bot = [ax.c2p(t, lo) for t, lo in zip(grid, y_lo)]
            poly = Polygon(*(pts_top + pts_bot[::-1]), stroke_width=0,
                           fill_color=color, fill_opacity=0.75)
            return poly

        zeros = np.zeros_like(pb)
        b_band = band(zeros, pb, C_B)
        c_band = band(pb, pb + pc, C_C)
        o_band = band(pb + pc, np.ones_like(pb), C_O)
        legend = VGroup(*[VGroup(Square(0.22, fill_color=c, fill_opacity=0.85,
                                        stroke_width=0),
                                 Text(s, font_size=18, color=GREY_A)
                                 ).arrange(RIGHT, buff=0.12)
                          for c, s in [(C_B, "B"), (C_C, "C"), (C_O, "other")]])
        legend.arrange(RIGHT, buff=0.45).next_to(ax, UP, buff=0.2).align_to(ax, RIGHT)
        note = Text("measured fork anchors, linear between", font_size=15, color=GREY_D)
        note.to_corner(DR, buff=0.35)
        self.play(FadeIn(b_band), FadeIn(c_band), FadeIn(o_band),
                  FadeIn(legend), FadeIn(note), run_time=1.4)
        self.wait(0.8)

        # ---------- Act 2: scan and hit forks ----------
        tv = ValueTracker(float(grid[0]))
        scan = always_redraw(lambda: Line(
            ax.c2p(tv.get_value(), 0), ax.c2p(tv.get_value(), 1),
            color=WHITE, stroke_width=2.5))
        self.add(scan)

        stops = [forks[4], forks[8]]     # t=216 TVD .30, t=420 TVD .27
        snippets = [
            '"…the options are a bit simplified.\n The question is asking which segment…"',
            '"Wait, maybe the question is phrased\n as which group is necessary…"',
        ]
        for fk, sn in zip(stops, snippets):
            self.play(tv.animate.set_value(fk["t"]), run_time=1.6,
                      rate_func=rate_functions.ease_in_out_sine)
            mark = DashedLine(ax.c2p(fk["t"], 0), ax.c2p(fk["t"], 1),
                              color=C_ACCENT, stroke_width=2.5)
            btxt = Text(sn, font_size=19, color=WHITE, line_spacing=0.9)
            card = SurroundingRectangle(btxt, corner_radius=0.1, buff=0.2,
                                        stroke_color=C_ACCENT, stroke_width=1.5,
                                        fill_color=BLACK, fill_opacity=0.85)
            g = VGroup(card, btxt)
            x_frac = (fk["t"] - 24) / 600
            g.next_to(ax.c2p(fk["t"], 1), UP, buff=0.15)
            if x_frac > 0.55:
                g.shift(LEFT * 1.5)
            tvd = Text(f"flip TVD = {fk['tvd']:.2f}", font_size=18, color=C_ACCENT)
            tvd.next_to(g, DOWN, buff=0.12).align_to(g, LEFT)
            self.play(Create(mark), FadeIn(g), FadeIn(tvd), run_time=0.9)
            self.wait(1.2)
            self.play(FadeOut(g), FadeOut(tvd), run_time=0.5)

        self.play(tv.animate.set_value(float(grid[-1])), run_time=1.2)
        all_marks = VGroup(*[DashedLine(ax.c2p(f["t"], 0), ax.c2p(f["t"], 1),
                                        color=C_ACCENT, stroke_width=1.5,
                                        stroke_opacity=0.7) for f in forks])
        count = Text("raw ε=0.10 flags 11 forks — but look at the shape",
                     font_size=23, color=WHITE).to_edge(DOWN, buff=0.5)
        self.play(Create(all_marks), FadeOut(xl), FadeIn(count), run_time=1.1)
        self.wait(1.2)

        verdict = Text("B ↔ C oscillation, no landing  →  drift, not decisions",
                       font_size=25, color=C_ACCENT).to_edge(DOWN, buff=0.5)
        self.play(ReplacementTransform(count, verdict), run_time=0.8)
        self.wait(1.6)

        # ---------- Act 3: college-physics cameo ----------
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        h3 = Text("It happens mid-sentence.", font_size=30,
                  color=C_ACCENT).to_edge(UP, buff=0.6)
        quote = Text('"…in the same orbital, their spins must be opposite. That\n'
                     ' would mean that the para state…"',
                     font_size=22, color=WHITE, line_spacing=0.9)
        src_l = Text("MMLU college physics · fork at t=288", font_size=19, color=GREY_B)
        before = VGroup(Text("before", font_size=19, color=GREY_B),
                        Text("B 0.75", font_size=26, color=C_B)).arrange(DOWN, buff=0.15)
        after = VGroup(Text("after", font_size=19, color=GREY_B),
                       Text("Other 0.55", font_size=26, color=C_O)).arrange(DOWN, buff=0.15)
        arrow = Arrow(LEFT * 0.8, RIGHT * 0.8, stroke_width=4, color=C_ACCENT)
        flip = VGroup(before, arrow, after).arrange(RIGHT, buff=0.7)
        tvd50 = Text("TVD 0.50 — the largest measured flip", font_size=21, color=C_ACCENT)
        body = VGroup(quote, src_l, flip, tvd50).arrange(DOWN, buff=0.45)
        body.next_to(h3, DOWN, buff=0.6)
        self.play(FadeIn(h3), Write(quote), run_time=1.4)
        self.play(FadeIn(src_l), FadeIn(flip), FadeIn(tvd50), run_time=1.0)
        self.wait(1.6)

        # ---------- Act 4: the contrast ----------
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        l1 = Text("MCQ, token level:  forks oscillate — drift.",
                  font_size=28, color=C_B)
        l2 = Text("Decision forks are rare here (~1% of 201 questions).",
                  font_size=22, color=GREY_B)
        l3 = Text("Agent tool tasks:  forks lock.   0.22 → 0.00 / 1.00",
                  font_size=28, color=C_ACCENT)
        l4 = Text("FPA's natural habitat is the agent stack.",
                  font_size=26, color=WHITE)
        g = VGroup(l1, l2, l3, l4).arrange(DOWN, buff=0.5).move_to(ORIGIN)
        self.play(FadeIn(l1), FadeIn(l2), run_time=1.0)
        self.play(FadeIn(l3), run_time=0.9)
        self.play(Write(l4), run_time=1.0)
        self.wait(1.8)
