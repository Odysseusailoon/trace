"""forkscope demo animation — t4 fork flip, all numbers from measured data.

Data sources (forkscope/data/reports/):
  replay_t4_..._sql_intdiv_10000.json      d=0: o_d={correct:11, intdiv:29, other:9, table:1}/50, p=0.22
                                           d=1 after sql:intdiv      -> {intdiv:50}/50,  p=0.00
  replay_t4_..._sql_avg_groupby_10015.json d=1 after sql:avg_groupby -> {correct:50}/50, p=1.00
  validation_t4_...json                    200 episodes: intdiv cluster 91 @ 0% correct

Render:  ~/.venvs/manim/bin/manim -qh fork_anim.py ForkFlip
"""
import random
from manim import *

C_OK = "#2fbf71"      # correct
C_BAD = "#e4572e"     # intdiv (silent failure)
C_OTH = "#8a8f98"     # wrong_other
C_TBL = "#e0b400"     # wrongtable
C_ACCENT = "#ffd166"
CODE_FONT = "Menlo"

# measured d=0 outcome counts (50 resamples, shared prefix)
D0 = [(C_OK, 11), (C_BAD, 29), (C_OTH, 9), (C_TBL, 1)]


def step_chip(label, color=WHITE, w=1.9, h=0.62, fs=22, font=None):
    box = RoundedRectangle(corner_radius=0.12, width=w, height=h,
                           stroke_color=color, stroke_width=2, fill_opacity=0.08,
                           fill_color=color)
    kw = {"font": font} if font else {}
    txt = Text(label, font_size=fs, color=color, **kw)
    if txt.width > w - 0.25:
        txt.scale_to_fit_width(w - 0.25)
    return VGroup(box, txt.move_to(box))


def stacked_bar(counts, height=2.6, width=0.55):
    total = sum(n for _, n in counts)
    parts = VGroup()
    for color, n in counts:
        h = height * n / total
        parts.add(Rectangle(width=width, height=max(h, 0.001), fill_color=color,
                            fill_opacity=0.9, stroke_width=1, stroke_color=BLACK))
    parts.arrange(UP, buff=0).move_to(ORIGIN)
    return parts


class ForkFlip(Scene):
    def construct(self):
        random.seed(4)

        # ---------- Act 1: a failed trajectory ----------
        title = Text("An agent fails.  Which step caused it?", font_size=34)
        title.to_edge(UP, buff=0.6)
        self.play(Write(title), run_time=1.2)

        labels = ["task", "think", "SQL", "result", "answer"]
        chips = VGroup(*[step_chip(s, w=1.55) for s in labels]).arrange(RIGHT, buff=0.75)
        arrows = VGroup(*[Arrow(chips[i].get_right(), chips[i + 1].get_left(),
                                buff=0.08, stroke_width=3, color=GREY_B)
                          for i in range(len(chips) - 1)])
        chain = VGroup(chips, arrows).move_to(ORIGIN)
        wrong = Text("✗ 1.0   (gold: 1.701)", font_size=26, color=C_BAD)
        wrong.next_to(chips[-1], DOWN, buff=0.35)
        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.3) for c in chips],
                              lag_ratio=0.15, run_time=1.6),
                  LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.15, run_time=1.6))
        self.play(FadeIn(wrong, shift=UP * 0.2), run_time=0.7)
        self.wait(0.6)

        # log-reading strawman
        strawman = Text("Reading the log locates the step 14% of the time.",
                        font_size=24, color=GREY_B).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(strawman), run_time=0.8)
        self.wait(1.0)

        # ---------- Act 2: measure — resample at the decision point ----------
        answer = Text("Measure it instead: resample.", font_size=30, color=C_ACCENT)
        answer.to_edge(UP, buff=0.6)
        self.play(FadeOut(title), FadeOut(strawman), FadeOut(wrong),
                  FadeIn(answer), run_time=0.9)

        # keep only prefix up to the SQL decision, move left
        keep = VGroup(chips[0], chips[1], arrows[0], arrows[1])
        drop = VGroup(chips[2], chips[3], chips[4], arrows[2], arrows[3])
        self.play(FadeOut(drop), keep.animate.to_edge(LEFT, buff=0.5), run_time=0.9)
        fork_pt = keep.get_right() + RIGHT * 0.15

        # fan out 50 continuations with measured outcome colors
        colors = [c for c, n in D0 for _ in range(n)]
        random.shuffle(colors)
        fans, dots = VGroup(), VGroup()
        x_end = fork_pt[0] + 4.2
        for i, col in enumerate(colors):
            y = np.interp(i, [0, 49], [-2.6, 2.6]) + random.uniform(-0.04, 0.04)
            end = np.array([x_end, y, 0])
            mid = np.array([fork_pt[0] + 2.0, y * 0.55, 0])
            path = CubicBezier(fork_pt, fork_pt + RIGHT * 1.1, mid, end)
            path.set_stroke(col, width=1.4, opacity=0.45)
            fans.add(path)
            dots.add(Dot(end, radius=0.045, color=col))
        tag = Text("resample ×50, same prefix", font_size=22, color=GREY_B)
        tag.next_to(fans, DOWN, buff=0.3)
        self.play(LaggedStart(*[Create(p) for p in fans], lag_ratio=0.015, run_time=2.2),
                  FadeIn(tag), run_time=2.2)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots],
                              lag_ratio=0.01, run_time=0.8))
        self.wait(0.4)

        # collapse into the measured distribution bar
        bar = stacked_bar(D0).move_to(np.array([x_end + 1.25, 0, 0]))
        p_lab = Text("P(success) = 0.22", font_size=26, color=C_OK)
        o_lab = Text("o_d  — measured outcome distribution", font_size=22, color=GREY_B)
        p_lab.next_to(bar, UP, buff=0.3)
        o_lab.next_to(bar, DOWN, buff=0.3)
        self.play(FadeOut(fans), Transform(dots, bar), run_time=1.2)
        self.play(FadeIn(p_lab), FadeIn(o_lab), run_time=0.7)
        self.wait(1.0)

        # ---------- Act 3: the fork flip (mirror pair) ----------
        head = Text("The next token decides.", font_size=30, color=C_ACCENT)
        head.to_edge(UP, buff=0.6)
        self.play(FadeOut(answer), FadeOut(p_lab), FadeOut(o_lab),
                  FadeOut(dots), FadeOut(tag), FadeIn(head), run_time=0.9)
        fork_pt = keep.get_right() + RIGHT * 0.15

        good = step_chip("AVG(n) GROUP BY artist", color=C_OK, w=3.6, h=0.6,
                         fs=20, font=CODE_FONT)
        bad = step_chip("COUNT(al)/COUNT(ar)", color=C_BAD, w=3.6, h=0.6,
                        fs=20, font=CODE_FONT)
        good.move_to(fork_pt + RIGHT * 2.6 + UP * 1.7)
        bad.move_to(fork_pt + RIGHT * 2.6 + DOWN * 1.7)
        e_good = Line(fork_pt, good.get_left(), color=C_OK, stroke_width=3, buff=0.1)
        e_bad = Line(fork_pt, bad.get_left(), color=C_BAD, stroke_width=3, buff=0.1)
        self.play(Create(e_good), Create(e_bad),
                  FadeIn(good, shift=UR * 0.2), FadeIn(bad, shift=DR * 0.2), run_time=1.1)

        bar_g = stacked_bar([(C_OK, 50)], height=1.9).next_to(good, RIGHT, buff=0.8)
        bar_b = stacked_bar([(C_BAD, 50)], height=1.9).next_to(bad, RIGHT, buff=0.8)
        lab_g = VGroup(Text("0.22 → 1.00", font_size=24, color=C_OK),
                       Text("50/50 correct", font_size=20, color=C_OK)
                       ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        lab_b = VGroup(Text("0.22 → 0.00", font_size=24, color=C_BAD),
                       Text("50/50 intdiv", font_size=20, color=C_BAD)
                       ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        lab_g.next_to(bar_g, RIGHT, buff=0.35)
        lab_b.next_to(bar_b, RIGHT, buff=0.35)
        self.play(GrowFromEdge(bar_g, DOWN), GrowFromEdge(bar_b, DOWN), run_time=1.0)
        self.play(FadeIn(lab_g), FadeIn(lab_b), run_time=0.7)
        note = Text("silent failure: intdiv returns 1.0 — no error, no recovery signal",
                    font_size=22, color=GREY_B).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(note), run_time=0.7)
        self.wait(1.6)

        # 200-episode stamp
        stamp = Text("200 episodes: 91 took the red path — 0 recovered.",
                     font_size=26, color=C_BAD).to_edge(DOWN, buff=0.55)
        self.play(ReplacementTransform(note, stamp), run_time=0.8)
        self.wait(1.4)

        # ---------- Act 4: end card ----------
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        t1 = Text("forkscope", font_size=54, color=C_ACCENT)
        t2 = Text("executed counterfactuals for agent trajectories", font_size=28)
        t3 = Text("measure, don't guess", font_size=26, color=GREY_B)
        t4 = Text("SGLang RadixAttention · 98.5% prefix cache hit",
                  font_size=20, color=GREY_C)
        card = VGroup(t1, t2, t3, t4).arrange(DOWN, buff=0.45).move_to(ORIGIN)
        self.play(FadeIn(t1, scale=1.1), run_time=0.8)
        self.play(Write(t2), run_time=1.0)
        self.play(FadeIn(t3), FadeIn(t4), run_time=0.8)
        self.wait(1.5)
