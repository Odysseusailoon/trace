"""forkscope showcases — one scene per case, all numbers from measured data.

  ShowcaseT7  : mid-trajectory fork (wording modulates the next decision)
                replay: d0 p=0.08 both; d1 fail 0.00 / success 0.14; d2 -> 1.00
  ShowcaseT10 : decoy lock-in (187/200 copied 67.3 from the snippet; gold 67.5)

Render: ~/.venvs/manim/bin/manim -qh showcase_anim.py ShowcaseT7
"""
from manim import *

C_OK = "#2fbf71"; C_BAD = "#e4572e"; C_OTH = "#8a8f98"
C_ACCENT = "#ffd166"; C_BLUE = "#4ea8de"
CODE_FONT = "Menlo"


def card(lines, colors=None, fs=22, font=None, buff=0.18, pad=0.28, stroke=GREY_C):
    colors = colors or [WHITE] * len(lines)
    kw = {"font": font} if font else {}
    txts = VGroup(*[Text(s, font_size=fs, color=c, **kw)
                    for s, c in zip(lines, colors)]).arrange(DOWN, buff=buff,
                                                             aligned_edge=LEFT)
    box = SurroundingRectangle(txts, corner_radius=0.12, buff=pad,
                               stroke_color=stroke, stroke_width=1.5,
                               fill_color=GREY_E, fill_opacity=0.25)
    return VGroup(box, txts)


class ShowcaseT7(Scene):
    def construct(self):
        title = Text("Showcase · t7: the fork is mid-trajectory",
                     font_size=32, color=C_ACCENT).to_edge(UP, buff=0.5)
        task = Text('"Average track length in minutes?"   gold: 6.56',
                    font_size=24, color=GREY_A).next_to(title, DOWN, buff=0.25)
        self.play(Write(title), FadeIn(task), run_time=1.2)

        # step 1: everyone writes the same SQL
        sql = card(["SELECT AVG(Milliseconds) FROM Track;"], [WHITE],
                   fs=21, font=CODE_FONT)
        sql.move_to(UP * 0.9)
        same = Text("98% of 200 episodes write exactly this — no fork here",
                    font_size=22, color=GREY_B).next_to(sql, DOWN, buff=0.25)
        self.play(FadeIn(sql), FadeIn(same), run_time=0.9)
        ret = card(["→ 393599.21   (milliseconds)"], [C_BLUE], fs=21, font=CODE_FONT)
        ret.next_to(same, DOWN, buff=0.35)
        self.play(FadeIn(ret, shift=UP * 0.2), run_time=0.7)
        self.wait(1.0)

        # step 2: the p_correct curve — two wordings diverge
        self.play(FadeOut(sql), FadeOut(same), FadeOut(ret), run_time=0.8)
        ax = Axes(x_range=[0, 2, 1], y_range=[0, 1, 0.5],
                  x_length=6.0, y_length=3.2,
                  axis_config={"color": GREY_C, "include_ticks": True},
                  ).move_to(UP * 0.15 + RIGHT * 0.4)
        xt = VGroup(*[Text(s, font_size=17, color=GREY_B).next_to(
                        ax.c2p(i, 0), DOWN, buff=0.2)
                      for i, s in enumerate(["d=0 · write SQL", "d=1 · result in ctx",
                                             "d=2 · after /60000"])])
        yl = Text("P(correct)", font_size=19, color=GREY_B)
        yl.next_to(ax.c2p(0, 1), UL, buff=0.15)
        self.play(Create(ax), FadeIn(xt), FadeIn(yl), run_time=1.0)

        fail_pts = [(0, 0.08), (1, 0.00), (2, 0.00)]
        ok_pts = [(0, 0.08), (1, 0.14), (2, 1.00)]
        f_line = VMobject(color=C_BAD, stroke_width=5).set_points_as_corners(
            [ax.c2p(x, y) for x, y in fail_pts])
        o_line = VMobject(color=C_OK, stroke_width=5).set_points_as_corners(
            [ax.c2p(x, y) for x, y in ok_pts])
        f_dots = VGroup(*[Dot(ax.c2p(x, y), color=C_BAD, radius=0.06)
                          for x, y in fail_pts])
        o_dots = VGroup(*[Dot(ax.c2p(x, y), color=C_OK, radius=0.06)
                          for x, y in ok_pts])
        lab_f = Text('"…is 393599 ms" → answers seconds', font_size=19, color=C_BAD)
        lab_o = Text('"…divide by 60,000" → calls calculator', font_size=19, color=C_OK)
        lab_o.next_to(ax.c2p(2, 1.0), LEFT, buff=0.35).shift(UP * 0.1)
        lab_f.next_to(ax.c2p(2, 0.0), UP, buff=0.2).shift(LEFT * 1.4)
        self.play(Create(o_line), Create(f_line), FadeIn(f_dots), FadeIn(o_dots),
                  run_time=1.6)
        self.play(FadeIn(lab_f), FadeIn(lab_o), run_time=0.8)
        self.wait(1.2)

        verdict = card(
            ["same SQL, same tool result — the fork is the wording",
             "of the assistant's own turn.  Early text modulates",
             "the next decision's distribution.  (replay K=50)"],
            [C_ACCENT, WHITE, GREY_B], fs=22)
        verdict.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(verdict, shift=UP * 0.2), run_time=0.9)
        self.wait(2.0)


class ShowcaseT10(Scene):
    def construct(self):
        title = Text("Showcase · t10: the decoy locks the outcome",
                     font_size=32, color=C_ACCENT).to_edge(UP, buff=0.5)
        task = Text('"Subscription share of streaming revenue?"   gold: 67.5',
                    font_size=24, color=GREY_A).next_to(title, DOWN, buff=0.25)
        self.play(Write(title), FadeIn(task), run_time=1.2)

        # the search snippet with decoy adjacent to raw numbers
        snip = card(
            ["web_search →",
             "  \"…streaming was 67.3% of total revenue…\"",
             "  \"…subscription streaming: $19.3B of $28.6B…\""],
            [GREY_B, C_BAD, C_OK], fs=21, font=CODE_FONT)
        snip.move_to(UP * 0.7)
        decoy = Text("← decoy", font_size=19, color=C_BAD)
        raw = Text("← = 67.5", font_size=19, color=C_OK)
        decoy.next_to(snip, RIGHT, buff=0.25).align_to(snip[1][1], UP)
        raw.next_to(snip, RIGHT, buff=0.25).align_to(snip[1][2], DOWN)
        self.play(FadeIn(snip), run_time=0.9)
        self.play(FadeIn(decoy), FadeIn(raw), run_time=0.8)
        self.wait(1.0)

        # outcome split, 200 episodes
        bar_bad = Rectangle(width=187 / 200 * 9.0, height=0.62, fill_color=C_BAD,
                            fill_opacity=0.9, stroke_width=0)
        bar_ok = Rectangle(width=max(13 / 200 * 9.0, 0.25), height=0.62,
                           fill_color=C_OK, fill_opacity=0.9, stroke_width=0)
        bars = VGroup(bar_bad, bar_ok).arrange(RIGHT, buff=0.06)
        bars.move_to(DOWN * 1.15)
        l_bad = Text("187/200 copy 67.3 from the snippet", font_size=21, color=C_BAD)
        l_ok = Text("13 compute it", font_size=21, color=C_OK)
        l_bad.next_to(bar_bad, DOWN, buff=0.2).align_to(bar_bad, LEFT)
        l_ok.next_to(bar_ok, UP, buff=0.2).align_to(bar_ok, RIGHT).shift(RIGHT * 1.2)
        self.play(GrowFromEdge(bar_bad, LEFT), GrowFromEdge(bar_ok, LEFT),
                  FadeIn(l_bad), FadeIn(l_ok), run_time=1.3)
        self.wait(1.2)

        verdict = card(
            ["same family as t4's schema trap: an adjacent wrong",
             "candidate in context locks the outcome.",
             "fix: de-confound retrieval, verify citations — not the prompt."],
            [WHITE, C_ACCENT, GREY_B], fs=22)
        verdict.to_edge(DOWN, buff=0.35)
        self.play(FadeOut(l_bad), FadeIn(verdict, shift=UP * 0.2), run_time=0.9)
        self.wait(2.0)
