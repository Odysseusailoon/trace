"""forkscope mechanics video — how token-level resampling works inside the model.

Schematic probabilities (illustrative, marked as such); pipeline structure and
thresholds (p>=0.05, S continuations, prefix sharing) match RFC F1-F3.
Render: ~/.venvs/manim/bin/manim -qh token_sample_anim.py TokenSampling
"""
import random
from manim import *

C_OK = "#2fbf71"; C_BAD = "#e4572e"; C_OTH = "#8a8f98"
C_ACCENT = "#ffd166"; C_BLUE = "#4ea8de"
CODE_FONT = "Menlo"


def transformer_stack(n=4, w=2.6, h=0.5):
    blocks = VGroup(*[
        RoundedRectangle(corner_radius=0.08, width=w, height=h,
                         stroke_color=C_BLUE, stroke_width=2,
                         fill_color=C_BLUE, fill_opacity=0.12)
        for _ in range(n)]).arrange(UP, buff=0.14)
    lab = Text("transformer ×N", font_size=18, color=C_BLUE).next_to(blocks, UP, buff=0.15)
    return VGroup(blocks, lab)


def prob_bars(items, bar_h=0.3, fs=19):
    """items: [(token, p, color)] -> aligned rows: padded mono label | bar | p."""
    pad = max(len(t) for t, _, _ in items)
    rows = VGroup()
    for tok, p, col in items:
        t = Text(tok.ljust(pad), font_size=fs, font=CODE_FONT, color=WHITE)
        bar = Rectangle(width=max(3.4 * p, 0.02), height=bar_h,
                        fill_color=col, fill_opacity=0.85, stroke_width=0)
        pt = Text(f"{p:.2f}", font_size=fs - 2, color=GREY_B)
        rows.add(VGroup(t, bar, pt).arrange(RIGHT, buff=0.25))
    rows.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
    return rows


def token_chip(s, color=WHITE, fs=21):
    txt = Text(s, font_size=fs, font=CODE_FONT, color=color)
    box = SurroundingRectangle(txt, corner_radius=0.08, buff=0.12,
                               stroke_color=color, stroke_width=1.5,
                               fill_color=color, fill_opacity=0.07)
    return VGroup(box, txt)


class TokenSampling(Scene):
    def construct(self):
        random.seed(7)

        # ---------- Act 1: one decoding step ----------
        title = Text("Every token is a dice roll.", font_size=34).to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1.0)

        prefix = VGroup(*[token_chip(s, fs=20) for s in
                          ["SELECT", " AVG", "(", "…", " FROM"]]).arrange(RIGHT, buff=0.12)
        prefix.to_edge(LEFT, buff=0.6).shift(DOWN * 0.2)
        stack = transformer_stack().next_to(prefix, RIGHT, buff=0.9)
        a_in = Arrow(prefix.get_right(), stack.get_left(), buff=0.15,
                     stroke_width=3, color=GREY_B)

        cands = [(" Album", 0.46, C_BAD), (" Track", 0.31, C_OK),
                 (" Invoice", 0.12, C_OTH), (" (", 0.05, C_OTH)]
        bars = prob_bars(cands).next_to(stack, RIGHT, buff=0.9)
        a_out = Arrow(stack.get_right(), bars.get_left(), buff=0.15,
                      stroke_width=3, color=GREY_B)
        soft = Text("logits → softmax", font_size=18, color=GREY_B)
        soft.next_to(bars, DOWN, buff=0.3)
        schem = Text("(probabilities illustrative)", font_size=15, color=GREY_D)
        schem.to_corner(DR, buff=0.4)

        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.2) for c in prefix],
                              lag_ratio=0.1), run_time=1.0)
        self.play(Create(a_in), FadeIn(stack), run_time=0.9)
        self.play(Create(a_out), FadeIn(soft), FadeIn(bars), FadeIn(schem), run_time=1.1)
        self.wait(0.8)

        # temperature roll: highlight sampled token
        pick = SurroundingRectangle(bars[0], color=C_ACCENT, buff=0.08, stroke_width=3)
        roll = Text("τ = 1.0 sample", font_size=20, color=C_ACCENT)
        roll.next_to(pick, UP, buff=0.2).align_to(pick, LEFT)
        self.play(Create(pick), FadeIn(roll), run_time=0.8)
        picked = token_chip(" Album", color=C_BAD, fs=20)
        picked.next_to(prefix, RIGHT, buff=0.12)
        self.play(FadeOut(a_in), TransformFromCopy(bars[0][0], picked), run_time=0.8)
        self.wait(0.8)

        # ---------- Act 2: the fork move ----------
        head2 = Text("The fork move: same prefix, force the other token.",
                     font_size=30, color=C_ACCENT).to_edge(UP, buff=0.5)
        self.play(ReplacementTransform(title, head2),
                  FadeOut(pick), FadeOut(roll), run_time=0.9)

        alt = token_chip(" Track", color=C_OK, fs=20)
        alt.next_to(prefix, RIGHT, buff=0.12).shift(DOWN * 0.9)
        e1 = Line(prefix.get_right(), picked.get_left(), color=C_BAD,
                  stroke_width=2.5, buff=0.05)
        e2 = Line(prefix.get_right(), alt.get_left(), color=C_OK,
                  stroke_width=2.5, buff=0.05)
        thresh = Text("keep every candidate with p ≥ 0.05", font_size=20, color=GREY_B)
        thresh.to_edge(DOWN, buff=0.6)
        self.play(TransformFromCopy(bars[1][0], alt), Create(e1), Create(e2),
                  FadeIn(thresh), run_time=1.0)

        # S continuations from each branch
        fans = VGroup()
        for src, col in [(picked, C_BAD), (alt, C_OK)]:
            start = src.get_right() + RIGHT * 0.05
            for i in range(12):
                y = start[1] + random.uniform(-0.45, 0.45)
                end = np.array([start[0] + 2.0, y, 0])
                p = CubicBezier(start, start + RIGHT * 0.7,
                                end + LEFT * 0.7, end)
                p.set_stroke(col, width=1.2, opacity=0.4)
                fans.add(p)
        s_lab = Text("×S continuations each, τ = 1.0", font_size=20, color=GREY_B)
        s_lab.next_to(thresh, UP, buff=0.25)
        self.play(FadeOut(stack), FadeOut(bars), FadeOut(a_out),
                  FadeOut(soft),
                  LaggedStart(*[Create(p) for p in fans], lag_ratio=0.02),
                  FadeIn(s_lab), run_time=1.8)
        self.wait(1.0)

        # ---------- Act 3: every position, one shared trunk ----------
        head3 = Text("Do it at every position — one shared trunk.",
                     font_size=30, color=C_ACCENT).to_edge(UP, buff=0.5)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not head2], 
                  ReplacementTransform(head2, head3), run_time=0.9)

        trunk_y = 0.4
        trunk = VGroup(*[token_chip(s, fs=17) for s in
                        ["SELECT", " AVG", "(", "n", ")", " FROM", " t", " GROUP"]])
        trunk.arrange(RIGHT, buff=0.1).move_to(UP * trunk_y).to_edge(LEFT, buff=0.7)
        self.play(LaggedStart(*[FadeIn(c) for c in trunk], lag_ratio=0.06), run_time=1.0)

        branches = VGroup()
        for i, chip in enumerate(trunk[1:7]):
            start = chip.get_bottom() + DOWN * 0.05
            col = [C_OK, C_BAD, C_OTH][i % 3]
            for k in range(5):
                end = start + DOWN * random.uniform(1.0, 2.2) + \
                      RIGHT * random.uniform(0.1, 0.6)
                b = CubicBezier(start, start + DOWN * 0.5,
                                end + UP * 0.5, end)
                b.set_stroke(col, width=1.1, opacity=0.38)
                branches.add(b)
        self.play(LaggedStart(*[Create(b) for b in branches], lag_ratio=0.01),
                  run_time=2.0)

        radix = VGroup(
            Text("prefixes are nested → RadixAttention computes the trunk once",
                 font_size=22, color=C_BLUE),
            Text("measured: 98.5% prefix cache hit", font_size=22, color=C_ACCENT),
        ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.55)
        hl = trunk.copy().set_stroke(C_BLUE, width=3)
        self.play(FadeIn(radix[0]), ShowPassingFlash(hl, time_width=0.6),
                  run_time=1.4)
        self.play(FadeIn(radix[1]), run_time=0.7)
        self.wait(1.2)

        # ---------- Act 4: outcome ----------
        out = Text("count where each branch lands  →  o_t, the fork detector",
                   font_size=26, color=WHITE).move_to(DOWN * 0.4)
        self.play(*[FadeOut(m) for m in [branches, radix]],
                  trunk.animate.shift(UP * 0.6), FadeIn(out), run_time=1.0)
        self.wait(1.5)
