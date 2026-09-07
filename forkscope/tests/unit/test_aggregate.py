import numpy as np

from forkscope.aggregate import aggregate
from forkscope.extractor.mcq import MCQExtractor, CATEGORIES


class FixedExtractor(MCQExtractor):
    mapping = {
        "answer is (A)": "A",
        "answer is (B)": "B",
        "junk": "Other",
    }

    def extract(self, text):
        return self.mapping[text]


def test_aggregate_hand_computed():
    records = [
        # t=0: base p=0.75 (A,A,B), alt p=0.25 (B,Other,B)
        {"t": 0, "tok_id": 1, "tok_p": 0.75, "continuations": ["answer is (A)", "answer is (A)", "answer is (B)"]},
        {"t": 0, "tok_id": 2, "tok_p": 0.25, "continuations": ["answer is (B)", "junk", "answer is (B)"]},
        # t=1: single branch p=1.0 all A
        {"t": 1, "tok_id": 3, "tok_p": 1.0, "continuations": ["answer is (A)", "answer is (A)"]},
    ]
    o_t, per_draw = aggregate(records, CATEGORIES, FixedExtractor())
    assert o_t.shape == (2, 5)
    # t0: base hist = (2/3, 1/3, 0, 0, 0); alt hist = (0, 2/3, 0, 0, 1/3)
    # weights normalized: 0.75/1.0, 0.25/1.0
    exp0 = 0.75 * np.array([2 / 3, 1 / 3, 0, 0, 0]) + 0.25 * np.array([0, 2 / 3, 0, 0, 1 / 3])
    np.testing.assert_allclose(o_t[0], exp0, atol=1e-9)
    np.testing.assert_allclose(o_t[1], [1, 0, 0, 0, 0], atol=1e-9)
    assert per_draw[(0, 1)] == ["A", "A", "B"]
