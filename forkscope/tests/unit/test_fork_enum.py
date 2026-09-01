import math

from forkscope.base_path import BasePath
from forkscope.fork_enum import enumerate_branches, observed_positions


def make_base():
    # 3 positions; pos0 greedy A p=0.9 alt B p=0.08; pos1 single; pos2 greedy p=0.5 alt p=0.3 (>=.05)
    return BasePath(
        case_id="c1",
        prompt_ids=[1, 2],
        gen_ids=[10, 20, 30],
        gen_strs=["a", "b", "c"],
        top_logprobs=[
            [(10, math.log(0.9)), (11, math.log(0.08)), (12, math.log(0.01))],
            [(20, math.log(0.999))],
            [(30, math.log(0.5)), (31, math.log(0.3)), (32, math.log(0.02))],
        ],
        finish_reason="stop",
    )


def test_positions_token_mode():
    assert observed_positions(9, 4) == [0, 4, 8]


def test_enumerate_threshold_and_greedy_kept():
    base = make_base()
    br = enumerate_branches(base, [0, 1, 2], p_thresh=0.05)
    by_t = {}
    for b in br:
        by_t.setdefault(b.t, []).append(b)
    # t0: 0.9 and 0.08 kept, 0.01 dropped
    assert sorted(b.tok_id for b in by_t[0]) == [10, 11]
    # t1: only greedy
    assert [b.tok_id for b in by_t[1]] == [20]
    assert by_t[1][0].is_base
    # t2: 0.5, 0.3 kept
    assert sorted(b.tok_id for b in by_t[2]) == [30, 31]


def test_prefix_ids():
    base = make_base()
    br = enumerate_branches(base, [2], p_thresh=0.05)
    alt = [b for b in br if not b.is_base][0]
    assert alt.prefix_ids(base) == [1, 2, 10, 20, 31]
