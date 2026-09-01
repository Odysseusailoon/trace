import numpy as np

from forkscope.metrics import (
    efficiency_multiplier,
    fork_points,
    fork_region_tvd,
    loglog_slope,
    replicate_tvd,
    tvd,
)


def test_tvd_known_values():
    p = np.array([[0.5, 0.5], [1.0, 0.0]])
    q = np.array([[0.25, 0.75], [0.0, 1.0]])
    np.testing.assert_allclose(tvd(p, q), [0.25, 1.0], atol=1e-9)


def test_fork_points():
    curve = np.array([[0.9, 0.1], [0.88, 0.12], [0.2, 0.8], [0.19, 0.81]])
    pos = np.array([0, 4, 8, 12])
    assert fork_points(curve, pos, threshold=0.10) == [8]


def test_fork_region_tvd():
    p = np.tile([0.5, 0.5], (10, 1))
    q = np.tile([0.6, 0.4], (10, 1))
    pos = np.arange(10) * 4
    v = fork_region_tvd(p, q, pos, forks=[16], radius=10)
    assert abs(v - 0.1) < 1e-9


def test_loglog_slope():
    s = np.array([20, 50, 100, 200], float)
    tv = s ** -0.5
    assert abs(loglog_slope(s, tv) - (-0.5)) < 0.02


def test_replicate_tvd_iid_shrinks_with_S():
    rng = np.random.default_rng(3)
    draws = rng.integers(0, 3, size=(30, 200))
    t20 = replicate_tvd(draws, 20)
    t100 = replicate_tvd(draws, 100)
    assert t100 < t20


def test_efficiency_multiplier():
    s = np.array([10, 20, 40, 80], float)
    m0 = s ** -0.5 * 0.5  # raw TV curve
    # model achieves same TV as raw at S=80 while using S=20
    mult = efficiency_multiplier(s, m0, tv_model=m0[-1], s_model=20)
    assert abs(mult - 4.0) < 0.2
