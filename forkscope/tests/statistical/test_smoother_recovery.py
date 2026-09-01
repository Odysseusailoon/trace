"""V3: smoother recovers reference from nested low-S subsample (synthetic)."""
import numpy as np

from forkscope.metrics import efficiency_multiplier, fork_points, pooled_tvd, tvd
from forkscope.smoothing.model import FullModel


def structured_draws(T=60, S_full=200, K=4, seed=3, forks=(20, 40)):
    rng = np.random.default_rng(seed)
    ps = [
        np.array([0.7, 0.1, 0.1, 0.1]),
        np.array([0.1, 0.7, 0.1, 0.1]),
        np.array([0.1, 0.1, 0.7, 0.1]),
    ]
    rows = []
    for t in range(T):
        p = ps[0] if t < forks[0] else (ps[1] if t < forks[1] else ps[2])
        rows.append(rng.multinomial(1, p, size=S_full).argmax(axis=1))
    return np.stack(rows), np.stack(
        [ps[0] if t < forks[0] else (ps[1] if t < forks[1] else ps[2]) for t in range(T)]
    )


def test_v3_smoother_recovers():
    draws, truth = structured_draws()
    T, S_full = draws.shape
    K = draws.max() + 1
    obs = np.arange(0, T, 4)          # N=4 spacing
    S_low = 15
    sub = draws[:, :S_low]

    model = FullModel([16, 64, 256, 1024], [2, 4, 8, 16], cv_folds=5)
    model.fit(obs, sub[obs])
    smoothed = model.predict(np.arange(T))

    # criterion 1: smoothed closer to the KNOWN ground truth than raw
    raw_at_obs = np.stack([np.bincount(r, minlength=K) for r in sub[obs]]) / S_low
    t_raw = pooled_tvd(raw_at_obs, truth[obs])
    t_smooth = pooled_tvd(smoothed[obs], truth[obs])
    assert t_smooth < t_raw

    # criterion 2: efficiency multiplier >= 2x vs raw TV-vs-S curve
    from forkscope.stats import measured_tv_vs_S
    s_arr = [10, 20, 50, 100]
    m0 = measured_tv_vs_S(draws, s_arr)
    mult = efficiency_multiplier(s_arr, m0, tv_model=t_smooth, s_model=S_low)
    assert mult >= 2.0

    # criterion 3: structural forks (ground truth flips at 20 and 40) recovered
    sm_forks = fork_points(smoothed, np.arange(T), threshold=0.10)
    for f in (20, 40):
        assert any(abs(sf - f) <= 10 for sf in sm_forks), f"fork at {f} missed"
