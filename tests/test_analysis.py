import numpy as np

from src.analysis import autocorrelation, grid_score


def test_grid_score_distinguishes_synthetic_patterns():
    axis = np.linspace(-5, 5, 96)
    x, y = np.meshgrid(axis, axis)
    hexagonal = (
        3
        + np.cos(4 * x)
        + np.cos(2 * x + 2 * np.sqrt(3) * y)
        + np.cos(-2 * x + 2 * np.sqrt(3) * y)
    )
    square = 2 + np.cos(4 * x) + np.cos(4 * y)
    stripe = 1 + np.cos(4 * x)
    rng = np.random.default_rng(1)
    noise = rng.normal(size=x.shape)
    scores = [
        grid_score(autocorrelation(m), len(axis))["score"]
        for m in (hexagonal, square, stripe, noise)
    ]
    assert scores[0] > 0.8, scores
    assert scores[0] > max(scores[1:]) + 0.5, scores
    sac = autocorrelation(hexagonal)
    np.testing.assert_allclose(sac[95, 95], 1, atol=1e-10)
    np.testing.assert_allclose(sac, sac[::-1, ::-1], atol=1e-10)


def test_constant_map_is_undefined_not_a_grid():
    score = grid_score(autocorrelation(np.ones((32, 32))), 32)
    assert score["score"] is None
