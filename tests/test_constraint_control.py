import numpy as np

from comparison.constraint_control import gaussian_energy, rates, reference_code


def test_constructed_code_matches_nonnegative_identity():
    w, k = reference_code(neurons=8)
    x = np.random.default_rng(0).normal(size=(100, 2)) * 5
    phase = np.arctan2(w[:, 2::2], w[:, 1::2])
    u = x @ k[0] - phase[:, 0, None]
    v = x @ k[1] - phase[:, 1, None]
    expected = 0.5 * np.abs(1 + np.exp(1j * u) + np.exp(-1j * v)) ** 2
    np.testing.assert_allclose(rates(w, k, x).T, expected, atol=1e-12)
    assert rates(w, k, x).min() >= -1e-12


def test_gaussian_energy_constant_and_translation_limits():
    w, k = reference_code(neurons=8)
    # Constant code: expectation is exactly the squared constant in every room.
    constant = w.copy()
    constant[:, 1:] = 0
    shifts = np.array([[0.0, 0.0], [0.7, -2.0]])
    np.testing.assert_allclose(gaussian_energy(constant, k, shifts), 1.5**2)
    # At high frequency, oscillating Gaussian moments vanish exponentially.
    np.testing.assert_allclose(gaussian_energy(w, k * 3, shifts), 3.75, atol=1e-12)
    # Independent Gaussian quadrature checks phase, pair sums and shifts.
    nodes, weights = np.polynomial.hermite.hermgauss(48)
    x, y = np.meshgrid(nodes * np.sqrt(2), nodes * np.sqrt(2))
    points = np.column_stack((x.ravel(), y.ravel()))
    probabilities = np.outer(weights, weights).ravel() / np.pi
    slower = k / 6
    numerical = np.stack(
        [probabilities @ rates(w, slower, points + s) ** 2 for s in shifts]
    )
    np.testing.assert_allclose(
        gaussian_energy(w, slower, shifts), numerical, atol=1e-11
    )
