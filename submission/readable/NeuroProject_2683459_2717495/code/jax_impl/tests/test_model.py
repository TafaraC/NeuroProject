from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.config import Config
from src.model import loss_components, objective, responses, translation_matrix
from src.training import (
    geco_update,
    initialise,
    load_checkpoint,
    save_checkpoint,
    train_block,
)


def small_config():
    return Config(
        neurons=7, frequencies=3, samples=12, shifts=3, steps=30, dtype="float64"
    )


def numpy_losses(w, k, points, shifts, c):
    # Deliberately simple independent expression, without the Gram-matrix shortcut.
    def response(x):
        phase = x @ k.T
        features = np.empty((len(x), 1 + 2 * len(k)))
        features[:, 0] = 1
        features[:, 1::2], features[:, 2::2] = np.cos(phase), np.sin(phase)
        return features @ w.T

    base = response(points)
    norm = np.linalg.norm(base, axis=0)
    base /= norm
    distance = ((base[:, None] - base[None, :]) ** 2).sum(axis=-1)
    input_distance = ((points[:, None] - points[None, :]) ** 2).sum(axis=-1)
    functional = np.mean(
        np.exp(-distance / (2 * c.sigma**2))
        * (1 - np.exp(-input_distance / (2 * c.spatial_scale**2)))
    )
    shifted = np.stack([response(points + shift) / norm for shift in shifts])
    return np.array(
        [
            functional,
            np.maximum(-shifted, 0).mean(),
            ((np.sum(shifted**2, axis=1) - 1) ** 2).mean(),
        ]
    )


def test_loss_matches_independent_numpy():
    c = small_config()
    s = initialise(c)
    rng = np.random.default_rng(42)
    x, shifts = rng.normal(size=(12, 2)), rng.normal(size=(3, 2)) * 3
    actual = loss_components(
        s.weights, s.frequencies, jnp.array(x), jnp.array(shifts), c
    )
    expected = numpy_losses(np.array(s.weights), np.array(s.frequencies), x, shifts, c)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    # Any per-neuron gain cancels in the ORIGINAL-room normalisation.
    gains = jnp.arange(1, 8)[:, None]
    np.testing.assert_allclose(
        loss_components(
            s.weights * gains, s.frequencies, jnp.array(x), jnp.array(shifts), c
        ),
        actual,
        rtol=1e-10,
    )
    assert float(actual[2]) > 1e-3  # shifted norms must not vanish by renormalisation


def test_gradients_match_finite_differences():
    c = small_config()
    s = initialise(c)
    rng = np.random.default_rng(10)
    x, shifts = rng.normal(size=(12, 2)), rng.normal(size=(3, 2))
    multipliers = jnp.array([0.3, 0.1])
    function = lambda w, k: objective(
        w, k, jnp.array(x), jnp.array(shifts), multipliers, c
    )[0]
    gradients = jax.grad(function, (0, 1))(s.weights, s.frequencies)
    for argument, gradient in enumerate(gradients):
        direction = rng.normal(size=gradient.shape)
        direction /= np.linalg.norm(direction)
        args = [np.array(s.weights), np.array(s.frequencies)]
        args[argument] += 1e-5 * direction
        plus = float(function(*args))
        args[argument] -= 2e-5 * direction
        minus = float(function(*args))
        np.testing.assert_allclose(
            (plus - minus) / 2e-5, np.sum(gradient * direction), rtol=2e-5, atol=1e-7
        )


def test_translation_and_composition():
    s = initialise(small_config())
    x = jnp.array([[0.1, -0.2], [0.3, 0.7]])
    a, b = jnp.array([0.2, 0.1]), jnp.array([-0.3, 0.4])
    r = translation_matrix(s.frequencies, a)
    matrix = s.weights @ r @ jnp.linalg.pinv(s.weights)
    np.testing.assert_allclose(
        responses(s.weights, s.frequencies, x) @ matrix.T,
        responses(s.weights, s.frequencies, x + a),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        r @ translation_matrix(s.frequencies, b),
        translation_matrix(s.frequencies, a + b),
        atol=1e-10,
    )


def test_geco_enforces_both_constraints():
    c = small_config()
    target = jnp.array([c.positivity_target, c.norm_target])
    high, _ = geco_update(jnp.zeros(2), jnp.zeros(2), 10 * target, c)
    low, _ = geco_update(jnp.zeros(2), jnp.zeros(2), 0.1 * target, c)
    assert np.all(high > 0) and np.all(low < 0)


def test_resume_preserves_adam_rng_and_partial_batch(tmp_path):
    c = small_config()
    initial = initialise(c)
    uninterrupted, _ = train_block(initial, c, 20)
    partial, _ = train_block(initial, c, 7)  # intentionally not a resampling boundary
    checkpoint = tmp_path / "checkpoint.npz"
    save_checkpoint(checkpoint, partial, c)
    restored, restored_config = load_checkpoint(checkpoint)
    continued, _ = train_block(restored, restored_config, 13)
    for left, right in zip(uninterrupted, continued):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)


def test_invalid_actionability_dimension():
    with pytest.raises(ValueError, match="Actionability"):
        replace(small_config(), frequencies=4).validate()
