"""Paper equations 6 and 22--25, using sampled L2 (sum) normalisation.

See references/implementation_notes.md for reference-code discrepancies.
Rates are not rectified: negative activity must be penalised and measured.
"""

import jax.numpy as jnp


def basis(points, frequencies):
    phase = points @ frequencies.T
    pairs = jnp.stack((jnp.cos(phase), jnp.sin(phase)), axis=-1)
    return jnp.concatenate(
        (
            jnp.ones((*phase.shape[:-1], 1), phase.dtype),
            pairs.reshape((*phase.shape[:-1], -1)),
        ),
        axis=-1,
    )


def responses(weights, frequencies, points):
    return basis(points, frequencies) @ weights.T


def squared_distances(x):
    square = jnp.sum(x * x, axis=-1)
    return jnp.maximum(square[:, None] + square[None, :] - 2 * x @ x.T, 0)


def loss_components(weights, frequencies, points, shifts, config):
    # One normalising factor per neuron, taken from the ORIGINAL room.
    # Renormalising each shifted room would make the norm constraint vacuous.
    shifted_points = points[None, :, :] + shifts[:, None, :]
    all_points = jnp.concatenate((points, shifted_points.reshape(-1, 2)))
    raw = responses(weights, frequencies, all_points)
    normaliser = jnp.sqrt(
        jnp.maximum(jnp.sum(raw[: config.samples] ** 2, axis=0), 1e-12)
    )
    rates = raw / normaliser
    base = rates[: config.samples]
    shifted = rates[config.samples :].reshape(
        config.shifts, config.samples, config.neurons
    )
    chi = -jnp.expm1(-squared_distances(points) / (2 * config.spatial_scale**2))
    # Diagonal pairs must contribute exactly zero, including in float32.
    chi = chi.at[jnp.diag_indices(config.samples)].set(0)
    functional = jnp.mean(
        jnp.exp(-squared_distances(base) / (2 * config.sigma**2)) * chi
    )
    positivity = jnp.mean(jnp.maximum(-shifted, 0))
    room_energy = jnp.sum(shifted**2, axis=1)
    norm = jnp.mean((room_energy - 1) ** 2)
    return jnp.stack((functional, positivity, norm))


def objective(weights, frequencies, points, shifts, multipliers, config):
    components = loss_components(weights, frequencies, points, shifts, config)
    total = config.functional_scale * components[0] + jnp.dot(
        multipliers, components[1:]
    )
    return total, components


def translation_matrix(frequencies, displacement):
    """Basis transformation R such that b(x+d) = R b(x), in column notation."""
    angles = frequencies @ displacement
    transform = jnp.eye(2 * len(frequencies) + 1, dtype=frequencies.dtype)
    for i in range(len(frequencies)):
        c, s = jnp.cos(angles[i]), jnp.sin(angles[i])
        block = jnp.array([[c, -s], [s, c]])
        transform = transform.at[2 * i + 1 : 2 * i + 3, 2 * i + 1 : 2 * i + 3].set(
            block
        )
    return transform
