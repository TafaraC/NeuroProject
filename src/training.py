"""Compiled Adam/GECO optimisation with complete, non-pickle checkpoints."""

import json
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import Config
from .model import objective


class State(NamedTuple):
    weights: jax.Array
    frequencies: jax.Array
    weight_mean: jax.Array
    weight_variance: jax.Array
    frequency_mean: jax.Array
    frequency_variance: jax.Array
    log_multipliers: jax.Array
    constraint_ema: jax.Array
    key: jax.Array
    points: jax.Array
    shifts: jax.Array
    step: jax.Array


def initialise(config):
    config.validate()
    if config.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = getattr(jnp, config.dtype)
    key, weight_key, frequency_key = jax.random.split(
        jax.random.PRNGKey(config.seed), 3
    )
    weights = jax.random.normal(
        weight_key, (config.neurons, 2 * config.frequencies + 1), dtype
    )
    frequencies = (
        jax.random.uniform(frequency_key, (config.frequencies, 2), dtype)
        * config.frequency_init_scale
    )
    return State(
        weights,
        frequencies,
        jnp.zeros_like(weights),
        jnp.zeros_like(weights),
        jnp.zeros_like(frequencies),
        jnp.zeros_like(frequencies),
        jnp.log(jnp.array([config.positivity_weight, config.norm_weight], dtype)),
        jnp.zeros(2, dtype),
        key,
        jnp.zeros((config.samples, 2), dtype),
        jnp.zeros((config.shifts, 2), dtype),
        jnp.array(0, jnp.int32),
    )


def geco_update(log_multipliers, ema, constraints, config):
    targets = jnp.array(
        [config.positivity_target, config.norm_target], constraints.dtype
    )
    error = jnp.log(jnp.maximum(constraints, 1e-12) / targets)
    ema = config.geco_smoothing * ema + (1 - config.geco_smoothing) * error
    # Floors prevent permanent underflow; hitting a bound is reported in evaluation.
    logs = jnp.clip(log_multipliers + config.geco_rate * ema, -20, 20)
    return logs, ema


def train_step(state, _, config):
    def sample(s):
        key, pk, sk = jax.random.split(s.key, 3)
        points = (
            jax.random.normal(pk, s.points.shape, s.weights.dtype)
            * config.occupancy_scale
        )
        shifts = (
            jax.random.normal(sk, s.shifts.shape, s.weights.dtype) * config.shift_scale
        )
        return s._replace(key=key, points=points, shifts=shifts)

    state = jax.lax.cond(
        state.step % config.resample_every == 0, sample, lambda s: s, state
    )
    (total, parts), (gw, gf) = jax.value_and_grad(objective, (0, 1), has_aux=True)(
        state.weights,
        state.frequencies,
        state.points,
        state.shifts,
        jnp.exp(state.log_multipliers),
        config,
    )
    step = state.step + 1

    def adam(parameter, grad, mean, variance, lr):
        mean = config.beta1 * mean + (1 - config.beta1) * grad
        variance = config.beta2 * variance + (1 - config.beta2) * grad**2
        m = mean / (1 - config.beta1**step)
        v = variance / (1 - config.beta2**step)
        return parameter - lr * m / jnp.sqrt(v + config.adam_eps_root), mean, variance

    w, mw, vw = adam(
        state.weights, gw, state.weight_mean, state.weight_variance, config.weight_lr
    )
    f, mf, vf = adam(
        state.frequencies,
        gf,
        state.frequency_mean,
        state.frequency_variance,
        config.frequency_lr,
    )
    logs, ema = geco_update(
        state.log_multipliers, state.constraint_ema, parts[1:], config
    )
    updated = state._replace(
        weights=w,
        frequencies=f,
        weight_mean=mw,
        weight_variance=vw,
        frequency_mean=mf,
        frequency_variance=vf,
        log_multipliers=logs,
        constraint_ema=ema,
        step=step,
    )
    # These are pre-update losses and multipliers, labelled by the zero-based step.
    metrics = jnp.concatenate(
        (jnp.array([total]), parts, jnp.exp(state.log_multipliers))
    )
    return updated, metrics


@partial(jax.jit, static_argnames=("config", "steps"))
def train_block(state, config, steps):
    return jax.lax.scan(partial(train_step, config=config), state, None, length=steps)


def save_checkpoint(path, state, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{name: np.asarray(value) for name, value in zip(State._fields, state)},
            config=json.dumps(config.to_dict()),
            checkpoint_version=np.array(1),
        )
    temporary.replace(path)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        if int(data["checkpoint_version"]) != 1:
            raise ValueError("Unsupported checkpoint format")
        config = Config(**json.loads(str(data["config"]))).validate()
        template = initialise(config)
        leaves = []
        for name, expected in zip(State._fields, template):
            value = data[name]
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"Checkpoint shape/dtype mismatch for {name}")
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite checkpoint state: {name}")
            leaves.append(jnp.asarray(value))
    state = State(*leaves)
    if not 0 <= int(state.step) <= config.steps:
        raise ValueError("Checkpoint step is outside configured run")
    return state, config
