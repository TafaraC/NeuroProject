"""Held-out constraint checks and descriptive spatial autocorrelation scores."""

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import rotate
from scipy.signal import correlate

from .model import loss_components, responses, translation_matrix


def autocorrelation(rate_map):
    """Overlap-normalised Pearson SAC, excluding overlaps below 20% of the map."""
    a = np.asarray(rate_map, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Expected a finite 2D rate map")
    ones = np.ones_like(a)
    corr = lambda x, y: correlate(x, y, mode="full", method="fft")
    count = np.maximum(corr(ones, ones), 1)
    sx, sy = corr(a, ones), corr(ones, a)
    numerator = corr(a, a) - sx * sy / count
    vx = np.maximum(corr(a * a, ones) - sx * sx / count, 0)
    vy = np.maximum(corr(ones, a * a) - sy * sy / count, 0)
    denominator = np.sqrt(vx * vy)
    valid = (count >= 0.2 * a.size) & (denominator > 1e-10)
    result = np.full(count.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=valid)
    return np.clip(result, -1, 1)


def grid_score(sac, map_size):
    """max[min(r60,r120)-max(r30,r90,r150)] over predeclared annuli.

    This descriptive score is NOT a shuffled significance test or module detector.
    The same annuli/interpolation are used for every cell and condition.
    """
    angles = (30, 60, 90, 120, 150)
    rotations = [
        rotate(
            sac,
            angle,
            reshape=False,
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        for angle in angles
    ]
    y, x = np.indices(sac.shape)
    radius = np.hypot(y - (sac.shape[0] - 1) / 2, x - (sac.shape[1] - 1) / 2)
    scores, correlations = [], []
    outer_radii = np.linspace(0.15, 0.7, 12) * map_size
    for outer in outer_radii:
        ring = (radius >= 0.5 * outer) & (radius <= outer) & np.isfinite(sac)
        # Use the SAME pixels for all angles within an annulus.
        for turned in rotations:
            ring &= np.isfinite(turned)
        values = []
        for turned in rotations:
            if (
                ring.sum() < 30
                or np.std(sac[ring]) < 1e-8
                or np.std(turned[ring]) < 1e-8
            ):
                values.append(np.nan)
            else:
                values.append(float(np.corrcoef(sac[ring], turned[ring])[0, 1]))
        correlations.append(values)
        scores.append(min(values[1], values[3]) - max(values[0], values[2], values[4]))
    valid = np.isfinite(scores)
    if not valid.any():
        return {"score": None, "outer_radius_pixels": None, "correlations": None}
    index = int(np.nanargmax(scores))
    return {
        "score": scores[index],
        "outer_radius_pixels": float(outer_radii[index]),
        "correlations": correlations[index],
    }


def minimum_map_resolution(frequencies, extent):
    """At least four samples per shortest axis wavelength (twice Nyquist)."""
    maximum = float(np.max(np.abs(frequencies)))
    return max(32, int(np.ceil(4 * extent * maximum / np.pi)) + 1)


def evaluate(state, config, seed=12345, resolution=320, extent=1.5, batches=8):
    rng = np.random.default_rng(seed)
    w, frequencies = np.asarray(state.weights), np.asarray(state.frequencies)
    if not np.isfinite(extent) or extent <= 0 or batches < 1:
        raise ValueError("extent and batches must be positive")
    required = minimum_map_resolution(frequencies, extent)
    if resolution < required:
        raise ValueError(
            f"Map grid undersamples learned frequencies: use resolution >= {required} "
            f"at extent={extent}, or a smaller shared extent for BOTH conditions"
        )
    # A fixed calibration set defines one static neural code for maps/actionability.
    calibration = rng.normal(0, config.occupancy_scale, (config.samples, 2))
    calibration_rates = np.asarray(responses(w, frequencies, jnp.asarray(calibration)))
    norm = np.sqrt(np.maximum((calibration_rates**2).sum(axis=0), 1e-12))
    axis = np.linspace(-extent, extent, resolution)
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel()))
    rates = np.asarray(responses(w, frequencies, jnp.asarray(points))) / norm
    maps = rates.T.reshape(config.neurons, resolution, resolution)
    sacs = np.stack([autocorrelation(m) for m in maps])
    scores = [grid_score(sac, resolution) for sac in sacs]

    losses, negative_fractions, energy_errors, minima = [], [], [], []
    for _ in range(batches):
        base = rng.normal(0, config.occupancy_scale, (config.samples, 2))
        shifts = rng.normal(0, config.shift_scale, (config.shifts, 2))
        losses.append(
            np.asarray(
                loss_components(
                    w, frequencies, jnp.asarray(base), jnp.asarray(shifts), config
                )
            )
        )
        base_rates = np.asarray(responses(w, frequencies, jnp.asarray(base)))
        base_norm = np.sqrt(np.maximum((base_rates**2).sum(axis=0), 1e-12))
        shifted = base[None] + shifts[:, None]
        values = np.asarray(responses(w, frequencies, jnp.asarray(shifted))) / base_norm
        negative_fractions.append(float(np.mean(values < -1e-6)))
        energy_errors.extend(np.abs((values**2).sum(axis=1) - 1).ravel())
        minima.append(float(values.min()))

    # Analytic translation in the Fourier basis, lifted into neural space.
    scaled_w = w.astype(float) / norm[:, None]
    singular_values = np.linalg.svd(scaled_w, compute_uv=False)
    displacement = np.array([0.37, -0.29])
    transform = (
        scaled_w
        @ np.asarray(
            translation_matrix(jnp.asarray(frequencies), jnp.asarray(displacement))
        )
        @ np.linalg.pinv(scaled_w)
    )
    start = np.asarray(responses(w, frequencies, jnp.asarray(calibration))) / norm
    end = (
        np.asarray(responses(w, frequencies, jnp.asarray(calibration + displacement)))
        / norm
    )
    actionability_error = np.linalg.norm(start @ transform.T - end) / max(
        np.linalg.norm(end), 1e-12
    )
    averages = np.mean(losses, axis=0)
    finite_scores = np.array([s["score"] for s in scores if s["score"] is not None])
    summary = {
        "step": int(state.step),
        "config": config.to_dict(),
        "evaluation_seed": seed,
        "evaluation_batches": batches,
        "map_resolution": resolution,
        "map_extent": [-extent, extent],
        "minimum_map_resolution": required,
        "maximum_axis_frequency": float(np.max(np.abs(frequencies))),
        "evaluation_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "functional_loss": float(averages[0]),
        "positivity_loss": float(averages[1]),
        "norm_loss": float(averages[2]),
        "positivity_target": config.positivity_target,
        "norm_target": config.norm_target,
        "positivity_target_met": bool(averages[1] <= config.positivity_target),
        "norm_target_met": bool(averages[2] <= config.norm_target),
        "negative_fraction_below_minus_1e_minus6": float(np.mean(negative_fractions)),
        "minimum_shifted_rate": min(minima),
        "room_energy_absolute_error_p95": float(np.quantile(energy_errors, 0.95)),
        "grid_score_median": float(np.median(finite_scores))
        if len(finite_scores)
        else None,
        "grid_score_q25": float(np.quantile(finite_scores, 0.25))
        if len(finite_scores)
        else None,
        "grid_score_q75": float(np.quantile(finite_scores, 0.75))
        if len(finite_scores)
        else None,
        "grid_score_positive_count": int(np.sum(finite_scores > 0)),
        "grid_score_valid_count": len(finite_scores),
        "grid_scores": scores,
        "actionability_relative_error": float(actionability_error),
        "weight_rank": int(np.linalg.matrix_rank(scaled_w)),
        "weight_condition_number": float(
            singular_values[0] / max(singular_values[-1], 1e-30)
        ),
        "geco_multipliers": np.exp(np.asarray(state.log_multipliers)).tolist(),
        "geco_clipping_active": bool(
            np.any(np.abs(np.asarray(state.log_multipliers)) >= 19.99)
        ),
        "interpretation": "Descriptive cell scores; no shuffle significance, independence of cells, or module count inferred.",
    }
    return summary, maps, sacs, axis


def save_evaluation(state, config, folder, **kwargs):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    summary, maps, sacs, axis = evaluate(state, config, **kwargs)
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    np.savez_compressed(
        folder / "maps.npz",
        rates=maps,
        autocorrelations=sacs,
        axis=axis,
        frequencies=np.asarray(state.frequencies),
        weights=np.asarray(state.weights),
    )
    plot_evaluation(
        summary,
        maps,
        sacs,
        np.asarray(state.weights),
        np.asarray(state.frequencies),
        folder,
    )
    return summary


def plot_evaluation(summary, maps, sacs, weights, frequencies, folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(maps)
    cols = 8
    rows = int(np.ceil(n / cols))
    for label, panels in (("rate_maps", maps), ("autocorrelations", sacs)):
        fig, axes = plt.subplots(rows, cols, figsize=(12, 1.65 * rows), squeeze=False)
        for i, ax in enumerate(axes.ravel()):
            ax.set_axis_off()
            if i < n:
                options = (
                    {"vmin": -1, "vmax": 1, "cmap": "coolwarm"}
                    if label == "autocorrelations"
                    else {"cmap": "viridis"}
                )
                ax.imshow(panels[i], origin="lower", **options)
                score = summary["grid_scores"][i]["score"]
                title = f"{i}: {score:.2f}" if score is not None else f"{i}: undefined"
                ax.set_title(title, fontsize=8)
        fig.suptitle(
            f"{label.replace('_', ' ').title()} | l={summary['config']['spatial_scale']} | step {summary['step']}\n"
            "All neurons, index order; titles show descriptive grid scores",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(folder / f"{label}.png", dpi=140)
        plt.close(fig)
    powers = np.sum(weights[:, 1::2] ** 2 + weights[:, 2::2] ** 2, axis=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    for sign in (1, -1):
        ax.scatter(
            sign * frequencies[:, 0],
            sign * frequencies[:, 1],
            s=10 + 180 * powers / max(powers.max(), 1e-12),
            alpha=0.65,
        )
    ax.set(
        xlabel="kx (radians / spatial unit)",
        ylabel="ky (radians / spatial unit)",
        title="Learned frequencies; area indicates coefficient power",
        aspect="equal",
    )
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(folder / "frequencies.png", dpi=140)
    plt.close(fig)
