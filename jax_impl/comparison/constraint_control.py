"""Calibrate the sampled norm penalty using a known non-negative hexagonal code.

This is a constructed diagnostic control, never a trained reproduction result.
g(x) = 1.5 + cos(u) + cos(v) + cos(u+v)
     = 0.5 * |1 + exp(i*u) + exp(-i*v)|**2.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def reference_code(neurons=64, frequency=6.0, seed=991):
    rng = np.random.default_rng(seed)
    first = rng.uniform(0, 2 * np.pi, (neurons, 2))
    phase = np.column_stack((first, first.sum(axis=1)))
    k = frequency * np.array([[1, 0], [-0.5, np.sqrt(3) / 2], [0.5, np.sqrt(3) / 2]])
    w = np.empty((neurons, 7))
    w[:, 0] = 1.5
    w[:, 1::2] = np.cos(phase)
    w[:, 2::2] = np.sin(phase)
    return w, k


def rates(w, k, x):
    phase = x @ k.T
    return w[:, 0] + np.cos(phase) @ w[:, 1::2].T + np.sin(phase) @ w[:, 2::2].T


def gaussian_energy(w, k, shifts):
    """Exact E[g(X+s)^2] for X~N(0,I), using its characteristic function."""
    wavevectors = np.vstack((np.zeros((1, 2)), k, -k))
    positive = (w[:, 1::2] - 1j * w[:, 2::2]) / 2
    coefficients = np.column_stack((w[:, 0], positive, positive.conj()))
    sums = wavevectors[:, None, :] + wavevectors[None, :, :]
    kernel = np.exp(
        1j * np.einsum("sd,ijd->sij", shifts, sums) - 0.5 * (sums**2).sum(-1)
    )
    return np.einsum("ni,nj,sij->sn", coefficients, coefficients, kernel).real


def measure(batches=128, sample_counts=(50, 150, 600), seed=24680):
    if (
        batches < 2
        or any(m < 1 for m in sample_counts)
        or len(set(sample_counts)) != len(sample_counts)
    ):
        raise ValueError("Use at least two batches and distinct positive sample counts")
    w, k = reference_code()
    rng = np.random.default_rng(seed)
    exact_base = gaussian_energy(w, k, np.zeros((1, 2)))
    rows = []
    exact_error = 0.0
    minimum_rate = float("inf")
    for batch in range(batches):
        # Nested samples and shared shifts pair the different M conditions.
        points = rng.normal(size=(max(sample_counts), 2))
        shifts = rng.normal(size=(15, 2)) * 3
        original = rates(w, k, points)
        shifted = rates(w, k, points[None, :, :] + shifts[:, None, :])
        minimum_rate = min(minimum_rate, float(original.min()), float(shifted.min()))
        exact_q = gaussian_energy(w, k, shifts) / exact_base
        exact_error = max(exact_error, float(np.max(np.abs(exact_q - 1))))
        for m in sample_counts:
            q = np.sum(shifted[:, :m] ** 2, axis=1) / np.sum(original[:m] ** 2, axis=0)
            rows.append(
                {
                    "batch": batch,
                    "samples": m,
                    "norm_loss": float(np.mean((q - 1) ** 2)),
                }
            )
    summary = {
        "constructed_control_not_training": True,
        "formula": "1.5+cos(u)+cos(v)+cos(u+v) = |1+exp(iu)+exp(-iv)|^2/2",
        "neurons": 64,
        "frequency_pairs": 3,
        "frequency_magnitude": 6.0,
        "phase_seed": 991,
        "sampling_seed": seed,
        "batches": batches,
        "shifts": 15,
        "shift_std": 3.0,
        "occupancy_std": 1.0,
        "target_under_test": 0.02,
        "maximum_analytic_room_energy_relative_error": exact_error,
        "minimum_sampled_raw_rate": minimum_rate,
        "numpy_version": np.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurements": [],
    }
    for m in sample_counts:
        values = np.array([r["norm_loss"] for r in rows if r["samples"] == m])
        summary["measurements"].append(
            {
                "samples": m,
                "mean_norm_loss": float(values.mean()),
                "standard_error_over_batches": float(
                    values.std(ddof=1) / np.sqrt(batches)
                ),
                "fraction_of_batches_at_or_below_target": float(
                    np.mean(values <= 0.02)
                ),
            }
        )
    return summary, rows, w, k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Control output must be empty")
    summary, rows, w, k = measure(args.batches)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (args.output / "batches.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output / "constructed_code.npz", weights=w, frequencies=k)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axis = np.linspace(-1.5, 1.5, 192)
    x, y = np.meshgrid(axis, axis)
    r = rates(w[:1], k, np.stack((x, y), axis=-1))[..., 0]
    axes[0].imshow(r, origin="lower", extent=(-1.5, 1.5, -1.5, 1.5))
    axes[0].set_title("Constructed non-negative hexagonal code")
    ms = summary["measurements"]
    axes[1].errorbar(
        [m["samples"] for m in ms],
        [m["mean_norm_loss"] for m in ms],
        yerr=[m["standard_error_over_batches"] for m in ms],
        fmt="o-",
        capsize=4,
    )
    axes[1].axhline(0.02, color="red", linestyle="--", label="Our chosen target")
    axes[1].set(
        xlabel="Original-room sample count M",
        ylabel="Mean sampled norm penalty",
        title="Error bars: one Monte Carlo SE",
    )
    axes[1].legend(fontsize=8)
    values = [r["norm_loss"] for r in rows if r["samples"] == 150]
    axes[2].hist(values, bins=16, edgecolor="white")
    axes[2].axvline(0.02, color="red", linestyle="--")
    axes[2].set(
        xlabel="Sampled norm penalty at M=150",
        ylabel="Independent batches",
        title="Sampling variability, fixed code",
    )
    fig.suptitle("Constraint calibration control | no training or reproduction claim")
    fig.tight_layout()
    fig.savefig(args.output / "constraint_control.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
