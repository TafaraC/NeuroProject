"""Export a portable numerical contract for comparing independent implementations.

The expected values use explicit NumPy equations and central finite differences,
not JAX autodiff. Other implementations should consume inputs.npz without
regenerating random numbers, and write the same JSON keys as expected.json.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.config import Config


def numpy_components(weights, frequencies, points, shifts, config):
    def response(x):
        phase = x @ frequencies.T
        features = np.ones((*phase.shape[:-1], 1 + 2 * len(frequencies)))
        features[..., 1::2] = np.cos(phase)
        features[..., 2::2] = np.sin(phase)
        return features @ weights.T

    original = response(points)
    normaliser = np.sqrt(np.maximum(np.sum(original**2, axis=0), 1e-12))
    base = original / normaliser
    shifted = response(points[None, :, :] + shifts[:, None, :]) / normaliser
    spatial_distance = np.sum((points[:, None] - points[None, :]) ** 2, axis=-1)
    neural_distance = np.sum((base[:, None] - base[None, :]) ** 2, axis=-1)
    chi = 1 - np.exp(-spatial_distance / (2 * config.spatial_scale**2))
    return np.array(
        [
            np.mean(np.exp(-neural_distance / (2 * config.sigma**2)) * chi),
            np.maximum(-shifted, 0).mean(),
            ((np.sum(shifted**2, axis=1) - 1) ** 2).mean(),
        ]
    )


def expected_values(data, config):
    w, f, x, shifts = [
        data[key] for key in ("weights", "frequencies", "points", "shifts")
    ]
    multipliers = data["multipliers"]

    def total(weights, frequencies):
        values = numpy_components(weights, frequencies, x, shifts, config)
        return config.functional_scale * values[0] + multipliers @ values[1:]

    parts = numpy_components(w, f, x, shifts, config)
    result = dict(zip(("functional", "positivity", "norm"), map(float, parts)))
    result["total"] = float(total(w, f))
    epsilon = float(data["epsilon"])
    dw, df = data["weight_direction"], data["frequency_direction"]
    result["weight_directional_derivative"] = float(
        (total(w + epsilon * dw, f) - total(w - epsilon * dw, f)) / (2 * epsilon)
    )
    result["frequency_directional_derivative"] = float(
        (total(w, f + epsilon * df) - total(w, f - epsilon * df)) / (2 * epsilon)
    )
    return result


def export_fixture(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Fixture output must be empty")
    c = Config(
        neurons=7, frequencies=3, samples=12, shifts=3, steps=30, dtype="float64"
    )
    rng = np.random.default_rng(20261025)
    data = {
        "weights": rng.normal(size=(7, 7)),
        "frequencies": rng.uniform(0, 2, size=(3, 2)),
        "points": rng.normal(size=(12, 2)),
        "shifts": rng.normal(size=(3, 2)) * 3,
        "multipliers": np.array([0.3, 0.1]),
        "weight_direction": rng.normal(size=(7, 7)),
        "frequency_direction": rng.normal(size=(3, 2)),
        "epsilon": np.array(1e-5),
    }
    for name in ("weight_direction", "frequency_direction"):
        data[name] /= np.linalg.norm(data[name])
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "inputs.npz", **data)
    (output / "config.json").write_text(
        json.dumps(c.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "expected.json").write_text(
        json.dumps(expected_values(data, c), indent=2), encoding="utf-8"
    )
    manifest = {
        "format_version": 1,
        "inputs_sha256": hashlib.sha256(
            (output / "inputs.npz").read_bytes()
        ).hexdigest(),
        "basis_order": "1, cos(k0.x), sin(k0.x), cos(k1.x), sin(k1.x), ...",
        "normalisation": "Per-neuron original-room L2 SUM, reused in every shifted room",
        "sigma": "Standard deviation; variance = sigma squared",
        "reference": "NumPy explicit pairwise differences; central finite-difference derivatives",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return data, c


def check_candidate(expected, candidate, rtol=2e-5, atol=1e-7):
    if set(candidate) != set(expected):
        raise ValueError(
            "Candidate keys must exactly match the six expected measurements"
        )
    errors = {}
    for name, target in expected.items():
        value = candidate[name]
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Non-finite or non-numeric candidate: {name}")
        if not np.isclose(value, target, rtol=rtol, atol=atol):
            errors[name] = {
                "expected": target,
                "actual": value,
                "absolute_error": abs(value - target),
            }
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fixture directory")
    parser.add_argument(
        "--candidate",
        type=Path,
        help="Check this other implementation's JSON against the saved fixture",
    )
    args = parser.parse_args()
    if args.candidate:
        expected = json.loads((args.output / "expected.json").read_text())
        candidate = json.loads(args.candidate.read_text())
        errors = check_candidate(expected, candidate)
        print(json.dumps({"passed": not errors, "mismatches": errors}, indent=2))
        if errors:
            raise SystemExit(1)
    else:
        export_fixture(args.output)
        print(f"Exported numerical contract to {args.output}")


if __name__ == "__main__":
    main()
