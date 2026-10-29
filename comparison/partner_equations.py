"""Compare Nathan's pinned numerical core with the portable equation contract.

The inspected partner core is loaded from Git without switching branches,
merging, or copying it into our model. Only its loss/gradient function is called.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

from experiments.equation_check import check_candidate
from src.config import Config

PARTNER_REVISION = "7db3147a434adb20b9b0514e3201bec40700d99b"
PARTNER_SOURCE_SHA256 = (
    "94b14977f143dd65beca6e360ad0a22532ce2ce4ca139604a55800645d0813c1"
)
ROOT = Path(__file__).resolve().parents[1]


def compare(revision, fixture, output, source_path=None):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a full immutable 40-character Git commit hash")
    if revision != PARTNER_REVISION:
        raise ValueError(
            "This audit executes only the inspected partner revision; review new code before updating the pin"
        )
    if output.exists() and any(output.iterdir()):
        raise ValueError("Comparison output must be empty")
    source = source_path.read_bytes() if source_path else subprocess.check_output(
        ["git", "show", f"{revision}:src/actionable/core.py"], cwd=ROOT
    )
    if hashlib.sha256(source).hexdigest() != PARTNER_SOURCE_SHA256:
        raise ValueError("Partner source differs from the inspected content")
    # The default revision was read and inspected before this adapter was written.
    # A different revision must likewise be reviewed before executing it.
    module = types.ModuleType("partner_numerical_core")
    sys.modules[module.__name__] = module
    exec(  # noqa: S102 - exact inspected Git revision AND source digest checked above
        compile(source, f"git:{revision}:src/actionable/core.py", "exec"),
        module.__dict__,
    )
    c = Config(**json.loads((fixture / "config.json").read_text())).validate()
    expected = json.loads((fixture / "expected.json").read_text())
    manifest = json.loads((fixture / "manifest.json").read_text())
    if (
        hashlib.sha256((fixture / "inputs.npz").read_bytes()).hexdigest()
        != manifest["inputs_sha256"]
    ):
        raise ValueError("Fixture input hash does not match manifest")
    with np.load(fixture / "inputs.npz", allow_pickle=False) as z:
        data = {name: z[name].copy() for name in z.files}
    w, f = data["weights"], data["frequencies"]
    p = {"a0": w[:, 0], "B": w[:, 1::2].T, "A": w[:, 2::2].T, "K": f}
    cfg = module.TrainConfig(
        N=c.neurons,
        D=c.frequencies,
        dim=2,
        M=c.samples,
        MS=c.shifts,
        sigma=c.sigma,
        l=c.spatial_scale,
        norm_mode="sum",
        dtype="float64",
    )
    lp, ln = map(float, data["multipliers"])
    native, _ = module.loss_and_grads(p, data["points"], data["shifts"], cfg, lp, ln)
    # Partner code weights functional loss by one; ours uses 25. Dividing the
    # constraint weights by 25 before the call, then multiplying its total and
    # gradients by 25, compares EXACTLY the same objective without editing core.
    parts, gradients = module.loss_and_grads(
        p,
        data["points"],
        data["shifts"],
        cfg,
        lp / c.functional_scale,
        ln / c.functional_scale,
    )
    dw, df = data["weight_direction"], data["frequency_direction"]
    candidate = {
        "functional": parts["Lf"],
        "positivity": parts["Lp"],
        "norm": parts["Lb"],
        "total": c.functional_scale * parts["Lf"] + lp * parts["Lp"] + ln * parts["Lb"],
        "weight_directional_derivative": float(
            c.functional_scale
            * (
                np.sum(gradients["a0"] * dw[:, 0])
                + np.sum(gradients["B"] * dw[:, 1::2].T)
                + np.sum(gradients["A"] * dw[:, 2::2].T)
            )
        ),
        "frequency_directional_derivative": float(
            c.functional_scale * np.sum(gradients["K"] * df)
        ),
    }
    mismatches = check_candidate(expected, candidate)
    result = {
        "partner_revision": revision,
        "partner_source_sha256": hashlib.sha256(source).hexdigest(),
        "fixture_inputs_sha256": manifest["inputs_sha256"],
        "passed_matched_equations": not mismatches,
        "rtol": 2e-5,
        "atol": 1e-7,
        "expected": expected,
        "partner_after_explicit_objective_alignment": candidate,
        "absolute_errors": {k: abs(candidate[k] - expected[k]) for k in expected},
        "mismatches": mismatches,
        "native_total_with_unmodified_multipliers": native["Lf"]
        + lp * native["Lp"]
        + ln * native["Lb"],
        "aligned_total": candidate["total"],
        "interpretation": "Agreement on a small float64 loss/gradient fixture after explicitly aligning functional weighting. It does NOT establish matched training, initialisation, constraints, evaluation metrics, or reproduction success.",
    }
    output.mkdir(parents=True, exist_ok=True)
    for filename, value in (
        ("comparison.json", result),
        ("partner_candidate.json", candidate),
    ):
        (output / filename).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=PARTNER_REVISION)
    parser.add_argument(
        "--fixture", type=Path, default=ROOT / "experiments/fixtures/equation_contract"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Local core.py from the packaged Nathan snapshot; the pinned SHA256 is still enforced")
    args = parser.parse_args()
    result = compare(args.revision, args.fixture, args.output, args.source)
    print(json.dumps(result, indent=2))
    if not result["passed_matched_equations"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
