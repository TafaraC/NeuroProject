"""Compare the NumPy implementation's numerical core (../numpy_impl) with the portable equation contract.

In the merged submission the NumPy implementation lives in the sibling folder ``numpy_impl``. Its loss/gradient
module is loaded from there, after its SHA-256 digest is checked against the digest of the file that was inspected
before this adapter was written (Git commit 7db3147a..., recorded below). That file is unchanged in the final
NumPy code, so the audit applies to the submitted implementation. Only its loss/gradient function is called.
"""

import argparse
import hashlib
import json
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
DEFAULT_CORE = ROOT.parent / "numpy_impl" / "src" / "actionable" / "core.py"


def compare(core_path, fixture, output):
    core_path = Path(core_path)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Comparison output must be empty")
    # Hash the LF-normalised bytes so a Windows checkout (CRLF) is accepted as the same content.
    source = core_path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() != PARTNER_SOURCE_SHA256:
        raise ValueError(
            "NumPy core differs from the inspected content; review the new code before updating the pinned digest"
        )
    # The pinned content was read and inspected before this adapter was written.
    module = types.ModuleType("partner_numerical_core")
    sys.modules[module.__name__] = module
    exec(  # noqa: S102 - exact inspected source; digest checked above
        compile(source, str(core_path), "exec"),
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
        "partner_revision": PARTNER_REVISION,
        "partner_core_path": "numpy_impl/src/actionable/core.py",
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
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE, help="path to numpy_impl/src/actionable/core.py")
    parser.add_argument(
        "--fixture", type=Path, default=ROOT / "experiments/fixtures/equation_contract"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.core, args.fixture, args.output)
    print(json.dumps(result, indent=2))
    if not result["passed_matched_equations"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
