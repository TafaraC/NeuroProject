"""Read Nathan's saved arrays from an immutable Git revision and re-evaluate them.

No partner Python code is executed and no training settings are changed.
The shared evaluator's configuration describes evaluation, not partner training.
"""

import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.analysis import evaluate, plot_evaluation
from src.config import Config

SELECTED = (
    "main_sigma0.2_seed0",
    "main_sigma1.0_seed0",
    "main_sigma1.0_seed1",
    "sweep_sigma0.4_seed0",
    "sweep_sigma0.4_seed1",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision")
    parser.add_argument("--snapshot", type=Path, help="Packaged Nathan directory with SNAPSHOT.json; avoids requiring Git")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.snapshot:
        manifest = json.loads((args.snapshot / "SNAPSHOT.json").read_text())
        revision = manifest["revision"]
        if args.revision and args.revision != revision:
            parser.error("Snapshot revision does not match requested revision")
    else:
        if not args.revision:
            parser.error("Supply --revision or --snapshot")
        revision = subprocess.check_output(
            ["git", "rev-parse", args.revision], text=True
        ).strip()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    def blob(path):
        if args.snapshot:
            data = (args.snapshot / path).read_bytes()
            if hashlib.sha256(data).hexdigest() != manifest["files_sha256"][path]:
                raise ValueError(f"Snapshot checksum mismatch: {path}")
            return data
        return subprocess.check_output(["git", "show", f"{revision}:{path}"])

    paths = list(manifest["files_sha256"]) if args.snapshot else subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision, "results/runs"],
        text=True,
    ).splitlines()
    inventory = []
    for path in sorted(paths):
        if not path.endswith("/config.json"):
            continue
        base = path.rsplit("/", 1)[0]
        c = json.loads(blob(path))
        metrics_path = base + "/metrics.json"
        m = json.loads(blob(metrics_path)) if metrics_path in paths else {}
        with np.load(io.BytesIO(blob(base + "/ckpt.npz")), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
        inventory.append({
            "name": base.split("/")[-1], "steps": meta["t_done"], "budget": c["steps"],
            "sigma": c["sigma"], "spatial_scale": c["l"], "neurons": c["N"],
            "frequencies": c["D"], "seed": c["seed"],
            "reported_hex_modules": m.get("n_hex_modules"),
            "reported_grid_fraction_analytic": m.get("frac_grid_analytic"),
            "reported_grid_fraction_windowed": m.get("frac_grid_windowed"),
        })
    with (out / "inventory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=inventory[0])
        writer.writeheader()
        writer.writerows(inventory)
    summaries = []
    for name in SELECTED:
        folder = out / name
        folder.mkdir(exist_ok=True)
        raw = blob(f"results/runs/{name}/ckpt.npz")
        digest = hashlib.sha256(raw).hexdigest()
        if (folder / "summary.json").exists():
            old = json.loads((folder / "summary.json").read_text())
            if old["partner_checkpoint_sha256"] != digest:
                raise ValueError("Existing output belongs to another checkpoint")
            summaries.append(old)
            continue
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            c = meta["cfg"]
            p = {k: z["p_" + k].copy() for k in ("a0", "A", "B", "K")}
        if (c["dim"], c["norm_mode"], c["occupancy"], c["functional"]) != (
            2, "sum", "gaussian", "full"
        ) or meta["t_done"] != c["steps"]:
            raise ValueError("Expected a completed compatible 2D run")
        weights = np.empty((c["N"], 2 * c["D"] + 1), dtype=p["A"].dtype)
        weights[:, 0] = p["a0"]
        weights[:, 1::2] = p["B"].T
        weights[:, 2::2] = p["A"].T
        # Verify the basis conversion independently on deterministic positions.
        x = np.random.default_rng(713).normal(size=(17, 2))
        phi = x @ p["K"].T
        expected = p["a0"] + np.sin(phi) @ p["A"] + np.cos(phi) @ p["B"]
        basis = np.ones((len(x), weights.shape[1]))
        basis[:, 1::2], basis[:, 2::2] = np.cos(phi), np.sin(phi)
        np.testing.assert_allclose(basis @ weights.T, expected, rtol=1e-12, atol=1e-10)
        state = SimpleNamespace(
            weights=weights, frequencies=p["K"], step=meta["t_done"],
            log_multipliers=np.log([meta["gp"][0], meta["gn"][0]]),
        )
        cfg = Config(
            neurons=c["N"], frequencies=c["D"], sigma=c["sigma"],
            spatial_scale=c["l"], occupancy_scale=c["L"], samples=c["M"],
            shifts=c["MS"], shift_scale=c["S"], steps=c["steps"], seed=c["seed"],
        ).validate()
        print("Common evaluation:", name, flush=True)
        summary, maps, sacs, _axis = evaluate(state, cfg, resolution=352, extent=1.5)
        summary.pop("geco_clipping_active")  # Our clipping rule is not Nathan's.
        summary.update(
            partner_revision=revision, partner_checkpoint_sha256=digest,
            partner_training_config=c, name=name,
            evaluation_note="Common 0.02 norm tolerance; config describes evaluator only, not partner optimiser. No retraining. Fixed window may underrepresent broad grids.",
        )
        plot_evaluation(summary, maps, sacs, weights, p["K"], folder)
        (folder / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        summaries.append(summary)
    fields = ["name", "step", "grid_score_median", "positivity_loss", "norm_loss",
              "negative_fraction_below_minus_1e_minus6", "maximum_axis_frequency"]
    with (out / "common_evaluation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Inventoried {len(inventory)} runs; evaluated {len(summaries)} checkpoints.")


if __name__ == "__main__":
    main()
