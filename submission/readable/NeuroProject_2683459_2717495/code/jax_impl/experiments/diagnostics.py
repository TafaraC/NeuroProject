"""A predeclared, matched-seed study of baseline numerical conventions.

python -m experiments.diagnostics plan --output results/data/diagnostics
python -m experiments.diagnostics run --output results/data/diagnostics --workers 2
python -m experiments.diagnostics report --output results/data/diagnostics
"""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import Config, load_config

CONDITIONS = {
    "baseline": {},
    "sigma_variance": {"sigma": math.sqrt(0.4)},
    "norm_table": {"norm_target": math.exp(4)},
}
RATIONALE = {
    "baseline": "Original equation-based implementation: sigma=0.4; norm target=0.02.",
    "sigma_variance": "Interpret 0.4 as variance, as the public sigma_sq argument does; change sigma only.",
    "norm_table": "Interpret table B.3 k_norm=4 literally in log(loss)-k: target=exp(4); change target only.",
}
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def make_plan(base, seeds):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Supply distinct training seeds")
    base.validate()
    runs = []
    for seed in seeds:
        for condition, changes in CONDITIONS.items():
            c = replace(base, seed=seed, **changes).validate()
            runs.append(
                {
                    "id": f"{condition}_seed{seed}",
                    "condition": condition,
                    "config": c.to_dict(),
                }
            )
    plan = {
        "version": 1,
        "question": "Do sigma interpretation or the literal published norm target explain the baseline failure?",
        "rationale": RATIONALE,
        "baseline": base.to_dict(),
        "seeds": seeds,
        "runs": runs,
        "evaluation": {
            "seed": 12345,
            "batches": 8,
            "extent": 1.5,
            "minimum_resolution": 320,
        },
        "common_positivity_target": base.positivity_target,
        "common_norm_target": base.norm_target,
        "interpretation": "Final checkpoints only. Runs, not neurons, are replicates. No arm is an exact upstream rerun. A relaxed target passing is not biological success.",
    }
    validate_plan(plan)
    return plan


def validate_plan(plan):
    """Reject missing pairs, duplicate runs and accidental multi-factor changes."""
    base = Config(**plan["baseline"]).validate()
    seeds = plan["seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Training seeds must be distinct")
    expected = {(name, seed) for name in CONDITIONS for seed in seeds}
    actual = set()
    ids = set()
    for run in plan["runs"]:
        name = run["condition"]
        c = Config(**run["config"]).validate()
        key = (name, c.seed)
        if name not in CONDITIONS or key not in expected or key in actual:
            raise ValueError("Study contains an unknown or duplicate condition/seed")
        if run["id"] != f"{name}_seed{c.seed}" or run["id"] in ids:
            raise ValueError("Invalid or duplicate run identifier")
        target = replace(base, seed=c.seed, **CONDITIONS[name])
        if c != target:
            raise ValueError("Condition changes more than its declared factor")
        actual.add(key)
        ids.add(run["id"])
    if actual != expected:
        raise ValueError("Study is missing a matched condition/seed")
    if (
        plan["common_norm_target"] != base.norm_target
        or plan["common_positivity_target"] != base.positivity_target
    ):
        raise ValueError("Common assessment targets must match baseline")
    return plan


def prepare(output, base, seeds):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Plan output must be empty; use a new directory")
    plan = make_plan(base, seeds)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "plan.json", plan)
    (output / "configs").mkdir()
    for run in plan["runs"]:
        write_json(output / "configs" / f"{run['id']}.json", run["config"])
    return plan


def run_study(output, plan, workers):
    from experiments.run import metadata

    def execute(run):
        folder = output / "runs" / run["id"]
        config = Config(**run["config"])
        config_path = output / "configs" / f"{run['id']}.json"
        if read_json(config_path) != run["config"]:
            raise ValueError("Run configuration differs from frozen plan")
        command = [
            sys.executable,
            "-m",
            "experiments.run",
            "--output",
            str(folder),
            "--block-size",
            "5000",
        ]
        snapshots = [s for s in (25000, 75000) if s < config.steps]
        if snapshots:
            command += ["--snapshot-steps", *map(str, snapshots)]
        if (folder / "checkpoint.npz").exists():
            old = read_json(folder / "metadata.json")
            current = metadata(config)
            if (
                old["config"] != config.to_dict()
                or old["source_sha256"] != current["source_sha256"]
                or old["packages"] != current["packages"]
            ):
                raise ValueError(
                    f"Changed source/environment/config in {run['id']}; start a new study"
                )
            if (folder / "status.json").exists() and read_json(folder / "status.json")[
                "status"
            ] == "complete":
                print(f"Already completed: {run['id']}", flush=True)
                return
            command += ["--resume", str(folder / "checkpoint.npz")]
        else:
            command += ["--config", str(config_path)]
        logs = output / "logs"
        logs.mkdir(exist_ok=True)
        print(f"Starting {run['id']}", flush=True)
        with (logs / f"{run['id']}.log").open("a", encoding="utf-8") as stream:
            subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True
            )
        print(f"Completed {run['id']}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Consume all results so a failed subprocess cannot be reported as success.
        list(executor.map(execute, plan["runs"]))


def validate_runs(output, plan):
    """Check actual saved state/provenance, not only filenames or status labels."""
    from src.training import load_checkpoint

    signatures = []
    maximum = 0.0
    for run in plan["runs"]:
        folder = output / "runs" / run["id"]
        state, config = load_checkpoint(folder / "checkpoint.npz")
        meta = read_json(folder / "metadata.json")
        if config.to_dict() != run["config"] or meta["config"] != run["config"]:
            raise ValueError("Checkpoint or metadata does not match plan")
        if int(state.step) != config.steps:
            raise ValueError(f"Incomplete run: {run['id']}")
        signatures.append((meta["source_sha256"], meta["packages"]))
        maximum = max(maximum, float(np.abs(np.asarray(state.frequencies)).max()))
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Training source or package versions differ across the study")
    for seed in plan["seeds"]:
        initial = [
            load_checkpoint(output / "runs" / f"{name}_seed{seed}" / "initial.npz")[0]
            for name in CONDITIONS
        ]
        for other in initial[1:]:
            for left, right in zip(initial[0], other):
                if not np.array_equal(left, right):
                    raise ValueError(
                        "Matched conditions did not start from identical states"
                    )
    return maximum


def paired_rows(rows):
    """Each difference is within a seed; never treat neurons as repetitions."""
    baseline = {r["seed"]: r for r in rows if r["condition"] == "baseline"}
    pairs = []
    for row in rows:
        if row["condition"] == "baseline":
            continue
        reference = baseline[row["seed"]]
        pair = {"condition": row["condition"], "seed": row["seed"]}
        for field in (
            "grid_score_median",
            "norm_loss",
            "negative_fraction_below_minus_1e_minus6",
        ):
            pair[f"delta_{field}"] = (
                None
                if row[field] is None or reference[field] is None
                else row[field] - reference[field]
            )
        pairs.append(pair)
    return pairs


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(output, plan, max_resolution):
    from src.analysis import minimum_map_resolution, save_evaluation
    from src.training import load_checkpoint

    maximum = validate_runs(output, plan)
    e = plan["evaluation"]
    required = minimum_map_resolution(np.array([[maximum, 0]]), e["extent"])
    resolution = max(e["minimum_resolution"], math.ceil(required / 32) * 32)
    if resolution > max_resolution:
        raise ValueError(
            f"Shared grid needs {resolution}; explicitly increase --max-resolution (memory cost) to evaluate all runs equally"
        )
    destination = output / "report"
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            "Report already exists; preserve it and use a separate study for changed analysis"
        )
    destination.mkdir(exist_ok=True)
    write_json(
        destination / "evaluation_settings.json",
        {**e, "resolution": resolution, "plan_sha256": digest(output / "plan.json")},
    )
    rows = []
    for run in plan["runs"]:
        print(f"Evaluating {run['id']} at {resolution} x {resolution}", flush=True)
        checkpoint = output / "runs" / run["id"] / "checkpoint.npz"
        state, config = load_checkpoint(checkpoint)
        s = save_evaluation(
            state,
            config,
            destination / run["id"],
            seed=e["seed"],
            batches=e["batches"],
            extent=e["extent"],
            resolution=resolution,
        )
        fields = (
            "grid_score_median",
            "positivity_loss",
            "norm_loss",
            "negative_fraction_below_minus_1e_minus6",
            "room_energy_absolute_error_p95",
            "maximum_axis_frequency",
            "actionability_relative_error",
            "weight_condition_number",
        )
        rows.append(
            {
                "condition": run["condition"],
                "seed": config.seed,
                "steps": config.steps,
                **{key: s[key] for key in fields},
                "training_targets_met": s["positivity_target_met"]
                and s["norm_target_met"],
                "common_targets_met": s["positivity_loss"]
                <= plan["common_positivity_target"]
                and s["norm_loss"] <= plan["common_norm_target"],
                "checkpoint_sha256": digest(checkpoint),
            }
        )
    save_csv(destination / "runs.csv", rows)
    save_csv(destination / "paired_differences.csv", paired_rows(rows))
    plot_results(destination, rows)
    lines = [
        "# Baseline diagnostic study",
        "",
        plan["question"],
        "",
        plan["interpretation"],
        "",
        "| Condition | Seed | Median grid score | Norm error | Mean negativity | Common targets met? |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        score = (
            "undefined"
            if row["grid_score_median"] is None
            else f"{row['grid_score_median']:.4f}"
        )
        lines.append(
            f"| {row['condition']} | {row['seed']} | {score} | {row['norm_loss']:.5g} | {row['positivity_loss']:.5g} | {row['common_targets_met']} |"
        )
    lines += [
        "",
        "All planned final checkpoints are included. Same extent, resolution, evaluation seed and batches for every condition. Initial states were checked for exact equality within seed; training source and package versions were checked across runs.",
        "",
        "Common assessment uses the ORIGINAL baseline norm tolerance, including for the relaxed norm arm. Neither tolerance satisfaction nor a positive grid score establishes a biological grid module. Functional objectives with different sigma are not directly compared. No p-values or neuron-level confidence intervals are reported.",
        "",
        "The norm-table arm tests the literal log target in isolation; it does not also copy the upstream raw-rate norm formula or nested-log updates. Thus it is not a full upstream reproduction.",
        "",
        "Changing sigma or the target changes the numerical problem; effects do not by themselves identify a bug or disprove the paper. Inspect all-cell maps and constraints before deciding whether the baseline is resolved.",
    ]
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(destination / "status.json", {"status": "complete", "runs": len(rows)})


def plot_results(destination, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = list(CONDITIONS)
    for ax, field, title in zip(
        axes,
        ("grid_score_median", "norm_loss", "negative_fraction_below_minus_1e_minus6"),
        ("Median grid score", "Held-out norm error", "Fraction of negative rates"),
    ):
        for seed in sorted({r["seed"] for r in rows}):
            lookup = {r["condition"]: r[field] for r in rows if r["seed"] == seed}
            ax.plot(
                range(len(names)),
                [np.nan if lookup[n] is None else lookup[n] for n in names],
                "o-",
                label=f"seed {seed}",
            )
        ax.set_xticks(range(len(names)), names, rotation=20)
        ax.set_title(title)
    axes[1].set_yscale("log")
    axes[0].legend()
    fig.suptitle("Matched-seed diagnostic conditions | each line is one training seed")
    fig.tight_layout()
    fig.savefig(destination / "diagnostics.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/configs/baseline.json"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--steps",
        type=int,
        help="Override budget when creating a plan; short budgets are smoke checks only",
    )
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-resolution", type=int, default=768)
    args = parser.parse_args()
    output = args.output.resolve()
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
    if args.action == "plan":
        base = load_config(args.config)
        if args.steps is not None:
            base = replace(base, steps=args.steps).validate()
        plan = prepare(output, base, args.seeds)
        print(
            f"Saved {len(plan['runs'])} runs at {base.steps} steps each to {output / 'plan.json'}"
        )
    else:
        if args.steps is not None:
            parser.error(
                "Existing study budgets are frozen; create a new plan to change steps"
            )
        plan = validate_plan(read_json(output / "plan.json"))
        if args.action == "run":
            run_study(output, plan, args.workers)
        else:
            report(output, plan, args.max_resolution)


if __name__ == "__main__":
    main()
