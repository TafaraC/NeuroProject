"""Summarise two completed evaluations; report cells descriptively, not as replicates."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def compare(evaluations, output):
    summaries = [json.loads((p / "summary.json").read_text()) for p in evaluations]
    for summary in summaries:
        if summary["step"] != summary["config"]["steps"]:
            raise ValueError(
                "Comparison requires completed runs, not partial checkpoints"
            )
    comparable = (
        "neurons",
        "frequencies",
        "sigma",
        "seed",
        "steps",
        "samples",
        "shifts",
        "norm_target",
        "positivity_target",
        "functional_scale",
        "dtype",
    )
    for name in comparable:
        if summaries[0]["config"][name] != summaries[1]["config"][name]:
            raise ValueError(
                f"Comparison differs in {name}; this script expects a paired l comparison"
            )
    for name in (
        "evaluation_seed",
        "evaluation_batches",
        "map_resolution",
        "map_extent",
    ):
        if summaries[0][name] != summaries[1][name]:
            raise ValueError(f"Evaluation settings differ: {name}")
    output.mkdir(parents=True, exist_ok=True)
    fields = (
        "step",
        "functional_loss",
        "positivity_loss",
        "norm_loss",
        "positivity_target_met",
        "norm_target_met",
        "grid_score_median",
        "grid_score_q25",
        "grid_score_q75",
        "grid_score_positive_count",
        "grid_score_valid_count",
        "actionability_relative_error",
        "negative_fraction_below_minus_1e_minus6",
        "room_energy_absolute_error_p95",
    )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["spatial_scale", "seed", *fields])
        writer.writeheader()
        for s in summaries:
            writer.writerow(
                {
                    "spatial_scale": s["config"]["spatial_scale"],
                    "seed": s["config"]["seed"],
                    **{name: s[name] for name in fields},
                }
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, (directory, s) in enumerate(zip(evaluations, summaries)):
        with np.load(directory / "maps.npz") as maps:
            extent = [maps["axis"][0], maps["axis"][-1]] * 2
            axes[row, 0].imshow(
                maps["rates"][0], origin="lower", extent=extent, cmap="viridis"
            )
            axes[row, 1].imshow(
                maps["autocorrelations"][0],
                origin="lower",
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
            )
        scores = [d["score"] for d in s["grid_scores"] if d["score"] is not None]
        axes[row, 2].hist(scores, bins=np.linspace(-2, 2, 21), edgecolor="white")
        axes[row, 2].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[row, 2].set(
            xlabel="Descriptive grid score", ylabel="Neuron count", xlim=(-2, 2)
        )
        axes[row, 0].set_title(
            f"l={s['config']['spatial_scale']}: neuron 0 (fixed choice)"
        )
        axes[row, 1].set_title("Spatial autocorrelation of neuron 0")
        axes[row, 2].set_title(f"All neurons; median={s['grid_score_median']:.3f}")
    fig.suptitle(
        "Paired spatial-discrimination comparison | one training seed per condition\n"
        "Rate-map colour scales are per panel; neurons are not independent replicates"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for directory, summary in zip(evaluations, summaries):
        history = [
            json.loads(line)
            for line in (directory.parent / "history.jsonl").read_text().splitlines()
        ]
        for ax, metric in zip(axes, ("functional", "positivity", "norm")):
            ax.plot(
                [r["loss_parameter_step"] for r in history],
                [r[metric] for r in history],
                label=f"l={summary['config']['spatial_scale']}",
                linewidth=0.9,
            )
            ax.set(xlabel="Training step", ylabel=metric, yscale="log")
    axes[1].axhline(summaries[0]["positivity_target"], color="grey", linestyle="--")
    axes[2].axhline(summaries[0]["norm_target"], color="grey", linestyle="--")
    axes[0].legend()
    fig.suptitle(
        "Training losses sampled at block ends; constraints assessed separately on held-out points"
    )
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)
    text = [
        "# Independent reproduction comparison",
        "",
        "One paired training seed; descriptive results only.",
        "",
        "| l | Median grid score | Positivity loss | Norm loss | Both held-out targets met? |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        passed = s["positivity_target_met"] and s["norm_target_met"]
        text.append(
            f"| {s['config']['spatial_scale']} | {s['grid_score_median']:.4f} | "
            f"{s['positivity_loss']:.6g} | {s['norm_loss']:.6g} | {passed} |"
        )
    text += [
        "",
        "These are final checkpoints, not hand-selected best cells or seeds. All-cell maps are available in each evaluation directory.",
        "Grid scores maximise a rotational-correlation contrast over fixed annuli. They are not shuffled significance tests, and do not identify modules.",
        "A positive score does not establish a biologically valid representation. Read the held-out constraint checks alongside the maps.",
        "Repeated seeds, a direct module analysis, and biological validation remain outside this single-pair result.",
        "The implementation follows the paper's normalised norm loss, with an explicit 0.02 tolerance; see references/implementation_notes.md.",
    ]
    (output / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs=2, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
    compare(args.evaluations, args.output)


if __name__ == "__main__":
    main()
