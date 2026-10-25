"""Archive completed diagnostic evidence without the large regenerable map arrays."""

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

from experiments.diagnostics import read_json, validate_plan, validate_runs, write_json


def archive(source, destination):
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Archive must be outside the source study directory")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Archive destination must be empty")
    plan = validate_plan(read_json(source / "plan.json"))
    validate_runs(source, plan)
    status = read_json(source / "report/status.json")
    if status != {"status": "complete", "runs": len(plan["runs"])}:
        raise ValueError("Only fully completed and evaluated studies can be archived")
    with (source / "report/runs.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(run["condition"], str(run["config"]["seed"])) for run in plan["runs"]}
    if (
        len(rows) != len(expected)
        or {(r["condition"], r["seed"]) for r in rows} != expected
    ):
        raise ValueError(
            "Report does not contain exactly the planned condition/seed pairs"
        )
    for row in rows:
        checkpoint = (
            source / "runs" / f"{row['condition']}_seed{row['seed']}" / "checkpoint.npz"
        )
        if (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            != row["checkpoint_sha256"]
        ):
            raise ValueError("A checkpoint changed after its report was generated")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "plan.json", destination / "plan.json")
    shutil.copytree(source / "configs", destination / "configs")
    shutil.copytree(source / "runs", destination / "runs")
    shutil.copytree(source / "logs", destination / "logs")
    shutil.copytree(
        source / "report",
        destination / "report",
        ignore=shutil.ignore_patterns("maps.npz"),
    )
    checksums = {
        p.relative_to(destination).as_posix(): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    }
    write_json(
        destination / "archive_manifest.json",
        {
            "files_sha256": checksums,
            "omitted": "Large report/*/maps.npz files; regenerate by evaluating the saved checkpoints with report/evaluation_settings.json",
            "training_sources": "Exact source hashes and software versions are preserved in every run's metadata.json. Git revisions can differ while source hashes match.",
        },
    )
    print(
        f"Archived {len(plan['runs'])} completed runs and {len(checksums)} files to {destination}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
