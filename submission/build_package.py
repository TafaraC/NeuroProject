"""Build a first-deadline submission from the development checkout."""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from submission.verify_manifest import hashes

ROOT = Path(__file__).resolve().parents[1]
NATHAN = "df44d694e56ed91d5c4f5ea7499d1ca85d0bf046"
FOLDERS = (
    "src", "experiments", "comparison", "tests", "references", "submission",
    "review/paper_review", "results/reference_runs", "results/diagnostic_study",
    "results/constraint_control", "results/partner_comparison",
    "results/partner_results_20261026", "results/figures",
)


def copy_tree(source, destination):
    for p in source.rglob("*"):
        if not p.is_file() or any(x in p.parts for x in ("__pycache__", ".pytest_cache")):
            continue
        if source.parts[-2:] == ("review", "paper_review") and p.suffix in (
            ".log", ".aux", ".out", ".toc"
        ):
            continue
        target = destination / p.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)


def build(output):
    if output.exists() or output.with_suffix(".zip").exists():
        raise ValueError("Choose a new output path to preserve previous packages")
    output.mkdir(parents=True)
    for folder in FOLDERS:
        copy_tree(ROOT / folder, output / folder)
    for name in ("requirements.txt", "requirements-test.txt", "requirements-dev.txt"):
        shutil.copy2(ROOT / name, output / name)
    shutil.copy2(ROOT / "submission/README.txt", output / "README.txt")
    shutil.copy2(ROOT / "review/paper_review/main.pdf", output / "Review.pdf")
    shutil.copy2(ROOT / "review/contribution_statement.pdf", output / "Contribution_statement.pdf")
    shutil.copy2(ROOT / "review/contribution_statement.tex", output / "review/contribution_statement.tex")
    # Archive only the requested implementation/evidence paths, never its old ZIP.
    data = subprocess.check_output([
        "git", "-c", "core.autocrlf=false", "archive", "--format=zip", NATHAN, "src", "experiments", "tests",
        "requirements.txt", "results/runs", "results/tables", "results/figures", "review/tables",
    ], cwd=ROOT)
    nathan = output / "nathan"
    nathan.mkdir()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for item in archive.infolist():
            path = (nathan / item.filename).resolve()
            if not path.is_relative_to(nathan.resolve()):
                raise ValueError("Invalid snapshot path")
            if not item.is_dir():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(item))
    snapshot_hashes = hashes(nathan)
    (nathan / "SNAPSHOT.json").write_text(json.dumps({
        "revision": NATHAN, "files_sha256": snapshot_hashes,
    }, indent=2), encoding="utf-8")
    provenance = {
        "created_utc": datetime.now(UTC).isoformat(),
        "deadline": "2026-10-27 17:00 Africa/Johannesburg",
        "tafara_base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "nathan_commit": NATHAN,
        "note": "Includes current working-tree submission changes; MANIFEST.json identifies exact delivered bytes. No virtual environment, Git history or source paper is bundled.",
        "author_supplied_remaining": "Tafara will append the completed official AI declaration in LaTeX before uploading.",
    }
    (output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (output / "MANIFEST.json").write_text(json.dumps(hashes(output), indent=2), encoding="utf-8")
    zip_path = output.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.rglob("*")):
            if p.is_file():
                archive.write(p, output.name + "/" + p.relative_to(output).as_posix())
    print(f"Folder: {output}\nZIP: {zip_path}\nZIP SHA256: {hashlib.sha256(zip_path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output.resolve())
