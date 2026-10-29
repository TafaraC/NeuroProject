"""Verify an extracted submission, or refresh hashes after author edits."""

import argparse
import hashlib
import json
from pathlib import Path


def hashes(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and p.name != "MANIFEST.json"
        and not any(x in p.parts for x in (".venv", "__pycache__", ".pytest_cache"))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / "MANIFEST.json"
    if args.refresh:
        target.write_text(json.dumps(hashes(root), indent=2), encoding="utf-8")
        print("Refreshed file manifest")
    else:
        expected = json.loads(target.read_text(encoding="utf-8"))
        bad = [n for n, h in expected.items() if not (root / n).is_file()
               or hashlib.sha256((root / n).read_bytes()).hexdigest() != h]
        if bad:
            raise SystemExit("Missing or changed files: " + ", ".join(bad))
        print(f"Verified {len(expected)} files")


if __name__ == "__main__":
    main()
