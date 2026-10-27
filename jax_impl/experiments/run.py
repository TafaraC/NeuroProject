"""Run with: python -m experiments.run --config ... --output ..."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import jax
import numpy as np

from src.config import load_config
from src.training import initialise, load_checkpoint, save_checkpoint, train_block


def metadata(config):
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    try:
        revision, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, status = None, "Git metadata unavailable"
    files = sorted([*root.glob("src/*.py"), *root.glob("experiments/*.py")])
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "config": config.to_dict(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "devices": [str(d) for d in jax.devices()],
        "git_revision": revision,
        "git_status": status,
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("jax", "jaxlib", "numpy", "scipy", "matplotlib")
        },
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--steps", type=int, help="Total configured training budget (new runs only)"
    )
    parser.add_argument(
        "--stop-at",
        type=int,
        help="Pause at this step without changing the configured budget",
    )
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--snapshot-steps",
        nargs="+",
        type=int,
        default=[],
        help="Also retain checkpoints at these steps (including across block boundaries)",
    )
    args = parser.parse_args()
    if args.block_size < 1:
        parser.error("block-size must be positive")
    output = args.output
    if args.resume:
        if args.config or args.seed is not None or args.steps is not None:
            parser.error("Resume uses the saved configuration; do not override it")
        if args.resume.resolve() != (output / "checkpoint.npz").resolve():
            parser.error("Resume must use the checkpoint in the output directory")
        state, config = load_checkpoint(args.resume)
        # Refuse a silent change of implementation halfway through a run.
        old = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        current = metadata(config)
        if (
            old["source_sha256"] != current["source_sha256"]
            or old["packages"] != current["packages"]
        ):
            parser.error(
                "Implementation or package versions changed since the checkpoint; start a new run"
            )
    else:
        if args.config is None:
            parser.error("--config is required for a new run")
        config = load_config(args.config)
        if args.seed is not None:
            config = replace(config, seed=args.seed)
        if args.steps is not None:
            config = replace(config, steps=args.steps)
        config.validate()
        if output.exists() and any(output.iterdir()):
            parser.error(
                "Output directory is not empty; choose a new one or use --resume"
            )
        state = initialise(config)
    target = args.stop_at if args.stop_at is not None else config.steps
    if any(s < 1 or s > config.steps for s in args.snapshot_steps):
        parser.error("snapshot steps must be within the configured training budget")
    if not int(state.step) <= target <= config.steps:
        parser.error(
            "stop-at must be between the current step and the configured budget"
        )
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
    if not args.resume:
        (output / "metadata.json").write_text(
            json.dumps(metadata(config), indent=2), encoding="utf-8"
        )
        (output / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2), encoding="utf-8"
        )
        save_checkpoint(output / "initial.npz", state, config)
        save_checkpoint(output / "checkpoint.npz", state, config)
    # A crash between log append and checkpoint replacement can leave a tail.
    log_path = output / "history.jsonl"
    if args.resume and log_path.exists():
        retained = [
            line
            for line in log_path.read_text().splitlines()
            if int(json.loads(line)["step"]) <= int(state.step)
        ]
        log_path.write_text(
            "\n".join(retained) + ("\n" if retained else ""), encoding="utf-8"
        )
    start = time.perf_counter()
    while int(state.step) < target:
        begin = int(state.step)
        block_start = time.perf_counter()
        next_boundary = min(
            [target, *[s for s in args.snapshot_steps if begin < s <= target]]
        )
        candidate, metrics = train_block(
            state, config, min(args.block_size, next_boundary - begin)
        )
        metrics = np.asarray(metrics)  # synchronise compiled work before timing
        if not np.isfinite(metrics).all() or not all(
            np.isfinite(np.asarray(a)).all() for a in candidate
        ):
            raise FloatingPointError(
                "Non-finite training state; last valid checkpoint retained"
            )
        state = candidate
        row = dict(
            zip(
                (
                    "objective",
                    "functional",
                    "positivity",
                    "norm",
                    "lambda_pos",
                    "lambda_norm",
                ),
                metrics[-1].tolist(),
            )
        )
        row.update(
            step=int(state.step),
            loss_parameter_step=int(state.step) - 1,
            block_start=begin,
            block_seconds=time.perf_counter() - block_start,
            maximum_axis_frequency=float(np.max(np.abs(np.asarray(state.frequencies)))),
            geco_clipping_active=bool(
                np.any(np.abs(np.asarray(state.log_multipliers)) >= 19.99)
            ),
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        save_checkpoint(output / "checkpoint.npz", state, config)
        if int(state.step) in args.snapshot_steps:
            save_checkpoint(
                output / "snapshots" / f"step_{int(state.step):06d}.npz", state, config
            )
        print(
            f"step={int(state.step):6d} sep={row['functional']:.5f} pos={row['positivity']:.3g} "
            f"norm={row['norm']:.3g} seconds={row['block_seconds']:.2f}",
            flush=True,
        )
    result = {
        "step": int(state.step),
        "budget": config.steps,
        "status": "complete" if int(state.step) == config.steps else "paused",
        "invocation_seconds": time.perf_counter() - start,
    }
    (output / "status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.evaluate:
        from src.analysis import save_evaluation

        summary = save_evaluation(
            state, config, output / f"evaluation_{int(state.step):06d}"
        )
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        "step",
                        "grid_score_median",
                        "positivity_loss",
                        "norm_loss",
                    )
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
