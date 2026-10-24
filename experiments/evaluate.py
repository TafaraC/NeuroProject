"""Evaluate a saved checkpoint, including a step-zero initial checkpoint."""

import argparse
import json
import os
from pathlib import Path

from src.analysis import save_evaluation
from src.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--extent", type=float, default=5)
    args = parser.parse_args()
    if args.resolution < 32 or args.extent <= 0:
        parser.error("Use resolution >= 32 and extent > 0")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Evaluation output must be empty")
    os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
    state, config = load_checkpoint(args.checkpoint)
    summary = save_evaluation(
        state, config, args.output, resolution=args.resolution, extent=args.extent
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("step", "grid_score_median", "positivity_loss", "norm_loss")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
