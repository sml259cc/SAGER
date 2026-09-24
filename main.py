#!/usr/bin/env python3
"""Run only the paper's main experiment, selecting on Dev before Test."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parent
DATASETS = ("iemocap_legacy7433", "meld_official")
SEEDS = (42, 19222, 831962)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--features-root", type=Path, default=REPO / "features")
    parser.add_argument("--output-root", type=Path, default=REPO / "runs/main")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training or creating outputs")
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    def absolute(path: Path) -> Path:
        path = path.expanduser()
        return path.resolve() if path.is_absolute() else (REPO / path).resolve()
    features = absolute(args.features_root)
    output = absolute(args.output_root)
    if not args.dry_run:
        missing = [features / ds / (split + ".rows.npz")
                   for ds in datasets for split in ("train", "dev", "test")
                   if not (features / ds / (split + ".rows.npz")).is_file()]
        if missing:
            parser.error("Frozen feature bundles are not included in this release. "
                         "See README.md. Missing: " + ", ".join(map(str, missing)))
    # Sequential execution also works on a single GPU; fail before the next run
    # when any train/Dev/Test process returns a nonzero exit code.
    for seed in args.seeds:
        for dataset in datasets:
            command = [sys.executable, str(REPO / "train.py"),
                       "--dataset", dataset, "--seed", str(seed),
                       "--config", str(REPO / "configs" / (dataset + ".yaml")),
                       "--bundle", str(features / dataset), "--device", args.device,
                       "--output", str(output / dataset / f"seed_{seed}"), "--auto-test"]
            if args.resume:
                command.append("--resume")
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=REPO, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
