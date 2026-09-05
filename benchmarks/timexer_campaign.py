#!/usr/bin/env python3
"""Run the frozen three-seed comparison protocol under one exclusive mlq job."""
import argparse
import math
from pathlib import Path
import re
import subprocess

SEEDS = (20260904, 20260905, 20260906)
MODELS = (
    "single-ticker-timexer", "patch-tst", "d-linear", "n-linear",
    "bar-trunk", "raw-timexer",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.campaign):
        parser.error("campaign must be a simple run-name prefix")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be positive and finite")
    root = Path(__file__).resolve().parents[1]
    output = root / "training/evaluations" / args.campaign
    if output.exists():
        parser.error(f"campaign output already exists: {output}")
    launcher = root / "torch-env.sh"
    subprocess.run([str(launcher), "cargo", "build", "--release", "-p", "trading_bot_0"], cwd=root, check=True)
    command = [str(root / "trading_bots/run-release-cuda.sh")]
    output.mkdir(parents=True)
    evidence = {model: [] for model in MODELS}
    data = ["--data-dir", str(args.data_dir.resolve())] if args.data_dir else []

    def run(*arguments):
        subprocess.run(command + list(arguments), cwd=root, check=True)

    for model in MODELS:
        for seed in SEEDS:
            name = f"{args.campaign}-{model}-{seed}"
            weights = root / "training/runs" / name / "weights"
            frozen = weights / "frozen"
            run("train-timexer", "--ticker", args.ticker, "--model", model,
                "--run", name, "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size), "--learning-rate", str(args.learning_rate),
                "--seed", str(seed), *data)
            run("freeze-timexer", "--checkpoint", str(weights / "best"), "--output", str(frozen))
            destination = output / model / str(seed)
            statistical = ["--statistical-baselines"] if model == MODELS[0] and seed == SEEDS[0] else []
            run("evaluate-timexer", "--checkpoint", str(frozen), "--split", "validation",
                "--batch-size", str(args.batch_size), "--output", str(destination), *statistical, *data)
            evidence[model].append(str(destination / "0/timexer_evidence.report.bin"))
            # Calibration remains a separate chronological partition and never fits weights.
            run("evaluate-timexer", "--checkpoint", str(frozen), "--split", "calibration",
                "--batch-size", str(args.batch_size), "--output", str(destination / "calibration"), *data)

    run("compare-timexer", "--candidate", *evidence["single-ticker-timexer"],
        "--baseline", *evidence["bar-trunk"], "--patch-tst", *evidence["patch-tst"],
        "--output", str(output / "gates"))
    print(f"All comparisons and gate reports: {output}. Terminal test was not evaluated.")


if __name__ == "__main__":
    main()
