"""Command-line entry point: prepare, train one fold, run CV, or summarize."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data import CONTRASTS, ENVIRONMENTS, ROOT, prepare_cache, source_directory


def parser() -> argparse.ArgumentParser:
    """Define explicit dataset preparation and fine-tuning commands."""
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Preprocess MNE files once into a reusable array cache")
    prepare.add_argument("--dataset", choices=ENVIRONMENTS, required=True)
    prepare.add_argument("--data-dir", type=Path)
    prepare.add_argument("--cache", type=Path, required=True)
    prepare.add_argument("--contrast", choices=CONTRASTS, default="solo-coordination")
    prepare.add_argument("--bad-segments", type=Path, help="Trusted local Circling exclusion pickle")
    prepare.add_argument("--fg-endpoint", choices=["strict", "trim"], default="strict")
    for command in ("train", "cv"):
        run = commands.add_parser(command, help="Fine-tune one fold" if command == "train" else "Run all ten folds")
        run.add_argument("--cache", type=Path, required=True)
        run.add_argument("--output", type=Path, required=True)
        run.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "labram-base.pth")
        if command == "train":
            run.add_argument("--fold", type=int, choices=range(10), required=True)
        run.add_argument("--epochs", type=int, default=10)
        run.add_argument("--warmup-epochs", type=int, default=5)
        run.add_argument("--batch-size", type=int, default=128)
        run.add_argument("--update-freq", type=int, default=2)
        run.add_argument("--lr", type=float, help="Default: 1e-4 for MG; 5e-4 for FG/Circling")
        run.add_argument("--min-lr", type=float, default=1e-6)
        run.add_argument("--layer-decay", type=float, default=0.65)
        run.add_argument("--weight-decay", type=float, default=0.05)
        run.add_argument("--drop-path", type=float, default=0.2)
        run.add_argument("--seed", type=int, default=0)
        run.add_argument("--num-workers", type=int, default=4)
        run.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        run.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
        run.add_argument(
            "--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "disabled")
        )
        run.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "labram-classification"))
        run.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
        run.add_argument("--wandb-group", default=os.environ.get("WANDB_RUN_GROUP"))
        run.add_argument(
            "--wandb-log-every",
            type=int,
            default=20,
            help="Log training progress every N optimizer updates, plus the last update of each epoch",
        )
        run.add_argument(
            "--smoke-windows-per-class",
            type=int,
            default=0,
            help="Diagnostic subset per partition; results excluded from CV summaries",
        )
    summary = commands.add_parser("summarize", help="Summarize ten completed full-data folds")
    summary.add_argument("--output", type=Path, required=True)
    plot = commands.add_parser("plot", help="Plot ten-fold train/validation/test mean ± SD curves")
    plot.add_argument("--output", type=Path, required=True, help="Completed CV experiment directory")
    plot.add_argument("--figures-dir", type=Path, help="Default: OUTPUT/figures")
    baseline = commands.add_parser("baseline", help="Fit log-band-power logistic regression on grouped folds")
    baseline.add_argument("--cache", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--fold", type=int, choices=range(10), help="Omit to run all ten folds")
    baseline.add_argument("--feature-mode", choices=["bandpower", "time-bandpower"], default="bandpower")
    baseline.add_argument("--c-values", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    baseline.add_argument("--max-iter", type=int, default=2000)
    baseline.add_argument("--seed", type=int, default=0)
    baseline.add_argument("--num-threads", type=int, default=4)
    baseline.add_argument("--feature-batch-size", type=int, default=64)
    baseline.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "disabled")
    )
    baseline.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "labram-classification"))
    baseline.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    baseline.add_argument("--wandb-group", default=os.environ.get("WANDB_RUN_GROUP"))
    return root


def main() -> None:
    """Validate arguments and dispatch with optional experiment tracking."""
    cli = parser()
    args = cli.parse_args()
    if args.command == "prepare":
        prepare_cache(
            source_directory(args.dataset, args.data_dir),
            args.cache,
            args.dataset,
            args.contrast,
            args.bad_segments,
            args.fg_endpoint,
        )
        return
    if args.command == "baseline":
        from .baseline import run_baseline

        options = {key: value for key, value in vars(args).items() if key not in {"command", "cache", "output", "fold"}}
        options["smoke_windows_per_class"] = 0
        run_baseline(args.cache, args.output, options, args.fold)
        return
    if args.command == "plot":
        from .plotting import plot_curves

        print(f"Saved figures and tables: {plot_curves(args.output, args.figures_dir)}")
        return
    from .runner import run_fold, summarize

    if args.command == "summarize":
        print(json.dumps(summarize(args.output), indent=2))
        return
    import math

    options = {
        key: value
        for key, value in vars(args).items()
        if key not in {"command", "cache", "output", "checkpoint", "fold"}
    }
    if (
        min(args.batch_size, args.update_freq, args.epochs) < 1
        or not 0 <= args.warmup_epochs < args.epochs
        or args.wandb_mode not in {"online", "offline", "disabled"}
        or args.wandb_log_every < 1
        or args.num_workers < 0
        or args.smoke_windows_per_class < 0
        or not 0 < args.layer_decay <= 1
        or not 0 <= args.drop_path < 1
        or args.weight_decay < 0
        or args.min_lr < 0
        or (args.lr is not None and args.lr <= 0)
        or any(isinstance(value, float) and not math.isfinite(value) for value in options.values())
    ):
        cli.error("Invalid training options: require positive sizes/LR, 0 <= warmup < epochs, and finite values.")
    if args.command == "train":
        run_fold(args.cache, args.output, args.checkpoint, args.fold, options)
    else:
        if args.output.exists():
            cli.error("CV output already exists; use a new directory or run missing folds individually.")
        for fold in range(10):
            run_fold(args.cache, args.output / f"fold_{fold}", args.checkpoint, fold, options)
        if not args.smoke_windows_per_class:
            print(json.dumps(summarize(args.output), indent=2))


if __name__ == "__main__":
    main()
