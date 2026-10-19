"""Command-line entry point for the initial Circling data stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import CONDITIONS, prepare, resolve_source


def main() -> None:
    """Run a metadata audit or prepare paired EEG signals from the local dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "prepare", "controls", "overfit"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--bad-segments", type=Path, help="Trusted local six-second artifact pickle.")
    parser.add_argument("--output", type=Path, required=True, help="New audit/cache directory.")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--window-strategy", choices=("max-clean", "fixed-grid"), default="max-clean")
    parser.add_argument("--cache", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(__file__).resolve().parents[2] / "models/labram-base.pth"
    )
    parser.add_argument("--initialization", choices=("pretrained", "random"), default="pretrained")
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--development-fold", type=int, default=0)
    parser.add_argument("--group", type=int)
    parser.add_argument("--condition", choices=CONDITIONS, default="open")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "overfit":
        from .overfit import run_overfit

        if args.cache is None:
            parser.error("overfit requires --cache")
        report = run_overfit(
            args.cache,
            args.output,
            args.checkpoint,
            args.initialization,
            args.train_backbone,
            args.device,
            args.pairs,
            args.steps,
            args.development_fold,
            args.group,
            args.condition,
            args.seed,
        )
        print(f"Tiny-set overfit passed: {report['passed']}; report: {args.output / 'overfit.json'}")
        if not report["passed"]:
            raise SystemExit(1)
        return
    if args.command == "controls":
        from .diagnostics import run_controls

        report = run_controls(args.output)
        print(f"Engineering controls passed: {report['passed']}; report: {args.output / 'controls.json'}")
        if not report["passed"]:
            raise SystemExit(1)
        return
    directory = resolve_source(args.data_dir)
    summary = prepare(
        directory,
        args.output,
        args.bad_segments or directory.parent / "dict_with_short_bad_epochs.pickle",
        write_signals=args.command == "prepare",
        conditions=tuple(args.conditions),
        strategy=args.window_strategy,
    )
    print(f"Completed {args.command}: {summary['group_count']} dyads, {summary['paired_window_count']} paired windows.")


if __name__ == "__main__":
    main()
