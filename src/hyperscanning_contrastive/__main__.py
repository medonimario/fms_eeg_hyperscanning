"""Command-line entry point for the initial Circling data stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import CONDITIONS, prepare, resolve_source


def main() -> None:
    """Run a metadata audit or prepare paired EEG signals from the local dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "audit",
            "prepare",
            "controls",
            "overfit",
            "ssl",
            "ssl-evaluate",
            "ssl-correspondence",
            "partner-baseline",
            "partner-train",
            "partner-adapt",
        ),
    )
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
    parser.add_argument("--freeze-backbone", action="store_true", help="Stage 2 projection-only control.")
    parser.add_argument("--validation-fold", type=int, default=1)
    parser.add_argument("--test-fold", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--channel-drop", type=float, default=0.05)
    parser.add_argument("--run", type=Path, help="Completed Stage 2 training directory for ssl-evaluate.")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--permutation-seed", type=int, default=1042)
    parser.add_argument("--projection-profile", choices=("legacy-mlp", "centered-linear"), default="legacy-mlp")
    parser.add_argument("--unfreeze-blocks", type=int, default=2, help="Final transformer blocks for partner-adapt.")
    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="Encoder learning rate for partner-adapt.")
    args = parser.parse_args()
    if args.command == "partner-adapt":
        from .adaptation import train_adaptation

        if args.cache is None:
            parser.error("partner-adapt requires --cache")
        if args.projection_profile != "legacy-mlp":
            parser.error("partner-adapt uses the original legacy-mlp projection")
        if args.validation_fold != 1 or args.test_fold != 2:
            parser.error("partner-adapt currently fixes validation fold 1 and test fold 2")
        train_adaptation(
            args.cache,
            args.output,
            args.checkpoint,
            args.initialization,
            args.device,
            args.pairs,
            args.epochs,
            args.steps_per_epoch,
            args.seed,
            args.permutation_seed,
            args.unfreeze_blocks,
            args.backbone_lr,
        )
        return
    if args.command == "partner-train":
        from .alignment import train_alignment

        if args.cache is None:
            parser.error("partner-train requires --cache")
        train_alignment(
            args.cache,
            args.output,
            args.checkpoint,
            args.initialization,
            args.device,
            args.pairs,
            args.epochs,
            args.steps_per_epoch,
            args.seed,
            args.permutation_seed,
            args.projection_profile,
        )
        return
    if args.command == "partner-baseline":
        from .partner import partner_baseline

        if args.cache is None:
            parser.error("partner-baseline requires --cache")
        partner_baseline(
            args.cache,
            args.output,
            args.checkpoint,
            args.initialization,
            args.device,
            args.pairs,
            args.seed,
            args.permutations,
            args.permutation_seed,
        )
        return
    if args.command == "ssl-correspondence":
        from .correspondence import check_correspondence

        if args.cache is None or args.run is None:
            parser.error("ssl-correspondence requires --cache and --run")
        check_correspondence(args.cache, args.run, args.output, args.device, args.permutations, args.permutation_seed)
        return
    if args.command in {"ssl", "ssl-evaluate"}:
        from .ssl import SSLConfig, test_ssl, train_ssl

        if args.cache is None:
            parser.error("Stage 2 requires --cache")
        if args.command == "ssl-evaluate":
            if args.run is None:
                parser.error("ssl-evaluate requires --run")
            test_ssl(args.cache, args.run, args.output, args.device)
        else:
            config = SSLConfig(
                initialization=args.initialization,
                freeze_backbone=args.freeze_backbone,
                validation_fold=args.validation_fold,
                test_fold=args.test_fold,
                pairs=args.pairs,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
                seed=args.seed,
                amplitude=args.amplitude,
                noise=args.noise,
                channel_drop=args.channel_drop,
            )
            train_ssl(args.cache, args.output, args.checkpoint, config, args.device)
        print(f"Completed {args.command}: {args.output}")
        return
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
