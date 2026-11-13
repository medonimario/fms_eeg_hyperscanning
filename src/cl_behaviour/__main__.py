"""Command line for audited Circling EEG–movement preparation."""

import argparse
from pathlib import Path


def main() -> None:
    """Prepare signals or freeze a metadata-only retrieval protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for stage in ("audit", "prepare"):
        command = commands.add_parser(stage)
        command.add_argument("--data-dir", type=Path)
        command.add_argument("--phase", type=Path, required=True, help="Trusted original phase_list.pickle.")
        command.add_argument("--bad-segments", type=Path)
        command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("split", help="Freeze whole-repetition splits and self-movement galleries.")
    command.add_argument("--cache", type=Path, required=True)
    command.add_argument("--cohort-manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--seed", type=int, default=2042)
    command.add_argument("--profile", choices=("combined-closed", "separate-closed", "open"), default="combined-closed")
    command = commands.add_parser("features", help="Extract frozen EEG tokens and aligned movement features.")
    command.add_argument("--cache", type=Path, required=True)
    command.add_argument("--run", type=Path, required=True, help="Completed split directory.")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--checkpoint", type=Path, default=Path("models/labram-base.pth"))
    command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    command.add_argument(
        "--batch-size", type=int, default=8, help="Paired windows per batch (twice as many EEG inputs)."
    )
    command.add_argument(
        "--smoke", action="store_true", help="Two training trials from the first development dyad only."
    )
    command = commands.add_parser("controls", help="Run self-contained synthetic correspondence checks on CPU.")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    command.add_argument("--steps", type=int, default=250)
    command = commands.add_parser("tiny-fit", help="Memorize one training-only real-feature gallery on CPU.")
    command.add_argument("--features", type=Path, required=True)
    command.add_argument("--run", type=Path, required=True, help="Approved split directory.")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--pairs", type=int, default=8)
    command.add_argument("--group", type=int)
    command.add_argument("--member", choices=("a", "b"), default="a")
    command.add_argument("--offset", type=int, choices=range(0, 15, 2), default=0)
    command.add_argument(
        "--readouts", nargs="+", choices=("pooled", "temporal", "full"), default=["pooled", "temporal", "full"]
    )
    command.add_argument("--steps", type=int, default=500)
    command.add_argument("--seed", type=int, default=42)
    command = commands.add_parser("plot", help="Plot a completed tiny-fit history without retraining.")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("train", help="Train shared heads on fixed within-person galleries.")
    command.add_argument("--features", type=Path, required=True)
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument(
        "--readouts", nargs="+", choices=("pooled", "temporal", "full"), default=["pooled", "temporal", "full"]
    )
    command.add_argument("--epochs", type=int, default=20)
    command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--groups-per-update", type=int, default=4)
    command.add_argument("--learning-rate", type=float, default=0.001)
    command.add_argument("--weight-decay", type=float, default=0.01)
    command.add_argument("--temperature", type=float, default=0.1)
    command = commands.add_parser("ridge", help="Shared and participant-specific fixed-target ridge retrieval.")
    command.add_argument("--features", type=Path, required=True)
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--lambdas", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    command.add_argument("--seed", type=int, default=42)
    command = commands.add_parser("whole-trial", help="Matched eight-window local/trial hard/soft comparison.")
    command.add_argument("--features", type=Path, required=True)
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--epochs", type=int, default=20)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    command.add_argument("--learning-rate", type=float, default=0.001)
    command.add_argument("--weight-decay", type=float, default=0.1)
    command.add_argument("--temperature", type=float, default=0.1)
    command.add_argument("--alpha", type=float, default=0.5)
    command.add_argument("--effective-positives", type=float, default=2.0)
    command.add_argument("--groups-per-update", type=int, default=4)
    command = commands.add_parser("temporal-trial", help="Temporal EEG and continuous full-trial movement encoders.")
    command.add_argument("--features", type=Path, required=True)
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--epochs", type=int, default=20)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    command.add_argument("--learning-rate", type=float, default=0.0001)
    command.add_argument("--weight-decay", type=float, default=0.1)
    command.add_argument("--temperature", type=float, default=0.1)
    command.add_argument("--groups-per-update", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "temporal-trial":
        from .temporal_trial import run_temporal_trial

        run_temporal_trial(
            args.features,
            args.run,
            args.output,
            args.epochs,
            args.seed,
            args.device,
            args.learning_rate,
            args.weight_decay,
            args.temperature,
            args.groups_per_update,
        )
        return
    if args.stage == "whole-trial":
        from .whole_trial import run_whole_trial

        run_whole_trial(
            args.features,
            args.run,
            args.output,
            args.epochs,
            args.seed,
            args.device,
            args.learning_rate,
            args.weight_decay,
            args.temperature,
            args.alpha,
            args.effective_positives,
            args.groups_per_update,
        )
        return
    if args.stage == "ridge":
        from .ridge import run_ridge

        run_ridge(args.features, args.run, args.output, tuple(args.lambdas), args.seed)
        return
    if args.stage == "train":
        from .training import run_training

        run_training(
            args.features,
            args.run,
            args.output,
            tuple(args.readouts),
            args.epochs,
            args.device,
            args.seed,
            args.groups_per_update,
            args.learning_rate,
            args.weight_decay,
            args.temperature,
        )
        return
    if args.stage == "plot":
        from .plotting import plot_tiny_run

        plot_tiny_run(args.run, args.output)
        return
    if args.stage == "tiny-fit":
        from .tiny_fit import run_tiny_fit

        run_tiny_fit(
            args.features,
            args.run,
            args.output,
            args.pairs,
            args.group,
            args.member,
            args.offset,
            tuple(args.readouts),
            args.steps,
            args.seed,
        )
        return
    if args.stage == "controls":
        from .controls import run_controls

        run_controls(args.output, tuple(args.seeds), args.steps)
        return
    if args.stage == "features":
        from .features import extract_features

        extract_features(args.cache, args.run, args.output, args.checkpoint, args.device, args.batch_size, args.smoke)
        return
    if args.stage == "split":
        from .splits import split_cache

        split_cache(args.cache, args.cohort_manifest, args.output, args.seed, args.profile)
        return
    from hyperscanning_contrastive.data import resolve_source

    from .data import prepare

    directory = resolve_source(args.data_dir)
    bad = args.bad_segments or directory.parent / "dict_with_short_bad_epochs.pickle"
    prepare(directory, args.phase, bad, args.output, args.stage == "prepare")


if __name__ == "__main__":
    main()
