"""Command line for audited Circling EEG–movement preparation."""

import argparse
from pathlib import Path

from hyperscanning_contrastive.data import resolve_source

from .data import prepare
from .splits import PROFILES, split_cache


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
    command.add_argument("--profile", choices=PROFILES, default="combined-closed")
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
    args = parser.parse_args()
    if args.stage == "features":
        from .features import extract_features

        extract_features(args.cache, args.run, args.output, args.checkpoint, args.device, args.batch_size, args.smoke)
        return
    if args.stage == "split":
        split_cache(args.cache, args.cohort_manifest, args.output, args.seed, args.profile)
        return
    directory = resolve_source(args.data_dir)
    bad = args.bad_segments or directory.parent / "dict_with_short_bad_epochs.pickle"
    prepare(directory, args.phase, bad, args.output, args.stage == "prepare")


if __name__ == "__main__":
    main()
