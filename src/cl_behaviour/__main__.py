"""Command line for audited Circling EEG–movement preparation."""

import argparse
from pathlib import Path

from hyperscanning_contrastive.data import resolve_source

from .data import prepare


def main() -> None:
    """Prepare or audit a new cache without learning features or assigning splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "prepare"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--phase", type=Path, required=True, help="Trusted original phase_list.pickle.")
    parser.add_argument("--bad-segments", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = resolve_source(args.data_dir)
    bad = args.bad_segments or directory.parent / "dict_with_short_bad_epochs.pickle"
    prepare(directory, args.phase, bad, args.output, args.stage == "prepare")


if __name__ == "__main__":
    main()
