"""Command-line entry point for the initial Circling data stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import CONDITIONS, prepare, resolve_source


def main() -> None:
    """Run a metadata audit or prepare paired EEG signals from the local dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "prepare"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--bad-segments", type=Path, help="Trusted local six-second artifact pickle.")
    parser.add_argument("--output", type=Path, required=True, help="New audit/cache directory.")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--window-strategy", choices=("max-clean", "fixed-grid"), default="max-clean")
    args = parser.parse_args()
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
