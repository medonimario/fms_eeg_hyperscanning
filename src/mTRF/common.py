"""File layout and cohort shared by the Circling replication commands."""

import os
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv

PAIRS = tuple(pair for pair in range(2010, 2540, 10) if pair != 2400)
CONDITIONS = ("closed1", "open", "closed2")


def dataset_root(directory: Path | None) -> Path:
    """Resolve the dataset root using the repository .env or a command-line override."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    value = directory or os.environ.get("CIRCLING_DATASET_PATH")
    if not value:
        raise ValueError("Set CIRCLING_DATASET_PATH in .env or supply --data-dir.")
    return Path(value).expanduser().resolve()


def save_pickle(path: Path, data: object) -> None:
    """Publish a complete notebook-compatible pickle, including on a retried job."""
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        try:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(path)
