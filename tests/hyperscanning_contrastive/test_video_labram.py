"""Frozen video representation provenance and extraction tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from hyperscanning_contrastive.data import CHANNELS, sha256
from hyperscanning_contrastive.video_labram import load_reference, run_video_labram


@pytest.fixture
def reference(tmp_path: Path) -> tuple[Path, Path]:
    """Make saved galleries with a reserved person whose EEG must never be opened."""
    cache, run = tmp_path / "cache", tmp_path / "reference"
    cache.mkdir()
    run.mkdir()
    windows, files = [], []
    rng = np.random.default_rng(4)
    times = {"train": [800, 2400, 4000, 5600], "validation": [8000, 9600, 11200, 12800]}
    for person in ("a", "b", "reserved"):
        path = f"{person}.npy"
        digest = "never-open"
        if person != "reserved":
            np.save(cache / path, rng.normal(size=(64, 13600)).astype("float32"))
            digest = sha256(cache / path)
        files.append(dict(array_path=path, array_sha256=digest))
        for split, starts in times.items():
            for start in starts:
                windows.append(
                    dict(
                        participant_id=person,
                        cache_start=start,
                        cache_stop=start + 800,
                        condition="solo",
                        video_id=0,
                        eligible=True,
                        array_path=path,
                        time_split=split,
                    )
                )
    pd.DataFrame(windows).to_csv(cache / "windows.csv", index=False)
    pd.DataFrame(files).to_csv(cache / "files.csv", index=False)
    metadata = dict(
        dataset="shared-stimulus-video",
        operation="prepare",
        channels=CHANNELS,
        sfreq=200,
        units="microvolts",
        model_input_divisor=100,
        window_seconds=4,
        manifest_sha256={p.name: sha256(p) for p in cache.glob("*.csv")},
    )
    (cache / "cache.json").write_text(json.dumps(metadata))
    pd.DataFrame(windows).to_csv(run / "window_splits.csv", index=False)
    pd.DataFrame(
        [
            dict(participant_id=p, split="reserved" if p == "reserved" else "development", condition="solo")
            for p in ("a", "b", "reserved")
        ]
    ).to_csv(run / "participant_splits.csv", index=False)
    pd.DataFrame([dict(split=s, first="a", second="b", starts=json.dumps(t)) for s, t in times.items()]).to_csv(
        run / "galleries.csv", index=False
    )
    pd.DataFrame(
        [
            dict(split=s, model=m, participant_id=p, top1=0.25, mrr=0.52, mean_rank=2.5)
            for s in times
            for m in ("waveform", "corrca", "misaligned_fit")
            for p in ("a", "b")
        ]
    ).to_csv(run / "person_metrics.csv", index=False)
    report = dict(
        cache_sha256=sha256(cache / "cache.json"),
        chance_top1=0.25,
        chance_mrr=sum(1 / np.arange(1, 5)) / 4,
        artifacts_sha256={p.name: sha256(p) for p in run.glob("*.csv")},
    )
    (run / "baseline.json").write_text(json.dumps(report))
    return cache, run


def test_frozen_extraction(reference, tmp_path, monkeypatch):
    """Keep original galleries, visit only required windows and bypass projection/weights for random."""
    seen = []

    class Encoder(torch.nn.Module):
        """Deterministic substitute for expensive LaBraM inference."""

        def __init__(self, initialization, checkpoint, train_backbone):
            """Assert random operation does not need the nonexistent checkpoint."""
            super().__init__()
            assert initialization == "random" and not train_backbone

        def features(self, x):
            """Return pooled features while checking eval mode and disabled gradients."""
            assert not self.training and not torch.is_grad_enabled()
            assert x.shape[1:] == (64, 800)
            seen.append(len(x))
            return x.mean(-1)

    monkeypatch.setattr("hyperscanning_contrastive.video_labram.ContrastiveLabram", Encoder)
    cache, run = reference
    output = tmp_path / "output"
    report = run_video_labram(cache, run, output, tmp_path / "missing.pth", "random", batch_size=3)
    assert sum(seen) == report["extracted_windows"] == 16
    assert max(seen) <= 3
    assert report["checkpoint_sha256"] is None
    assert sha256(run / "galleries.csv") == sha256(output / "galleries.csv")
    assert (output / "comparison.png").is_file()
    assert set(report["metrics"]["validation"]) == {"waveform", "corrca", "misaligned_fit", "labram_random"}


def test_reference_tampering(reference):
    """Reject changed reference files before extraction."""
    cache, run = reference
    with (run / "galleries.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        load_reference(cache, run)


def test_reserved_gallery_rejected(reference):
    """Even rehashed galleries cannot request a reserved participant's signals."""
    cache, run = reference
    galleries = pd.read_csv(run / "galleries.csv")
    galleries["second"] = "reserved"
    galleries.to_csv(run / "galleries.csv", index=False)
    report = json.loads((run / "baseline.json").read_text())
    report["artifacts_sha256"]["galleries.csv"] = sha256(run / "galleries.csv")
    (run / "baseline.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="reserved"):
        load_reference(cache, run)
