"""Checks for locked checkpoint transfer, provenance and external-fold isolation."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from test_ssl import TinyEncoder
from test_temporal import temporal_cache

from hyperscanning_contrastive import temporal, temporal_transfer
from hyperscanning_contrastive.ssl import EEGCache


def test_external_transfer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only external EEG is forwarded; saved normalization is reused and checkpoints never change."""
    monkeypatch.setattr(temporal, "ContrastiveLabram", TinyEncoder)
    monkeypatch.setattr(temporal, "plot_temporal", lambda _: None)
    monkeypatch.setattr(temporal_transfer, "ContrastiveLabram", TinyEncoder)
    cache = temporal_cache(tmp_path / "cache")
    run = tmp_path / "run"
    report = temporal.run_temporal(
        cache, run, tmp_path / "absent", initialization="random", pairs=2, epochs=1, temperatures=(0.05,)
    )
    original = EEGCache.signals
    forwarded = []

    def signals(self: EEGCache, rows: pd.DataFrame) -> torch.Tensor:
        """Check every signal read by evaluation is from fold one."""
        assert set(rows.fold) == {1}
        forwarded.extend(rows.trial_id)
        return original(self, rows)

    monkeypatch.setattr(EEGCache, "signals", signals)
    before = (run / "temperature_0.05/true/best_projection.pt").read_bytes()
    output = tmp_path / "external"
    result = temporal_transfer.evaluate_transfer(cache, run, output, tmp_path / "absent")
    assert len(forwarded) == 24
    assert result["paired_trials_evaluated"] == 6
    assert not result["test_evaluated"] and not result["training_performed"]
    assert result["source_best_epochs"]["true"] == report["experiments"]["temperature_0.05"]["true"]["best_epoch"]
    assert before == (run / "temperature_0.05/true/best_projection.pt").read_bytes()
    assert (output / "transfer_comparison.png").stat().st_size > 1000
    assert set(pd.read_csv(output / "galleries.csv").fold) == {1}
    assert len(pd.read_csv(output / "paired_dyad_differences.csv")) == 3
    legacy = json.loads((run / "temporal.json").read_text())
    legacy.pop("experiment_settings")
    (run / "temporal.json").write_text(json.dumps(legacy))
    second = temporal_transfer.evaluate_transfer(cache, run, tmp_path / "legacy", tmp_path / "absent")
    assert second["representations"] == result["representations"]
    (run / "temperature_0.05/true/best_projection.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance mismatch"):
        temporal_transfer.evaluate_transfer(cache, run, tmp_path / "bad", tmp_path / "absent")
    assert not (tmp_path / "bad").exists()
