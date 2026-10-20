"""Stage 2 checks for split isolation, independent augmentation, ranking, and checkpoint selection."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from hyperscanning_contrastive.data import CHANNELS, sha256
from hyperscanning_contrastive.ssl import (
    EEGCache,
    SSLConfig,
    augment,
    gallery_metrics,
    train_ssl,
)
from hyperscanning_contrastive.ssl import (
    test_ssl as evaluate_run,
)


def make_cache(path: Path) -> Path:
    """Build three dyads with distinct trial signals and complete minimal cache provenance."""
    path.mkdir()
    rng = np.random.default_rng(42)
    rows, hashes = [], {}
    for fold in range(3):
        name = f"pair_{fold}.npy"
        np.save(path / name, rng.normal(size=(12, 2, 64, 800)).astype(np.float32))
        hashes[name] = sha256(path / name)
        for condition_index, condition in enumerate(("open", "closed1", "closed2")):
            for trial in range(4):
                rows.append(
                    dict(
                        group=fold,
                        fold=fold,
                        condition=condition,
                        condition_trial=trial,
                        trial_id=f"{fold}:{condition}:{trial}",
                        crop_start_s=0,
                        array=name,
                        array_index=condition_index * 4 + trial,
                    )
                )
    pd.DataFrame(rows).to_csv(path / "windows.csv", index=False)
    metadata = dict(
        format_version=1,
        operation="prepare",
        dataset="circling-local-project1",
        channels=CHANNELS,
        sfreq=200,
        units="microvolts",
        model_input_divisor=100,
        window_seconds=4,
        conditions=["open", "closed1", "closed2"],
        group_folds={str(i): i for i in range(3)},
        manifest_sha256={"windows.csv": sha256(path / "windows.csv")},
        array_sha256=hashes,
    )
    (path / "cache.json").write_text(json.dumps(metadata))
    return path


class TinyEncoder(nn.Module):
    """Fast encoder stand-in for end-to-end orchestration tests; native LaBraM is tested separately."""

    def __init__(self, initialization: str, checkpoint: Path, train_backbone: bool) -> None:
        """Create deterministic-sized layers without checkpoint access."""
        super().__init__()
        self.backbone = nn.Linear(64, 8)
        self.backbone.requires_grad_(train_backbone)
        self.projection = nn.Linear(8, 8)

    def features(self, signal: torch.Tensor) -> torch.Tensor:
        """Pool temporal samples to retain participant/window differences."""
        return self.backbone(signal.mean(-1))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply a learned projection to pooled signals."""
        return self.projection(self.features(signal))


def test_independent_reproducible_views() -> None:
    """Sequential view draws differ but rerunning the seed reproduces both exactly."""
    config = SSLConfig()
    signal = torch.ones(8, 64, 800)
    generator = torch.Generator().manual_seed(4)
    a, b = augment(signal, config, generator), augment(signal, config, generator)
    assert not torch.equal(a, b)
    replay = torch.Generator().manual_seed(4)
    torch.testing.assert_close(a, augment(signal, config, replay))
    torch.testing.assert_close(b, augment(signal, config, replay))
    identity = SSLConfig(amplitude=0, noise=0, channel_drop=0)
    torch.testing.assert_close(signal, augment(signal, identity, generator))


def test_expected_tie_mrr() -> None:
    """Collapsed embeddings receive the exact random-tie chance MRR, not inverse mean rank."""
    result = gallery_metrics(torch.ones(8, 4), torch.ones(8, 4))
    assert result["top1"] == 1 / 8
    assert result["mrr"] == pytest.approx(sum(1 / i for i in range(1, 9)) / 8)
    assert result["recall5"] == 5 / 8
    assert result["embedding_std"] == 0
    assert gallery_metrics(torch.eye(8), torch.eye(8))["mrr"] == 1


def test_training_and_locked_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep test signals untouched during fitting, persist fixed galleries, and select by validation only."""
    cache_path = make_cache(tmp_path / "cache")
    config = SSLConfig(initialization="random", pairs=4, epochs=2, steps_per_epoch=2)
    cache = EEGCache(cache_path, config)
    for split, group in (("train", 0), ("validation", 1), ("test", 2)):
        assert {int(f.group.iloc[0]) for f in cache.strata[split]} == {group}
        assert all(f.trial_id.nunique() == 4 for f in cache.galleries(split, config))
    monkeypatch.setattr("hyperscanning_contrastive.ssl.ContrastiveLabram", TinyEncoder)
    original = EEGCache.signals
    seen = set()

    def tracked_signals(self: EEGCache, rows: pd.DataFrame) -> torch.Tensor:
        """Track signal reads independently of manifest/hash verification."""
        seen.update(rows.split.unique())
        return original(self, rows)

    monkeypatch.setattr(EEGCache, "signals", tracked_signals)
    run = tmp_path / "run"
    report = train_ssl(cache_path, run, tmp_path / "nonexistent", config)
    assert seen == {"train", "validation"}
    assert not report["test_evaluated"]
    history = pd.read_csv(run / "history.csv")
    assert report["best_epoch"] == int(history.loc[history.mrr.idxmax(), "epoch"])
    assert json.loads((run / "settings.json").read_text())["checkpoint_sha256"] is None
    from hyperscanning_contrastive.correspondence import check_correspondence

    monkeypatch.setattr("hyperscanning_contrastive.correspondence.ContrastiveLabram", TinyEncoder)
    control = check_correspondence(cache_path, run, tmp_path / "correspondence", permutations=20)
    assert seen == {"train", "validation"}
    assert not control["test_evaluated"]
    initial_csv = pd.read_csv(run / "validation_initial.csv")
    assert control["representations"]["initial"]["actual"]["top1"] == pytest.approx(initial_csv.top1.mean())
    results = evaluate_run(cache_path, run, tmp_path / "test")
    assert "test" in seen
    assert set(results) == {"initial", "best", "initial_pooled", "scope"}
    assert (tmp_path / "test" / "test.json").exists()
    with pytest.raises(FileExistsError):
        train_ssl(cache_path, run, tmp_path / "nonexistent", config)


def test_split_errors_and_checksum(tmp_path: Path) -> None:
    """Reject overlapping held-out folds and modified signal files before fitting."""
    with pytest.raises(ValueError, match="distinct"):
        SSLConfig(validation_fold=2, test_fold=2).validate()
    path = make_cache(tmp_path / "cache")
    with (path / "pair_0.npy").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        EEGCache(path, SSLConfig(pairs=4))


def test_missing_run_reports_resolved_path(tmp_path: Path) -> None:
    """Identify a misdirected run before attempting to load cache or model weights."""
    from hyperscanning_contrastive.ssl import load_evaluation

    run = tmp_path / "wrong-run"
    with pytest.raises(ValueError, match="Missing Stage 2 completion marker") as error:
        load_evaluation(tmp_path / "cache", run, "validation")
    assert str(run / "ssl.json") in str(error.value)
    assert "--run" in str(error.value)


def test_partner_baseline_uses_aligned_members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use matched trial rows for opposite members and never read held-out test signals."""
    from hyperscanning_contrastive.partner import partner_baseline

    path = make_cache(tmp_path / "cache")
    array = np.load(path / "pair_1.npy")
    array[:, 1] = array[:, 0]
    np.save(path / "pair_1.npy", array)
    metadata = json.loads((path / "cache.json").read_text())
    metadata["array_sha256"]["pair_1.npy"] = sha256(path / "pair_1.npy")
    (path / "cache.json").write_text(json.dumps(metadata))
    monkeypatch.setattr("hyperscanning_contrastive.partner.ContrastiveLabram", TinyEncoder)
    original = EEGCache.signals
    reads = []

    def record(self: EEGCache, rows: pd.DataFrame) -> torch.Tensor:
        """Capture exact source rows and participant axis requested by each branch."""
        assert set(rows.split) == {"validation"}
        reads.append((rows.member.iloc[0], rows.array_index.tolist()))
        return original(self, rows)

    monkeypatch.setattr(EEGCache, "signals", record)
    report = partner_baseline(
        path, tmp_path / "partner", tmp_path / "missing", initialization="random", pairs=4, permutations=100
    )
    assert report["checkpoint_sha256"] is None
    assert not report["test_evaluated"]
    for a, b in zip(reads[::2], reads[1::2], strict=True):
        assert a[0] == 0 and b[0] == 1 and a[1] == b[1]
    for condition in report["conditions"].values():
        assert condition["actual"]["top1"] == 1
        assert condition["shuffled_mean"]["top1"] == pytest.approx(0.25, abs=0.08)
    assert (tmp_path / "partner" / "open_minus_closed_by_dyad.csv").exists()
    assert (tmp_path / "partner" / "similarities.npz").exists()


def test_partner_directional_metrics() -> None:
    """Retain asymmetric retrieval directions and assign chance credit for ties."""
    from hyperscanning_contrastive.partner import query_metrics

    rows = query_metrics(torch.tensor([[2.0, 3.0], [0.0, 1.0]]))
    assert rows.top1.tolist() == [0, 1, 1, 0]
    assert rows.mrr.mean() == 0.75
    tied = query_metrics(torch.ones(8, 8))
    assert tied.top1.mean() == 0.125
    assert tied.mrr.mean() == pytest.approx(sum(1 / i for i in range(1, 9)) / 8)


def test_partner_training_controls_and_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit only projection weights with matched controls and select against true validation pairs."""
    from hyperscanning_contrastive.alignment import train_alignment

    cache = make_cache(tmp_path / "cache")
    monkeypatch.setattr("hyperscanning_contrastive.alignment.ContrastiveLabram", TinyEncoder)
    original = EEGCache.signals
    seen = set()

    def track(self: EEGCache, rows: pd.DataFrame) -> torch.Tensor:
        """Reject any test signal extraction during partner training."""
        seen.update(rows.split.unique())
        assert "test" not in seen
        return original(self, rows)

    monkeypatch.setattr(EEGCache, "signals", track)
    output = tmp_path / "alignment"
    report = train_alignment(
        cache, output, tmp_path / "nonexistent", initialization="random", pairs=4, epochs=2, steps_per_epoch=2
    )
    assert seen == {"train", "validation"}
    assert report["checkpoint_sha256"] is None
    assert not report["test_evaluated"]
    mapping = pd.read_csv(output / "shuffled_training_pairs.csv")
    assert set(mapping.group) == {0}
    for _, frame in mapping.groupby(["group", "condition"]):
        assert set(frame.a_trial_id) == set(frame.b_trial_id)
    for arm in ("true", "shuffled"):
        history = pd.read_csv(output / arm / "history.csv")
        assert report["arms"][arm]["best_epoch"] == int(history.loc[history.mrr.idxmax(), "epoch"])
        assert history.iloc[0].mrr == pytest.approx(report["initial_projection"]["macro"]["mrr"])
        assert (output / arm / "best_projection.pt").exists()
    assert (output / "backbone.pt").exists()
