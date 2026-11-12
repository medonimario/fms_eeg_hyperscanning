"""Held-trial training, provenance, statistics and checkpoint selection regression tests."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from cl_behaviour.models import LinearHeads
from cl_behaviour.tiny_fit import digest
from cl_behaviour.training import FeatureBank, epoch_schedule, fit_arm, run_training, summarize


@pytest.fixture
def training_cache(tmp_path):
    """Build two development dyads and reserved metadata without reserved signal files."""
    features, run = tmp_path / "features", tmp_path / "split"
    features.mkdir()
    run.mkdir()
    trials, windows, galleries = [], [], []
    for group in (2010, 2020, 2070):
        for index in range(6):
            split = "external_test" if group == 2070 else "train" if index < 4 else "validation"
            row = dict(
                trial_id=f"{group}:trial:{index}",
                group=group,
                condition="closed1" if index % 2 == 0 else "closed2",
                condition_trial=index // 2,
                split=split,
                eligible=group != 2070,
                main_summary_eligible=group != 2020,
                cohort_fold=2 if group == 2070 else 0,
            )
            trials.append(row)
            for offset in range(0, 15, 2):
                windows.append(dict(row, pair_id=f"{row['trial_id']}:{offset}", crop_start_s=offset))
        if group == 2070:
            continue
        for member in "ab":
            for split in ("train", "validation"):
                rows = [r for r in trials if r["group"] == group and r["split"] == split]
                galleries.append(
                    dict(
                        gallery_id=f"{group}:{member}:{split}",
                        group=group,
                        member=member,
                        split=split,
                        trial_ids=json.dumps([r["trial_id"] for r in rows]),
                        trial_conditions=json.dumps([r["condition"] for r in rows]),
                        candidate_count=len(rows),
                        main_summary_eligible=group != 2020,
                    )
                )
    trials, windows = pd.DataFrame(trials), pd.DataFrame(windows)
    trials.to_csv(run / "splits.csv", index=False)
    windows.to_csv(run / "windows.csv", index=False)
    pd.DataFrame(galleries).to_csv(run / "galleries.csv", index=False)
    protocol = dict(
        format_version=2,
        operation="split",
        profile="combined-closed",
        cache_sha256="preparation",
        manifest_sha256={p.name: digest(p) for p in run.iterdir()},
    )
    (run / "split.json").write_text(json.dumps(protocol))
    trials = trials[trials.eligible].copy()
    windows = windows[windows.eligible].copy()
    trials["movement_feature_array"] = trials.group.astype(str) + "_movement.npy"
    trials["movement_feature_index"] = trials.groupby("group").cumcount()
    windows["token_array"] = windows.group.astype(str) + "_tokens.npy"
    windows["token_array_index"] = windows.groupby("group").cumcount()
    windows["movement_feature_array"] = windows.group.astype(str) + "_movement.npy"
    windows["movement_feature_index"] = windows.trial_id.map(trials.set_index("trial_id").movement_feature_index)
    windows["movement_feature_start"] = windows.crop_start_s * 50
    windows["movement_feature_stop"] = (windows.crop_start_s + 4) * 50
    trials.to_csv(features / "trials.csv", index=False)
    windows.to_csv(features / "windows.csv", index=False)
    rng = np.random.default_rng(42)
    for group in (2010, 2020):
        tokens = rng.normal(size=(48, 2, 64, 4, 200)).astype(np.float32)
        tokens[1::2] = np.nan
        np.save(features / f"{group}_tokens.npy", tokens)
        np.save(features / f"{group}_movement.npy", rng.normal(size=(6, 2, 2, 900)).astype(np.float32))
    np.savez(
        features / "pooling_norm.npz",
        weight=np.ones(200, dtype=np.float32),
        bias=np.zeros(200, dtype=np.float32),
        eps=np.array(1e-6),
    )
    metadata = dict(
        format_version=1,
        operation="features",
        complete=True,
        profile="combined-closed",
        members=["a", "b"],
        token_shape=[2, 64, 4, 200],
        token_order="channel-major, time-minor",
        token_dtype="float32",
        movement_shape=[2, 2, 900],
        movement_components=["sin", "cos"],
        movement_sfreq=50,
        centered=False,
        statistics_fitted=False,
        backbone_frozen=True,
        split_sha256=digest(run / "split.json"),
        cache_sha256="preparation",
        artifact_sha256={p.name: digest(p) for p in features.iterdir()},
    )
    (features / "features.json").write_text(json.dumps(metadata))
    return features, run


def test_galleries_train_only_stats_and_coherent_shuffle(training_cache):
    """Holdouts cannot change fitted means; four offsets share each person's trial shuffle."""
    features, run = training_cache
    bank = FeatureBank(features, run)
    sets, stats = bank.build("pooled", 1042)
    assert len(sets) == 32
    assert all(g["group"] != 2070 for g in sets)
    for group in (2010, 2020):
        for member in "ab":
            train = [g for g in sets if g["group"] == group and g["member"] == member and g["split"] == "train"]
            assert len(train) == 4
            assert all(torch.equal(g["mapping"], train[0]["mapping"]) for g in train)
            assert torch.cat([g["eeg"] for g in train]).mean(0).abs().max() < 1e-6
    metadata = json.loads((features / "features.json").read_text())
    for name in ("2010_tokens.npy", "2010_movement.npy"):
        array = np.load(features / name)
        if "tokens" in name:
            array[32::2] = np.random.default_rng(7).normal(size=array[32::2].shape)
        else:
            array[4:] += 100
        np.save(features / name, array)
        metadata["artifact_sha256"][name] = digest(features / name)
    (features / "features.json").write_text(json.dumps(metadata))
    changed_sets, changed_stats = FeatureBank(features, run).build("pooled", 1042)
    for person in stats:
        for modality in stats[person]:
            for key in ("mean", "scale"):
                torch.testing.assert_close(
                    stats[person][modality][key], changed_stats[person][modality][key], rtol=0, atol=0
                )
    for before, after in zip(sets, changed_sets, strict=True):
        if before["split"] == "train":
            torch.testing.assert_close(before["eeg"], after["eeg"], rtol=0, atol=0)
            torch.testing.assert_close(before["movement"], after["movement"], rtol=0, atol=0)


def test_training_artifacts_and_matching(training_cache, tmp_path):
    """Complete a tiny integration experiment, saving matched arms and PNG-only curves."""
    features, run = training_cache
    result = run_training(features, run, tmp_path / "training", profiles=("pooled",), epochs=1)
    assert result["complete"] and not result["test_features_accessed"]
    root = tmp_path / "training" / "pooled"
    histories = [pd.read_csv(root / arm / "history.csv") for arm in ("true", "shuffled")]
    assert histories[0].iloc[0].validation_mrr == histories[1].iloc[0].validation_mrr
    for arm, history in zip(("true", "shuffled"), histories, strict=True):
        summary = json.loads((root / arm / "selection.json").read_text())
        assert summary["best_epoch"] == int(history.validation_mrr.idxmax())
        assert (root / arm / "learning_curves.png").is_file()
        assert not list((root / arm).glob("*.pdf"))
        assert int(history.iloc[-1].updates) == 4
    for name, checksum in result["output_sha256"].items():
        assert digest(tmp_path / "training" / name) == checksum


def test_epoch_zero_selection_and_equal_dyads(training_cache, tmp_path, monkeypatch):
    """Retain initial weights when validation deteriorates; never use training loss for selection."""
    features, run = training_cache
    sets, _ = FeatureBank(features, run).build("pooled", 1042)
    initial = LinearHeads(200, 400, 32)
    original = summarize
    calls = []

    def artificial_validation(frame):
        """Provide a deterministic decreasing validation metric for the selection audit."""
        result = original(frame)
        result["validation_mrr"] = 0.9 - 0.1 * len(calls)
        calls.append(1)
        return result

    monkeypatch.setattr("cl_behaviour.training.summarize", artificial_validation)
    summary, _ = fit_arm(initial, sets, tmp_path / "arm", "true", epoch_schedule(16, 1, 7), "cpu", 0.1, 0.001, 0.01, 4)
    assert summary["best_epoch"] == 0
    saved = torch.load(tmp_path / "arm/best_heads.pt", weights_only=True)
    for name, value in initial.state_dict().items():
        torch.testing.assert_close(saved[name], value)


def test_tampered_manifest_rejected(training_cache):
    """Refuse altered alignment before model fitting."""
    features, run = training_cache
    with (features / "windows.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        FeatureBank(features, run)


def test_summary_equal_dyads_and_pair2020_exclusion():
    """Unequal record counts cannot give a dyad extra influence or admit pair 2020."""
    rows = []
    metrics = ("top1", "mrr", "chance_top1", "chance_mrr", "assigned_loss", "reverse_top1", "reverse_mrr")
    for split in ("train", "validation"):
        for group, count, value in ((2010, 1, 0.2), (2030, 3, 0.8), (2020, 10, 1.0)):
            rows.extend(
                dict(group=group, split=split, condition="all", **dict.fromkeys(metrics, value)) for _ in range(count)
            )
    assert all(value == pytest.approx(0.5) for value in summarize(pd.DataFrame(rows)).values())
