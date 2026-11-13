"""Continuous movement data contracts and temporal-dependency ablation tests."""

import json

import numpy as np
import pandas as pd
import torch
from test_training import training_cache  # noqa: F401
from test_whole_trial import all_window_cache  # noqa: F401

from cl_behaviour.temporal_trial import PROFILES, TrialModel, build_sequences, run_temporal_trial
from cl_behaviour.tiny_fit import digest
from cl_behaviour.training import FeatureBank


def test_temporal_dependency_and_matched_capacity():
    """Only temporal blocks transmit changes to other positions; shared initial weights match."""
    models = {}
    for profile in PROFILES:
        torch.manual_seed(42)
        models[profile] = TrialModel(profile).eval()
    counts = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
    assert counts == dict(linear=47072, temporal=56480, pointwise=56480)
    reference = models["linear"].state_dict()
    for model in models.values():
        for key, value in reference.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    x = torch.randn(2, 8, 32)
    altered = x.clone()
    altered[:, 3] += torch.randn(2, 32)
    for profile, model in models.items():
        before, after = model.eeg_blocks(x), model.eeg_blocks(altered)
        if profile == "temporal":
            assert not torch.allclose(before[:, 2], after[:, 2])
            assert not torch.allclose(before[:, 7], after[:, 7])
        else:
            torch.testing.assert_close(before[:, 2], after[:, 2], rtol=0, atol=0)
            torch.testing.assert_close(before[:, 7], after[:, 7], rtol=0, atol=0)
    eeg, movement = torch.randn(3, 8, 800), torch.randn(3, 2, 900)
    for model in models.values():
        e, m = model(eeg, movement)
        assert e.shape == m.shape == (3, 32)
        assert not torch.allclose(m, model(eeg, movement.flip(-1))[1])
        (e.square().mean() + m.square().mean()).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_continuous_normalization_and_heldout_isolation(all_window_cache):
    """Use unique raw 900-sample rows; validation values cannot affect fitted statistics."""
    features, run = all_window_cache
    sets, stats = build_sequences(FeatureBank(features, run), 1042)
    raw = torch.from_numpy(np.load(features / "2010_movement.npy")[:, 0].copy())
    mean = raw[:4].mean((0, 2), keepdim=True)
    scale = (raw[:4] - mean).square().mean().sqrt()
    g = next(g for g in sets if g["group"] == 2010 and g["member"] == "a" and g["split"] == "train")
    torch.testing.assert_close(g["movement"], (raw[:4] - mean) / scale)
    assert g["movement"].shape == (4, 2, 900)
    assert "movement" not in stats["2010:a"]
    metadata = json.loads((features / "features.json").read_text())
    for name, start in [("2010_movement.npy", 4), ("2010_tokens.npy", 32)]:
        a = np.load(features / name)
        a[start:] += 50
        np.save(features / name, a)
        metadata["artifact_sha256"][name] = digest(features / name)
    (features / "features.json").write_text(json.dumps(metadata))
    changed, changed_stats = build_sequences(FeatureBank(features, run), 1042)
    for person in stats:
        for modality in stats[person]:
            for key in ("mean", "scale"):
                torch.testing.assert_close(
                    stats[person][modality][key], changed_stats[person][modality][key], rtol=0, atol=0
                )
    for before, after in zip(sets, changed, strict=True):
        assert before["group"] != 2070
        if before["split"] == "train":
            for modality in ("eeg", "movement"):
                torch.testing.assert_close(before[modality], after[modality], rtol=0, atol=0)


def test_temporal_trial_integration(all_window_cache, tmp_path):
    """Run all six fits and verify controls, saved models, plots and validation selection."""
    features, run = all_window_cache
    output = tmp_path / "temporal_trial"
    result = run_temporal_trial(features, run, output, epochs=1)
    assert result["complete"] and not result["test_features_accessed"]
    assert len(result["results"]) == 6
    for profile in PROFILES:
        histories = [pd.read_csv(output / profile / arm / "history.csv") for arm in ("true", "shuffled")]
        assert histories[0].iloc[0].validation_mrr == histories[1].iloc[0].validation_mrr
        for arm, history in zip(("true", "shuffled"), histories, strict=True):
            folder = output / profile / arm
            assert history.updates.tolist() == [0, 1]
            selected = json.loads((folder / "selection.json").read_text())
            assert selected["best_epoch"] == int(history.validation_mrr.idxmax())
            model = TrialModel(profile)
            model.load_state_dict(torch.load(folder / "best_model.pt", weights_only=True))
            assert (folder / "learning_curves.png").is_file()
    for name, checksum in result["output_sha256"].items():
        assert digest(output / name) == checksum
    assert not list(output.rglob("*.pdf"))
