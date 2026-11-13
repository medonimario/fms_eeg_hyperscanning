"""Whole-trial alignment, soft supervision and matched held-out evaluation checks."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from test_training import training_cache  # noqa: F401
from torch.nn import functional

from cl_behaviour.contrastive import symmetric_loss
from cl_behaviour.models import LinearHeads
from cl_behaviour.tiny_fit import digest
from cl_behaviour.training import FeatureBank
from cl_behaviour.whole_trial import (
    build_trials,
    calibrate_teacher,
    objective,
    run_whole_trial,
    teacher_probabilities,
    window_logits,
)


@pytest.fixture
def all_window_cache(training_cache):
    """Populate the odd offsets intentionally absent in the legacy four-window fixture."""
    features, run = training_cache
    metadata = json.loads((features / "features.json").read_text())
    for group in (2010, 2020):
        path = features / f"{group}_tokens.npy"
        tokens = np.load(path)
        tokens[1::2] = np.random.default_rng(group).normal(size=tokens[1::2].shape)
        np.save(path, tokens)
        metadata["artifact_sha256"][path.name] = digest(path)
    (features / "features.json").write_text(json.dumps(metadata))
    return features, run


def test_objectives_and_shuffle():
    """Recover hard CL exactly and verify both soft directions under a non-involutive permutation."""
    torch.manual_seed(1)
    logits = torch.randn(8, 4, 4, requires_grad=True)
    movement = functional.normalize(torch.randn(8, 4, 5), dim=-1)
    sim = movement @ movement.transpose(-2, -1)
    q = teacher_probabilities(sim, 0.7)
    mapping = torch.tensor([2, 0, 3, 1])
    gallery = dict(mapping=mapping, local_teacher=q, trial_teacher=q[0])
    for arm, permutation in (("true", None), ("shuffled", mapping)):
        local = objective(logits, gallery, "local-hard", arm, 0.5)
        expected = torch.stack([symmetric_loss(x, permutation) for x in logits]).mean()
        torch.testing.assert_close(local, expected)
        torch.testing.assert_close(
            objective(logits, gallery, "trial-hard", arm, 0.5), symmetric_loss(logits.mean(0), permutation)
        )
        torch.testing.assert_close(objective(logits, gallery, "local-soft", arm, 0.0), local)
    aligned = logits[:, :, mapping]
    target = 0.5 * torch.eye(4) + 0.5 * q[:, mapping][:, :, mapping]
    assert not torch.allclose(target, target.transpose(-2, -1))
    torch.testing.assert_close(target.sum(-1), torch.ones(8, 4))
    expected = -0.5 * (
        (target * aligned.log_softmax(-1)).sum(-1).mean()
        + (target * aligned.transpose(-2, -1).log_softmax(-1)).sum(-1).mean()
    )
    loss = objective(logits, gallery, "local-soft", "shuffled", 0.5)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert not torch.isclose(
        objective(logits, gallery, "local-hard", "true", 0.5), objective(logits, gallery, "trial-hard", "true", 0.5)
    )


def test_aggregation_is_ordered_concatenation():
    """Trial scores equal ordered normalized concatenation, with no extra parameters."""
    torch.manual_seed(5)
    model = LinearHeads(3, 4, 5)
    g = dict(eeg=torch.randn(4, 8, 3), movement=torch.randn(4, 8, 4))
    logits = window_logits(model, g, "cpu", 0.1)
    eeg = functional.normalize(functional.normalize(model.eeg(g["eeg"]), dim=-1).flatten(1), dim=-1)
    movement = functional.normalize(functional.normalize(model.movement(g["movement"]), dim=-1).flatten(1), dim=-1)
    torch.testing.assert_close(logits.mean(0), eeg @ movement.T / 0.1)
    g["movement"] = g["movement"].flip(1)
    assert not torch.allclose(logits.mean(0), window_logits(model, g, "cpu", 0.1).mean(0))


def test_teacher_and_normalization_ignore_validation(all_window_cache):
    """Changing held-out trajectories cannot change fitted statistics or teacher calibration."""
    features, run = all_window_cache
    sets, stats = build_trials(FeatureBank(features, run), 1042)
    temperatures, diagnostic = calibrate_teacher(sets)
    assert diagnostic.gallery_id.str.endswith("train").all()
    assert sets[0]["eeg"].shape == (4, 8, 800)
    assert stats["2010:a"]["trajectory_teacher"]["mean"].shape == (1, 1800)
    raw = np.load(features / "2010_movement.npy")
    expected = torch.from_numpy(raw[:4, 0].reshape(4, -1)).mean(0, keepdim=True)
    torch.testing.assert_close(stats["2010:a"]["trajectory_teacher"]["mean"], expected)
    metadata = json.loads((features / "features.json").read_text())
    raw[4:] += 100
    np.save(features / "2010_movement.npy", raw)
    metadata["artifact_sha256"]["2010_movement.npy"] = digest(features / "2010_movement.npy")
    (features / "features.json").write_text(json.dumps(metadata))
    changed, changed_stats = build_trials(FeatureBank(features, run), 1042)
    assert calibrate_teacher(changed)[0] == temperatures
    for person in stats:
        for modality in stats[person]:
            for key in ("mean", "scale"):
                torch.testing.assert_close(stats[person][modality][key], changed_stats[person][modality][key])
    for g in sets:
        assert g["group"] != 2070


def test_whole_trial_integration(all_window_cache, tmp_path):
    """All eight fits share initialization/budgets; outputs select trial MRR and hash correctly."""
    features, run = all_window_cache
    output = tmp_path / "whole_trial"
    result = run_whole_trial(features, run, output, epochs=1)
    assert result["complete"] and not result["test_features_accessed"]
    assert len(result["results"]) == 8
    initial_scores = []
    for row in result["results"]:
        folder = output / row["experiment"] / row["arm"]
        history = pd.read_csv(folder / "history.csv")
        assert history.updates.tolist() == [0, 1]
        assert row["best_epoch"] == int(history.validation_mrr.idxmax())
        initial_scores.append(history.iloc[0].validation_mrr)
        detail = pd.read_csv(folder / "best_gallery_metrics.csv")
        assert set(detail.level) == {"trial", "window"}
        assert set(detail[detail.level == "window"].offset) == set(range(0, 15, 2))
        restricted = detail[detail.scope == "within-condition"]
        assert (restricted.candidates == restricted.queries).all()
        assert (folder / "learning_curves.png").is_file()
    assert len(set(initial_scores)) == 1
    assert not list(output.rglob("*.pdf"))
    for name, checksum in result["output_sha256"].items():
        assert digest(output / name) == checksum
