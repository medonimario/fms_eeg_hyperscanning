"""Check diagnostic evaluation, checkpoint independence, and fold curve exports."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from labram_classification.data import sha256
from labram_classification.monitoring import CURVE_KEYS, evaluate_partition
from labram_classification.plotting import collect_curves, plot_curves
from labram_classification.training import make_classifier


def options() -> dict:
    """Use a short CPU run with stochastic layers and accumulation."""
    return dict(
        num_workers=0,
        device="cpu",
        epochs=2,
        batch_size=2,
        lr=0.01,
        update_freq=2,
        layer_decay=1,
        weight_decay=0.05,
        warmup_epochs=0,
        min_lr=1e-6,
        amp=False,
    )


def test_evaluation_preserves_state_and_includes_tail(tmp_path: Path) -> None:
    """Evaluation restores RNG/modes/weights and correctly weights a final partial batch."""
    model = nn.Sequential(nn.Linear(1, 1), nn.Dropout(0.5), nn.Flatten(0))
    with torch.no_grad():
        model[0].weight.fill_(1)
        model[0].bias.zero_()
    data = TensorDataset(torch.tensor([[-2.0], [-1.0], [1.0], [2.0], [3.0]]), torch.tensor([0, 0, 1, 1, 0]))
    net = make_classifier(model, data, tmp_path, options()).initialize()
    model.train()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = evaluate_partition(net, data)
    expected_loss = nn.BCEWithLogitsLoss()(data.tensors[0].flatten(), data.tensors[1].float()).item()
    assert result["loss"] == pytest.approx(expected_loss)
    assert result["acc"] == 0.8
    assert result["balanced_accuracy"] == pytest.approx((2 / 3 + 1) / 2)
    assert all(module.training for module in model.modules())
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])


def test_test_labels_do_not_change_training_or_selection(tmp_path: Path) -> None:
    """Changing diagnostic test labels changes curves but never learned weights or selection."""
    data = TensorDataset(torch.tensor([[-2.0], [-1.0], [1.0], [2.0]]), torch.tensor([0, 0, 1, 1]))
    runs = []
    for flip in (False, True):
        torch.manual_seed(12)
        folder = tmp_path / str(flip)
        folder.mkdir()
        test = TensorDataset(data.tensors[0], 1 - data.tensors[1] if flip else data.tensors[1])
        model = nn.Sequential(nn.Linear(1, 1), nn.Dropout(0.2), nn.Flatten(0))
        net = make_classifier(model, data, folder, options(), test=test)
        net.set_params(verbose=0)
        net.fit(data, y=None)
        runs.append(net)
        history = json.loads((folder / "history.json").read_text())
        assert len(history) == 2
        assert set(CURVE_KEYS) <= history[-1].keys()
        saved = torch.load(folder / "best_model.pt", weights_only=True)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    for name, value in runs[0].module_.state_dict().items():
        torch.testing.assert_close(value, runs[1].module_.state_dict()[name], rtol=0, atol=0)
    assert runs[0].history[:, "valid_balanced_accuracy_best"] == runs[1].history[:, "valid_balanced_accuracy_best"]
    assert runs[0].history[:, "test_loss"] != runs[1].history[:, "test_loss"]


def make_experiment(directory: Path) -> None:
    """Create ten completed fold histories with analytically known mean and SD."""
    for fold in range(10):
        folder = directory / f"fold_{fold}"
        folder.mkdir(parents=True)
        config = dict(fold=fold, options=dict(epochs=2), steps_per_epoch=1)
        (folder / "config.json").write_text(json.dumps(config))
        result = dict(fold=fold, smoke=False, config_sha256=sha256(folder / "config.json"), test=dict(accuracy=0.5))
        (folder / "result.json").write_text(json.dumps(result))
        history = [{"epoch": epoch, **{key: 0.1 + fold * 0.01 for key in CURVE_KEYS}} for epoch in (1, 2)]
        (folder / "history.json").write_text(json.dumps(history))


def test_plot_means_sd_and_artifacts(tmp_path: Path) -> None:
    """Verify sample SD, three split lines' source values, and usable figure files."""
    make_experiment(tmp_path)
    values, summary = collect_curves(tmp_path)
    assert len(values) == 10 * 2 * 3 * 3
    np.testing.assert_allclose(summary["mean"], 0.145)
    np.testing.assert_allclose(summary.sd, np.std(0.1 + np.arange(10) * 0.01, ddof=1))
    assert (summary.n_folds == 10).all()
    output = plot_curves(tmp_path)
    for extension in ("png", "pdf", "svg"):
        assert (output / f"learning_curves.{extension}").stat().st_size > 1000
    assert len(pd.read_csv(output / "learning_curves_folds.csv")) == len(values)


def test_plot_rejects_missing_metrics(tmp_path: Path) -> None:
    """Old histories must not silently acquire fabricated training/test curves."""
    make_experiment(tmp_path)
    path = tmp_path / "fold_0" / "history.json"
    history = json.loads(path.read_text())
    del history[0]["test_loss"]
    path.write_text(json.dumps(history))
    with pytest.raises(ValueError, match="Older runs"):
        collect_curves(tmp_path)
