"""Verify W&B lifecycle and progress logging without contacting the service."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from labram_classification.tracking import WandbProgress, log_result, tracked_run
from labram_classification.training import make_classifier


def configuration(mode: str) -> dict:
    """Return a minimal fold tracking configuration."""
    return {"fold": 3, "options": {"wandb_mode": mode, "wandb_group": "experiment", "smoke_windows_per_class": 0}}


def test_disabled_never_imports_wandb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Disabled training works even when the optional SDK cannot be imported."""
    monkeypatch.setitem(sys.modules, "wandb", None)
    with tracked_run(configuration("disabled"), tmp_path) as run:
        assert run is None


@pytest.mark.parametrize("fail", [False, True])
def test_fold_lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail: bool) -> None:
    """Finish each fold once, marking failures without hiding their exceptions."""
    run = Mock()
    sdk = SimpleNamespace(init=Mock(return_value=run))
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    try:
        with tracked_run(configuration("offline"), tmp_path):
            if fail:
                raise ValueError("training failure")
    except ValueError as error:
        assert fail and str(error) == "training failure"
    run.finish.assert_called_once_with(exit_code=int(fail))
    assert sdk.init.call_args.kwargs["name"] == "experiment-fold-3"
    assert sdk.init.call_args.kwargs["mode"] == "offline"
    assert "api_key" not in sdk.init.call_args.kwargs


def test_progress_with_classifier(tmp_path: Path) -> None:
    """Exercise callback ordering and optimizer-update accounting in real training."""
    run = Mock()
    data = TensorDataset(torch.tensor([[-2.0], [-1.0], [1.0], [2.0]]), torch.tensor([0, 0, 1, 1]))
    model = nn.Sequential(nn.Linear(1, 1), nn.Flatten(start_dim=0))
    options = dict(
        num_workers=0,
        device="cpu",
        epochs=2,
        batch_size=1,
        lr=0.001,
        update_freq=2,
        layer_decay=1,
        weight_decay=0.05,
        warmup_epochs=0,
        min_lr=1e-6,
        amp=False,
    )
    net = make_classifier(model, data, tmp_path, options, test=data)
    net.set_params(verbose=0, callbacks=net.callbacks + [("wandb", WandbProgress(run, log_every=1))])
    net.fit(data, y=None)
    logs = [call.args[0] for call in run.log.call_args_list]
    updates = [row for row in logs if "optimizer_step" in row]
    epochs = [row for row in logs if "epoch" in row]
    assert [row["optimizer_step"] for row in updates] == [1, 2, 3, 4]
    assert [row["epoch"] for row in epochs] == [1, 2]
    assert epochs[-1]["epoch/valid_balanced_accuracy"] == net.history[-1]["valid_balanced_accuracy"]
    assert all("epoch/train_eval_acc" in row and "epoch/test_balanced_accuracy" in row for row in epochs)
    assert (tmp_path / "epoch_metrics.csv").exists()
    result = {
        "best_epoch": 1,
        "valid": {"accuracy": 0.5, "confusion_matrix": [[1, 0], [1, 0]]},
        "test": {"accuracy": 0.75, "confusion_matrix": [[2, 0], [1, 1]]},
    }
    run.summary = {}
    log_result(run, result)
    assert run.summary["final/test/accuracy"] == 0.75


def test_real_offline_sdk(tmp_path: Path) -> None:
    """Create and finish a local SDK run with no login or network requirement."""
    pytest.importorskip("wandb")
    with tracked_run(configuration("offline"), tmp_path) as run:
        run.log({"epoch": 1, "epoch/train_loss": 0.7})
    assert list((tmp_path / "wandb").glob("offline-run-*/*.wandb"))
