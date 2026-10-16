"""Exercise native EEGNet training, model profiles, and shared experiment exports."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from labram_classification.__main__ import parser
from labram_classification.data import CHANNELS, fold_datasets, sha256
from labram_classification.model import BinaryEEGNet
from labram_classification.monitoring import CURVE_KEYS, evaluate_partition
from labram_classification.plotting import plot_curves
from labram_classification.runner import run_fold, summarize
from labram_classification.training import BinaryLoss, make_classifier, parameter_groups, resolve_model_options


def options() -> dict:
    """Resolve actual CLI defaults with a small two-epoch CPU override."""
    args = parser().parse_args(
        [
            "cv",
            "--model",
            "eegnet",
            "--cache",
            "unused",
            "--output",
            "unused",
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--wandb-mode",
            "disabled",
            "--no-amp",
        ]
    )
    return resolve_model_options(
        {key: value for key, value in vars(args).items() if key not in {"cache", "output", "checkpoint", "command"}}
    )


def make_cache(path: Path) -> None:
    """Create distinct labelled windows in ten disjoint triads."""
    path.mkdir()
    data = np.random.RandomState(7).normal(size=(20, 64, 800)).astype(np.float32) * 20
    np.save(path / "signals.npy", data)
    rows = [
        dict(
            group=fold,
            participant=f"{fold}A",
            fold=fold,
            target=target,
            array="signals.npy",
            array_index=fold * 2 + target,
            source="synthetic",
        )
        for fold in range(10)
        for target in (0, 1)
    ]
    pd.DataFrame(rows).to_csv(path / "windows.csv", index=False)
    (path / "cache.json").write_text(
        json.dumps(
            dict(
                format_version=1,
                channels=CHANNELS,
                dataset="fg",
                contrast="interaction",
                metadata_sha256=sha256(path / "windows.csv"),
            )
        )
    )


def test_model_profiles_preserve_labram_defaults() -> None:
    """Select scratch defaults explicitly without changing existing LaBraM commands."""
    for model, expected in [("labram", (10, 128, 2, 0.65)), ("eegnet", (100, 64, 1, 1))]:
        args = parser().parse_args(["cv", "--model", model, "--cache", "cache", "--output", "out"])
        resolved = resolve_model_options(vars(args))
        assert tuple(resolved[key] for key in ("epochs", "batch_size", "update_freq", "layer_decay")) == expected
    assert resolve_model_options(dict(model="eegnet", epochs=17, lr=0.002))["lr"] == 0.002
    with pytest.raises(ValueError, match="layer-decay"):
        resolve_model_options(dict(model="eegnet", layer_decay=0.65))


def test_eegnet_gradients_and_readonly_evaluation(tmp_path: Path) -> None:
    """Keep batch-one logits, all gradients, and BN/max-norm state safe during diagnostics."""
    torch.set_num_threads(2)
    model = BinaryEEGNet()
    assert model(torch.randn(1, 64, 800)).shape == (1,)
    BinaryLoss()(model(torch.randn(2, 64, 800)), torch.tensor([0, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(group["lr_scale"] == 1 for group in parameter_groups(model, 1e-3, 1, 0))
    make_cache(tmp_path / "cache")
    _, partitions = fold_datasets(tmp_path / "cache", 0)
    net = make_classifier(model, partitions["valid"], tmp_path, options()).initialize()
    model.train()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    evaluate_partition(net, partitions["test"])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(module.training for module in model.modules())


def test_eegnet_cv_without_checkpoint_and_plot(tmp_path: Path) -> None:
    """Train native EEGNet on all folds, reload the best state, and generate mean/SD figures."""
    torch.set_num_threads(2)
    cache = tmp_path / "cache"
    make_cache(cache)
    experiment = tmp_path / "experiment"
    for fold in range(10):
        output = experiment / f"fold_{fold}"
        result = run_fold(cache, output, tmp_path / "absent.pth", fold, options())
        assert result["best_epoch"] in (1, 2)
        config = json.loads((output / "config.json").read_text())
        assert config["model"] == "eegnet"
        assert config["initialization"] == "random"
        assert config["checkpoint_sha256"] is None
        history = json.loads((output / "history.json").read_text())
        assert set(CURVE_KEYS) <= history[-1].keys()
        model = BinaryEEGNet().eval()
        model.load_state_dict(torch.load(output / "best_model.pt", weights_only=True))
        _, partitions = fold_datasets(cache, fold)
        x = torch.from_numpy(np.stack([partitions["test"][i][0] for i in range(2)]))
        with torch.no_grad():
            expected = model(x).sigmoid().numpy()
        actual = pd.read_csv(output / "test_predictions.csv").probability_class_1.to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5)
    assert "balanced_accuracy" in summarize(experiment)
    figures = plot_curves(experiment)
    assert (figures / "learning_curves.pdf").is_file()


def test_eegnet_test_labels_do_not_affect_selection(tmp_path: Path) -> None:
    """Changing diagnostic test labels leaves scratch training and best-epoch selection identical."""
    torch.set_num_threads(2)
    signal = torch.from_numpy(np.random.RandomState(11).normal(size=(4, 64, 800)).astype(np.float32))
    targets = torch.tensor([0, 1, 0, 1])
    data = TensorDataset(signal, targets)
    runs = []
    for flip in (False, True):
        torch.manual_seed(17)
        output = tmp_path / str(flip)
        output.mkdir()
        test = TensorDataset(signal, 1 - targets if flip else targets)
        net = make_classifier(BinaryEEGNet(), data, output, options() | {"batch_size": 2}, test=test)
        net.set_params(verbose=0)
        net.fit(data, y=None)
        runs.append(net)
    for name, value in runs[0].module_.state_dict().items():
        torch.testing.assert_close(value, runs[1].module_.state_dict()[name], rtol=0, atol=0)
    assert runs[0].history[:, "valid_balanced_accuracy_best"] == runs[1].history[:, "valid_balanced_accuracy_best"]
    assert runs[0].history[:, "test_loss"] != runs[1].history[:, "test_loss"]
