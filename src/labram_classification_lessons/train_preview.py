"""Lesson 4: a small EEGClassifier fine-tuning run with explicit validation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from braindecode import EEGClassifier
from braindecode.datasets import BaseConcatDataset
from sklearn.metrics import balanced_accuracy_score
from skorch.callbacks import Checkpoint, EpochScoring
from skorch.helper import predefined_split
from torch import nn
from torch.utils.data import Dataset, Subset

if __package__:
    from . import inspect_mirrorgame as preprocessing
    from .check_labram import DEFAULT_CHECKPOINT, load_model
    from .mirrorgame_dataset import build_datasets, make_manifest
else:
    import inspect_mirrorgame as preprocessing
    from check_labram import DEFAULT_CHECKPOINT, load_model
    from mirrorgame_dataset import build_datasets, make_manifest


class BinaryLoss(nn.BCEWithLogitsLoss):
    """Accept the integer labels produced by EEGClassifier's batch loader."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute binary cross entropy after matching the target floating dtype.

        Args:
            logits: One raw logit per window.
            targets: Binary labels from the batch loader.

        Returns:
            Scalar binary cross-entropy loss.
        """
        return super().forward(logits, targets.to(dtype=logits.dtype))


def select_preview(dataset: BaseConcatDataset, windows_per_recording: int) -> Subset:
    """Keep a few windows from each participant and condition for a smoke test.

    Args:
        dataset: Windows belonging to one partition.
        windows_per_recording: Maximum windows retained per source file.

    Returns:
        Subset with both participants and conditions represented.
    """
    if windows_per_recording < 1:
        raise ValueError("windows_per_recording must be positive.")
    metadata = dataset.get_metadata().reset_index(drop=True)
    indices = metadata.groupby("path", sort=False).head(windows_per_recording).index.tolist()
    return Subset(dataset, indices)


def make_classifier(
    model: nn.Module, validation: Dataset, output_dir: Path,
    epochs: int = 2, batch_size: int = 4, lr: float = 1e-4, device: str = "cpu",
) -> EEGClassifier:
    """Configure training and restore weights selected only by validation score.

    Args:
        model: Already loaded pretrained model with a new binary head.
        validation: Explicit validation dataset from held-out dyads.
        output_dir: Directory for the best model weights.
        epochs: Number of training passes.
        batch_size: Windows per optimizer step.
        lr: Constant learning rate for this introductory run.
        device: PyTorch device name.

    Returns:
        Classifier ready for fit(training_dataset, y=None).
    """
    scoring = EpochScoring(
        "balanced_accuracy", name="valid_balanced_accuracy", lower_is_better=False, on_train=False,
    )
    checkpoint = Checkpoint(
        monitor="valid_balanced_accuracy_best", dirname=str(output_dir),
        f_params="best_model.pt", f_optimizer=None, f_criterion=None, f_history=None, load_best=True,
    )
    return EEGClassifier(
        module=model, criterion=BinaryLoss, optimizer=torch.optim.AdamW,
        optimizer__weight_decay=0.05, lr=lr, max_epochs=epochs, batch_size=batch_size,
        classes=[0, 1], cropped=False, train_split=predefined_split(validation),
        iterator_train__shuffle=True, iterator_train__drop_last=False,
        iterator_train__num_workers=0, iterator_valid__num_workers=0,
        callbacks=[("balanced_accuracy", scoring), ("checkpoint", checkpoint)], device=device,
    )


def main() -> None:
    """Fine-tune on preview training windows and save validation-selected weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrast", choices=preprocessing.CONTRASTS, default=preprocessing.DEFAULT_CONTRAST)
    parser.add_argument("--fold", type=int, choices=range(10), default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--windows-per-recording", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.windows_per_recording) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("Epochs, batch size, windows per recording, and learning rate must be positive.")
    np.random.seed(42)
    torch.manual_seed(42)
    manifest = make_manifest(preprocessing.get_data_directory(), args.contrast, args.fold)
    candidates = manifest[manifest.split.isin(["train", "valid"])]
    chosen_dyads = candidates.groupby("split").dyad.min().tolist()
    preview_manifest = candidates[candidates.dyad.isin(chosen_dyads)]
    datasets = build_datasets(preview_manifest)
    channels = datasets["train"].datasets[0].windows.ch_names
    for dataset in datasets.values():
        if any(child.windows.ch_names != channels for child in dataset.datasets):
            raise ValueError("All training and validation windows must have the same ordered channels.")
    training = select_preview(datasets["train"], args.windows_per_recording)
    validation = select_preview(datasets["valid"], args.windows_per_recording)
    model = load_model(channels, args.checkpoint)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = args.output_dir or Path(__file__).resolve().parent / "runs" / f"preview-{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    manifest.to_csv(output / "fold_manifest.csv", index=False)
    selected_metadata = {}
    for name, subset in (("train", training), ("valid", validation)):
        metadata = datasets[name].get_metadata().reset_index(drop=True).iloc[subset.indices].copy()
        metadata.to_csv(output / f"{name}_windows.csv", index=False)
        selected_metadata[name] = metadata
        print(f"PREVIEW {name}: dyads={sorted(metadata.dyad.unique())}, windows={len(subset)}")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        preview_only=True, seed=42, labels=preprocessing.CONTRASTS[args.contrast], channels=channels,
        optimizer="AdamW", weight_decay=0.05, scheduler=None, gradient_accumulation=1,
        output_dir=str(output.resolve()),
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    net = make_classifier(model, validation, output, args.epochs, args.batch_size, args.lr, args.device)
    net.fit(training, y=None)
    net.history.to_file(str(output / "history.json"))
    probabilities = net.predict_proba(validation)
    if probabilities.shape != (len(validation), 2) or not np.isfinite(probabilities).all():
        raise ValueError("Expected two finite class probabilities per validation window.")
    predictions = probabilities.argmax(axis=1)
    metadata = selected_metadata["valid"]
    score = balanced_accuracy_score(metadata.target, predictions)
    metadata["probability_class_1"] = probabilities[:, 1]
    metadata["prediction"] = predictions
    metadata.to_csv(output / "validation_predictions.csv", index=False)
    best_epoch = max(net.history, key=lambda row: row["valid_balanced_accuracy"])["epoch"]
    print(f"Restored best epoch: {best_epoch}; preview validation balanced accuracy: {score:.4f}")
    print(f"Saved run: {output.resolve()}")
    print("This small, balanced preview tests the training pipeline; it is not a generalization estimate.")


if __name__ == "__main__":
    main()
