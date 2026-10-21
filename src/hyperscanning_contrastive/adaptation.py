"""Limited LaBraM adaptation with matched true and shuffled partner training."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .alignment import aggregate_queries, evaluate_projection, training_permutations
from .alignment_diagnostics import record_diagnostics
from .contrastive import contrastive_loss
from .data import sha256
from .overfit import ContrastiveLabram
from .partner import METRICS
from .ssl import EEGCache, SSLConfig
from .trial_split import split_within_dyad


def configure_adaptation(model: ContrastiveLabram, blocks: int) -> list[str]:
    """Enable the final transformer blocks, pooling normalization, and shared projection."""
    if not 1 <= blocks <= len(model.backbone.blocks):
        raise ValueError("Unfreeze count must be between one and the number of transformer blocks.")
    model.backbone.requires_grad_(False)
    for block in model.backbone.blocks[-blocks:]:
        block.requires_grad_(True)
    model.backbone.fc_norm.requires_grad_(True)
    model.projection.requires_grad_(True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def diagnostic_bank(model: nn.Module, cache: EEGCache, frames: list[pd.DataFrame]) -> dict:
    """Recompute current features for fixed diagnostic dyads without retaining gradients."""
    bank = {}
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for frame in frames:
            entry = {"rows": frame, "split": frame.split.iloc[0]}
            for member, label in enumerate(("a", "b")):
                chunks = []
                for start in range(0, len(frame), 8):
                    signals = cache.signals(frame.iloc[start : start + 8].assign(member=member))
                    chunks.append(model.features(signals.to(device)).cpu())
                entry[label] = torch.cat(chunks)
            bank[(int(frame.group.iloc[0]), frame.condition.iloc[0])] = entry
    return bank


def train_adaptation(
    cache_path: Path,
    output: Path,
    checkpoint: Path,
    initialization: str = "pretrained",
    device: str = "cpu",
    pairs: int = 8,
    epochs: int = 20,
    steps_per_epoch: int = 100,
    seed: int = 42,
    permutation_seed: int = 1042,
    unfreeze_blocks: int = 2,
    backbone_lr: float = 1e-5,
    validation_protocol: str = "unseen-dyad",
    validation_trials: int = 8,
    split_seed: int = 2042,
) -> dict:
    """Train limited encoder adaptation on real EEG with validation-only model selection.

    The original shared MLP, galleries, batch schedule, and fixed shuffled trial mappings
    match the frozen projection experiment. Four training dyads supply bounded diagnostics.
    Evaluation mode disables stochastic layers but leaves gradients enabled during fitting.
    """
    if not np.isfinite(backbone_lr) or backbone_lr <= 0:
        raise ValueError("Backbone learning rate must be finite and positive.")
    if validation_protocol not in {"unseen-dyad", "within-dyad"}:
        raise ValueError("Unknown validation protocol.")
    config = SSLConfig(
        initialization=initialization, pairs=pairs, epochs=epochs, steps_per_epoch=steps_per_epoch, seed=seed
    )
    cache = EEGCache(cache_path, config)
    if validation_protocol == "within-dyad":
        split_within_dyad(cache, validation_trials, pairs, split_seed)
    torch.manual_seed(seed)
    model = ContrastiveLabram(initialization, checkpoint, False).to(device).eval()
    trainable = configure_adaptation(model, unfreeze_blocks)
    output.mkdir(parents=True, exist_ok=False)
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save(initial_state, output / "initial.pt")
    strata = [frame for frame in cache.strata["train"] if int(frame.member.iloc[0]) == 0]
    training = {(int(frame.group.iloc[0]), frame.condition.iloc[0]): frame for frame in strata}
    mapping_bank = {key: {"split": "train", "a": frame} for key, frame in training.items()}
    mappings = training_permutations(mapping_bank, permutation_seed)
    mapping_rows = []
    for key, permutation in mappings.items():
        trials = training[key].trial_id.tolist()
        mapping_rows.extend(
            {"group": key[0], "condition": key[1], "a_trial_id": trial, "b_trial_id": trials[permutation[i]]}
            for i, trial in enumerate(trials)
        )
    pd.DataFrame(mapping_rows).to_csv(output / "shuffled_training_pairs.csv", index=False)
    galleries = [frame for frame in cache.galleries("validation", config) if int(frame.member.iloc[0]) == 0]
    diagnostic_groups = sorted({key[0] for key in training})[:4]
    training_galleries = [
        frame
        for frame in cache.galleries("train", config)
        if int(frame.member.iloc[0]) == 0 and int(frame.group.iloc[0]) in diagnostic_groups
    ]
    diagnostic_frames = [frame for key, frame in training.items() if key[0] in diagnostic_groups]
    pd.concat(galleries).to_csv(output / "galleries.csv", index=False)
    pd.concat(training_galleries).to_csv(output / "training_galleries.csv", index=False)
    cache.rows[["group", "fold", "split"]].drop_duplicates().to_csv(output / "splits.csv", index=False)
    cache.rows[["group", "condition", "trial_id", "condition_trial", "fold", "split"]].to_csv(
        output / "trial_splits.csv", index=False
    )
    bank = diagnostic_bank(model, cache, galleries)
    head = copy.deepcopy(model.projection).cpu()
    pooled = evaluate_projection(nn.Identity(), bank, galleries)
    initial = evaluate_projection(head, bank, galleries)
    pooled.to_csv(output / "validation_pooled.csv", index=False)
    initial.to_csv(output / "validation_initial.csv", index=False)
    report = {
        "scope": "Exploratory validation-only limited-backbone shared partner adaptation",
        "validation_protocol": validation_protocol,
        "validation_trials_per_condition": validation_trials if validation_protocol == "within-dyad" else None,
        "split_seed": split_seed if validation_protocol == "within-dyad" else None,
        "split_unit": "paired whole trial",
        "crop_offset_seconds": 0,
        "trial_splits_sha256": sha256(output / "trial_splits.csv"),
        "split_counts": {
            split: {"dyads": int(frame.group.nunique()), "paired_windows": len(frame)}
            for split, frame in cache.rows.groupby("split")
        },
        "external_validation_evaluated": False,
        "initialization": initialization,
        "projection_profile": "legacy-mlp",
        "unfreeze_blocks": unfreeze_blocks,
        "pooling_norm_trainable": True,
        "trainable_parameters": trainable,
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "seed": seed,
        "permutation_seed": permutation_seed,
        "checkpoint_sha256": sha256(checkpoint) if initialization == "pretrained" else None,
        "cache_sha256": sha256(cache_path / "cache.json"),
        "initial_sha256": sha256(output / "initial.pt"),
        "gallery_sha256": sha256(output / "galleries.csv"),
        "shuffled_training_pairs_sha256": sha256(output / "shuffled_training_pairs.csv"),
        "training_gallery_sha256": sha256(output / "training_galleries.csv"),
        "training_diagnostic_groups": diagnostic_groups,
        "validation_fold": 1 if validation_protocol == "unseen-dyad" else None,
        "reserved_external_validation_fold": 1 if validation_protocol == "within-dyad" else None,
        "test_fold": 2,
        "test_evaluated": False,
        "pairs": pairs,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "backbone_lr": backbone_lr,
        "projection_lr": 1e-3,
        "weight_decay": 0.01,
        "temperature": 0.1,
        "gradient_clip": 1.0,
        "schedule": "constant; no warm-up",
        "model_mode": "eval with gradients; no augmentation",
        "device": device,
        "torch_version": torch.__version__,
        "pooled": aggregate_queries(pooled),
        "initial_projection": aggregate_queries(initial),
        "chance_top1": 1 / pairs,
        "chance_mrr": sum(1 / i for i in range(1, pairs + 1)) / pairs,
        "arms": {},
    }
    (output / "settings.json").write_text(json.dumps(report, indent=2) + "\n")
    keys = list(training)
    for arm in ("true", "shuffled"):
        folder = output / arm
        folder.mkdir()
        model.load_state_dict(initial_state)
        model.eval()
        encoder_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [{"params": encoder_parameters, "lr": backbone_lr}, {"params": model.projection.parameters(), "lr": 1e-3}],
            weight_decay=0.01,
        )
        rng = np.random.default_rng(seed + 4000)
        best, best_epoch, best_rows = report["initial_projection"]["macro"]["mrr"], 0, initial.copy()
        torch.save(initial_state, folder / "best.pt")
        history = []
        for epoch in range(epochs + 1):
            losses, gradient_norms = [], []
            if epoch:
                model.eval()
                for _ in range(steps_per_epoch):
                    key = keys[int(rng.integers(len(keys)))]
                    frame = training[key]
                    indices = rng.choice(len(frame), pairs, replace=False)
                    partner_indices = mappings[key][indices] if arm == "shuffled" else indices
                    a = model(cache.signals(frame.iloc[indices].assign(member=0)).to(device))
                    b = model(cache.signals(frame.iloc[partner_indices].assign(member=1)).to(device))
                    loss = contrastive_loss(a, b, 0.1)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite adaptation loss.")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(loss.item())
                    gradient_norms.append(float(gradient_norm))
            bank = diagnostic_bank(model, cache, diagnostic_frames)
            validation_bank = diagnostic_bank(model, cache, galleries)
            head = copy.deepcopy(model.projection).cpu()
            validation = evaluate_projection(head, validation_bank, galleries)
            metrics = aggregate_queries(validation)["macro"]
            diagnostics = record_diagnostics(
                head,
                bank,
                training_galleries,
                galleries,
                mappings if arm == "shuffled" else None,
                epoch,
                folder,
                validation_bank=validation_bank,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)) if losses else None,
                    **metrics,
                    **diagnostics,
                    "gradient_norm_mean": float(np.mean(gradient_norms)) if gradient_norms else None,
                }
            )
            pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
            validation.to_csv(folder / f"validation_epoch_{epoch:03d}.csv", index=False)
            if metrics["mrr"] > best:
                best, best_epoch, best_rows = metrics["mrr"], epoch, validation
                torch.save(model.state_dict(), folder / "best.pt")
            print(f"{arm} epoch {epoch}: validation MRR={metrics['mrr']:.4f}", flush=True)
        torch.save(model.state_dict(), folder / "last.pt")
        best_rows.to_csv(folder / "validation_best.csv", index=False)
        best_rows.groupby(["group", "condition"])[METRICS].mean().to_csv(folder / "dyads.csv")
        report["arms"][arm] = {
            "best_epoch": best_epoch,
            **aggregate_queries(best_rows),
            "model_sha256": sha256(folder / "best.pt"),
        }
    report["settings_sha256"] = sha256(output / "settings.json")
    (output / "adaptation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
