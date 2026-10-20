"""Train shared partner projections on frozen features with a matched shuffled-pair control."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .contrastive import contrastive_loss, pair_scores
from .data import sha256
from .overfit import ContrastiveLabram
from .partner import METRICS, query_metrics
from .ssl import EEGCache, SSLConfig


def feature_bank(model: nn.Module, cache: EEGCache, batch_size: int = 8) -> dict:
    """Extract unaugmented frozen features from training/validation only, in bounded batches."""
    bank = {}
    model.eval()
    with torch.no_grad():
        for split in ("train", "validation"):
            for frame in cache.strata[split]:
                if int(frame.member.iloc[0]) != 0:
                    continue
                members = []
                for member in (0, 1):
                    chunks = []
                    for start in range(0, len(frame), batch_size):
                        signals = cache.signals(frame.iloc[start : start + batch_size].assign(member=member))
                        chunks.append(model.features(signals.to(next(model.parameters()).device)).cpu())
                    members.append(torch.cat(chunks))
                key = (int(frame.group.iloc[0]), frame.condition.iloc[0])
                bank[key] = {"split": split, "rows": frame.copy(), "a": members[0], "b": members[1]}
                print(f"Extracted {split} dyad {key[0]}, {key[1]}", flush=True)
    return bank


def evaluate_projection(head: nn.Module, bank: dict, galleries: list[pd.DataFrame]) -> pd.DataFrame:
    """Evaluate true partner matches on the fixed validation galleries in both directions."""
    head.eval()
    rows = []
    with torch.no_grad():
        for frame in galleries:
            key = (int(frame.group.iloc[0]), frame.condition.iloc[0])
            entry = bank[key]
            lookup = {trial: index for index, trial in enumerate(entry["rows"].trial_id)}
            indices = [lookup[trial] for trial in frame.trial_id]
            a, b = head(entry["a"][indices]), head(entry["b"][indices])
            metrics = query_metrics(pair_scores(a, b))
            metrics["trial_id"] = [frame.trial_id.iloc[i] for i in metrics.query_index]
            rows.append(metrics.assign(group=key[0], condition=key[1]))
    return pd.concat(rows, ignore_index=True)


def aggregate_queries(rows: pd.DataFrame) -> dict:
    """Average queries, directions, conditions, and dyads without optimizing condition differences."""
    dyads = rows.groupby(["group", "condition", "direction"])[METRICS].mean().groupby(["group", "condition"]).mean()
    return {
        "macro": dyads.groupby("group").mean().mean().to_dict(),
        "conditions": {
            condition: frame.droplevel("condition").mean().to_dict()
            for condition, frame in dyads.groupby(level="condition")
        },
    }


def training_permutations(bank: dict, seed: int) -> dict:
    """Fix a uniform trial bijection within each training dyad/condition, allowing fixed points."""
    rng = np.random.default_rng(seed)
    return {key: rng.permutation(len(entry["a"])) for key, entry in bank.items() if entry["split"] == "train"}


def train_alignment(
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
) -> dict:
    """Fit true-pair and shuffled-pair heads from identical weights and matched batch schedules.

    Backbone extraction may use a GPU; the small projection heads train on CPU using detached features.
    Validation always uses real correspondence. Test signals are not extracted or evaluated.
    """
    config = SSLConfig(
        initialization=initialization, pairs=pairs, epochs=epochs, steps_per_epoch=steps_per_epoch, seed=seed
    )
    cache = EEGCache(cache_path, config)
    output.mkdir(parents=True, exist_ok=False)
    galleries = [frame for frame in cache.galleries("validation", config) if int(frame.member.iloc[0]) == 0]
    pd.concat(galleries).to_csv(output / "galleries.csv", index=False)
    cache.rows[["group", "fold", "split"]].drop_duplicates().to_csv(output / "splits.csv", index=False)
    torch.manual_seed(seed)
    model = ContrastiveLabram(initialization, checkpoint, False).to(device).eval()
    bank = feature_bank(model, cache)
    initial_head = copy.deepcopy(model.projection).cpu()
    torch.save(initial_head.state_dict(), output / "initial_projection.pt")
    torch.save(model.backbone.state_dict(), output / "backbone.pt")
    del model
    pooled = evaluate_projection(nn.Identity(), bank, galleries)
    initial = evaluate_projection(initial_head, bank, galleries)
    pooled.to_csv(output / "validation_pooled.csv", index=False)
    initial.to_csv(output / "validation_initial.csv", index=False)
    mappings = training_permutations(bank, permutation_seed)
    mapping_rows = []
    for key, permutation in mappings.items():
        trials = bank[key]["rows"].trial_id.tolist()
        mapping_rows.extend(
            {"group": key[0], "condition": key[1], "a_trial_id": trial, "b_trial_id": trials[permutation[i]]}
            for i, trial in enumerate(trials)
        )
    pd.DataFrame(mapping_rows).to_csv(output / "shuffled_training_pairs.csv", index=False)
    report = {
        "scope": "Exploratory validation-only frozen-backbone shared partner projection training",
        "initialization": initialization,
        "seed": seed,
        "permutation_seed": permutation_seed,
        "checkpoint_sha256": sha256(checkpoint) if initialization == "pretrained" else None,
        "cache_sha256": sha256(cache_path / "cache.json"),
        "gallery_sha256": sha256(output / "galleries.csv"),
        "backbone_sha256": sha256(output / "backbone.pt"),
        "shuffled_training_pairs_sha256": sha256(output / "shuffled_training_pairs.csv"),
        "validation_fold": 1,
        "test_fold": 2,
        "test_evaluated": False,
        "pairs": pairs,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "projection_lr": 1e-3,
        "weight_decay": 0.01,
        "temperature": 0.1,
        "feature_device": device,
        "projection_device": "cpu",
        "torch_version": torch.__version__,
        "pooled": aggregate_queries(pooled),
        "initial_projection": aggregate_queries(initial),
        "chance_top1": 1 / pairs,
        "chance_mrr": sum(1 / i for i in range(1, pairs + 1)) / pairs,
        "arms": {},
    }
    keys = list(mappings)
    for arm in ("true", "shuffled"):
        folder = output / arm
        folder.mkdir()
        head = copy.deepcopy(initial_head)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
        rng = np.random.default_rng(seed + 4000)
        best, best_epoch = report["initial_projection"]["macro"]["mrr"], 0
        history = [{"epoch": 0, "train_loss": None, **report["initial_projection"]["macro"]}]
        best_rows = initial.copy()
        torch.save(head.state_dict(), folder / "best_projection.pt")
        for epoch in range(1, epochs + 1):
            head.train()
            losses = []
            for _ in range(steps_per_epoch):
                key = keys[int(rng.integers(len(keys)))]
                entry = bank[key]
                indices = rng.choice(len(entry["a"]), pairs, replace=False)
                partner_indices = mappings[key][indices] if arm == "shuffled" else indices
                a, b = head(entry["a"][indices]), head(entry["b"][partner_indices])
                loss = contrastive_loss(a, b, 0.1)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite alignment loss.")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(loss.item())
            validation = evaluate_projection(head, bank, galleries)
            metrics = aggregate_queries(validation)["macro"]
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
            pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
            validation.to_csv(folder / f"validation_epoch_{epoch:03d}.csv", index=False)
            if metrics["mrr"] > best:
                best, best_epoch, best_rows = metrics["mrr"], epoch, validation
                torch.save(head.state_dict(), folder / "best_projection.pt")
            print(f"{arm} epoch {epoch}: loss={np.mean(losses):.4f}, validation MRR={metrics['mrr']:.4f}", flush=True)
        best_rows.to_csv(folder / "validation_best.csv", index=False)
        best_rows.groupby(["group", "condition"])[METRICS].mean().to_csv(folder / "dyads.csv")
        report["arms"][arm] = {
            "best_epoch": best_epoch,
            **aggregate_queries(best_rows),
            "projection_sha256": sha256(folder / "best_projection.pt"),
        }
    (output / "alignment.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
