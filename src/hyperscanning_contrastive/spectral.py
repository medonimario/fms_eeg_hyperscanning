"""CPU spectral partner retrieval on a verified, existing within-dyad experiment."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from labram_classification.baseline import BANDS, bandpower_features, feature_names

from .alignment import aggregate_queries, evaluate_projection, training_permutations
from .alignment_diagnostics import record_diagnostics
from .contrastive import contrastive_loss
from .data import sha256
from .partner import METRICS
from .ssl import EEGCache, SSLConfig
from .trial_split import split_within_dyad


class FeatureStandardizer(nn.Module):
    """Apply fixed training-only means and population standard deviations."""

    def __init__(self, bank: dict) -> None:
        """Fit both participants jointly, using only the supplied training bank."""
        super().__init__()
        if any(entry["split"] != "train" for entry in bank.values()):
            raise ValueError("Standardization requires training features only.")
        values = torch.cat([entry[member] for entry in bank.values() for member in ("a", "b")])
        self.register_buffer("mean", values.mean(0))
        self.register_buffer("scale", values.std(0, unbiased=False).clamp_min(1e-6))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Transform without fitting on validation galleries or batches."""
        return (values - self.mean) / self.scale


def spectral_bank(cache: EEGCache, split: str) -> dict:
    """Extract fixed log-power features from one eligible split in bounded CPU batches."""
    bank = {}
    for frame in cache.strata[split]:
        if int(frame.member.iloc[0]) != 0:
            continue
        entry = {"rows": frame.copy(), "split": split}
        for member, label in enumerate(("a", "b")):
            chunks = []
            for start in range(0, len(frame), 8):
                signals = cache.signals(frame.iloc[start : start + 8].assign(member=member)).numpy()
                chunks.append(torch.from_numpy(bandpower_features(signals)))
            entry[label] = torch.cat(chunks)
        bank[(int(frame.group.iloc[0]), frame.condition.iloc[0])] = entry
    return bank


def train_spectral(cache_path: Path, output: Path, reference: Path) -> dict:
    """Compare direct spectra and learned projections on exactly the reference trial protocol."""
    source = json.loads((reference / "adaptation.json").read_text())
    if source.get("validation_protocol") != "within-dyad":
        raise ValueError("Spectral comparison requires a completed within-dyad adaptation run.")
    if sha256(cache_path / "cache.json") != source["cache_sha256"]:
        raise ValueError("Reference cache hash mismatch.")
    checks = {
        "trial_splits.csv": "trial_splits_sha256",
        "galleries.csv": "gallery_sha256",
        "shuffled_training_pairs.csv": "shuffled_training_pairs_sha256",
    }
    for name, key in checks.items():
        if sha256(reference / name) != source[key]:
            raise ValueError(f"Reference manifest hash mismatch: {name}")
    config = SSLConfig(
        pairs=source["pairs"], epochs=source["epochs"], steps_per_epoch=source["steps_per_epoch"], seed=source["seed"]
    )
    cache = EEGCache(cache_path, config)
    split_within_dyad(cache, source["validation_trials_per_condition"], config.pairs, source["split_seed"])
    output.mkdir(parents=True, exist_ok=False)
    cache.rows[["group", "condition", "trial_id", "condition_trial", "fold", "split"]].to_csv(
        output / "trial_splits.csv", index=False
    )
    galleries = [f for f in cache.galleries("validation", config) if int(f.member.iloc[0]) == 0]
    pd.concat(galleries).to_csv(output / "galleries.csv", index=False)
    training_rows = {
        (int(f.group.iloc[0]), f.condition.iloc[0]): f for f in cache.strata["train"] if int(f.member.iloc[0]) == 0
    }
    mappings = training_permutations(
        {key: {"a": frame, "split": "train"} for key, frame in training_rows.items()}, source["permutation_seed"]
    )
    mapping_rows = []
    for key, permutation in mappings.items():
        trials = training_rows[key].trial_id.tolist()
        mapping_rows.extend(
            {"group": key[0], "condition": key[1], "a_trial_id": trial, "b_trial_id": trials[permutation[i]]}
            for i, trial in enumerate(trials)
        )
    pd.DataFrame(mapping_rows).to_csv(output / "shuffled_training_pairs.csv", index=False)
    for name, key in checks.items():
        if sha256(output / name) != source[key]:
            raise ValueError(f"Reconstructed protocol differs from reference: {name}")
    bank = spectral_bank(cache, "train")
    validation_bank = spectral_bank(cache, "validation")
    standardizer = FeatureStandardizer(bank)
    torch.manual_seed(config.seed)
    initial_head = nn.Sequential(copy.deepcopy(standardizer), nn.Linear(320, 64, bias=False))
    torch.save(standardizer.state_dict(), output / "standardizer.pt")
    torch.save(initial_head.state_dict(), output / "initial_projection.pt")
    initial = evaluate_projection(initial_head, validation_bank, galleries)
    direct = evaluate_projection(standardizer, validation_bank, galleries)
    initial.to_csv(output / "validation_initial.csv", index=False)
    direct.to_csv(output / "validation_direct.csv", index=False)
    groups = source["training_diagnostic_groups"]
    training_galleries = [
        f for f in cache.galleries("train", config) if int(f.member.iloc[0]) == 0 and int(f.group.iloc[0]) in groups
    ]
    pd.concat(training_galleries).to_csv(output / "training_galleries.csv", index=False)
    if sha256(output / "training_galleries.csv") != source["training_gallery_sha256"]:
        raise ValueError("Training diagnostic galleries differ from reference.")
    report = {
        "scope": "Exploratory within-dyad spectral partner retrieval; no LaBraM or test scoring",
        "validation_protocol": "within-dyad",
        "reference_run": str(reference.resolve()),
        "reference_report_sha256": sha256(reference / "adaptation.json"),
        "cache_sha256": source["cache_sha256"],
        **{key: sha256(output / name) for name, key in checks.items()},
        "features": "log10 absolute band power in microvolts squared; channel-major then band",
        "bands_hz": BANDS,
        "feature_names": feature_names("bandpower"),
        "feature_count": 320,
        "welch": {
            "sfreq": 200,
            "window": "periodic Hann",
            "nperseg": 400,
            "noverlap": 200,
            "nfft": 400,
            "detrend": "constant",
            "average": "mean",
            "power_floor": 1e-12,
        },
        "standardization": "global window-weighted training-only mean and population SD; both participants",
        "standardizer_sha256": sha256(output / "standardizer.pt"),
        "initial_projection_sha256": sha256(output / "initial_projection.pt"),
        "training_gallery_sha256": sha256(output / "training_galleries.csv"),
        "projection_profile": "standardized-linear-64",
        "trainable_parameter_count": 320 * 64,
        "seed": config.seed,
        "split_seed": source["split_seed"],
        "permutation_seed": source["permutation_seed"],
        "split_counts": source["split_counts"],
        "training_diagnostic_groups": groups,
        "crop_offset_seconds": 0,
        "test_evaluated": False,
        "external_validation_evaluated": False,
        "pairs": config.pairs,
        "epochs": config.epochs,
        "steps_per_epoch": config.steps_per_epoch,
        "projection_lr": 1e-3,
        "weight_decay": 0.01,
        "temperature": 0.1,
        "torch_version": torch.__version__,
        "device": "cpu",
        "direct": aggregate_queries(direct),
        "initial_projection": aggregate_queries(initial),
        "chance_top1": 1 / config.pairs,
        "chance_mrr": sum(1 / i for i in range(1, config.pairs + 1)) / config.pairs,
        "arms": {},
    }
    (output / "settings.json").write_text(json.dumps(report, indent=2) + "\n")
    for arm in ("true", "shuffled"):
        folder = output / arm
        folder.mkdir()
        head = copy.deepcopy(initial_head)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
        rng = np.random.default_rng(config.seed + 4000)
        best, best_epoch, best_rows = report["initial_projection"]["macro"]["mrr"], 0, initial.copy()
        torch.save(head.state_dict(), folder / "best_projection.pt")
        history = []
        keys = list(bank)
        for epoch in range(config.epochs + 1):
            losses = []
            for _ in range(config.steps_per_epoch if epoch else 0):
                key = keys[int(rng.integers(len(keys)))]
                indices = rng.choice(len(bank[key]["a"]), config.pairs, replace=False)
                partner = mappings[key][indices] if arm == "shuffled" else indices
                loss = contrastive_loss(head(bank[key]["a"][indices]), head(bank[key]["b"][partner]))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite spectral loss.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(loss.item())
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
                {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None, **metrics, **diagnostics}
            )
            pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
            validation.to_csv(folder / f"validation_epoch_{epoch:03d}.csv", index=False)
            if metrics["mrr"] > best:
                best, best_epoch, best_rows = metrics["mrr"], epoch, validation
                torch.save(head.state_dict(), folder / "best_projection.pt")
            print(f"{arm} epoch {epoch}: validation MRR={metrics['mrr']:.4f}", flush=True)
        best_rows.to_csv(folder / "validation_best.csv", index=False)
        best_rows.groupby(["group", "condition"])[METRICS].mean().to_csv(folder / "dyads.csv")
        torch.save(head.state_dict(), folder / "last_projection.pt")
        report["arms"][arm] = {
            "best_epoch": best_epoch,
            **aggregate_queries(best_rows),
            "projection_sha256": sha256(folder / "best_projection.pt"),
        }
    report["settings_sha256"] = sha256(output / "settings.json")
    (output / "spectral.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
