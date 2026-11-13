"""Shared and participant-specific ridge retrieval with fixed movement targets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional

from .contrastive import retrieval_metrics
from .tiny_fit import digest, require
from .training import OFFSETS, FeatureBank

LAMBDAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


class RidgeSystem:
    """Cache a float64 eigensystem for positive-regularization primal or dual solves."""

    def __init__(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        """Form weighted normal equations with a unit-total example weight vector."""
        x, y, weights = (
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(weights, dtype=np.float64),
        )
        require(x.ndim == y.ndim == 2 and len(x) == len(y) == len(weights), "Invalid regression shapes")
        require(len(x) > 0 and weights.ndim == 1 and np.all(weights > 0), "Invalid sample weights")
        require(
            np.isclose(weights.sum(), 1) and all(np.isfinite(v).all() for v in (x, y, weights)),
            "Invalid weighted inputs",
        )
        xw, yw = x * np.sqrt(weights[:, None]), y * np.sqrt(weights[:, None])
        self.basis = xw if len(x) < x.shape[1] else None
        gram = xw @ xw.T if self.basis is not None else xw.T @ xw
        rhs = yw if self.basis is not None else xw.T @ yw
        values, vectors = np.linalg.eigh(gram)
        self.values = np.maximum(values, 0)
        self.vectors = vectors
        self.rotated_rhs = vectors.T @ rhs

    def solve(self, regularization: float) -> dict[str, np.ndarray]:
        """Solve without explicit inversion, preserving compact dual coefficients when useful."""
        require(np.isfinite(regularization) and regularization > 0, "Regularization must be finite and positive")
        coefficients = self.vectors @ (self.rotated_rhs / (self.values[:, None] + regularization))
        require(np.isfinite(coefficients).all(), "Nonfinite ridge solution")
        result = {"coefficients": coefficients}
        if self.basis is not None:
            result["basis"] = self.basis
        return result


def predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Apply a fitted primal or dual map to normalized EEG features."""
    x = np.asarray(x, dtype=np.float64)
    return (x @ model["basis"].T if "basis" in model else x) @ model["coefficients"]


def training_examples(sets: list[dict], shuffled: bool) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Collect training-only examples, keeping the same trial permutation across offsets."""
    people: dict[str, list] = {}
    for gallery in sets:
        if gallery["split"] != "train":
            continue
        name = f"{gallery['group']}:{gallery['member']}"
        movement = gallery["movement"][gallery["mapping"]] if shuffled else gallery["movement"]
        people.setdefault(name, []).append((gallery["eeg"].numpy(), movement.numpy()))
    require(bool(people), "No training examples")
    return {
        name: (np.concatenate([x for x, _ in rows]), np.concatenate([y for _, y in rows]))
        for name, rows in people.items()
    }


def make_systems(people: dict[str, tuple[np.ndarray, np.ndarray]], family: str) -> dict[str, RidgeSystem]:
    """Fit shared equal-participant or individual unit-weighted eigensystems."""
    require(family in ("shared", "participant"), "Unknown ridge family")
    if family == "shared":
        x = np.concatenate([xy[0] for xy in people.values()])
        y = np.concatenate([xy[1] for xy in people.values()])
        weights = np.concatenate([np.full(len(xy[0]), 1 / (len(people) * len(xy[0]))) for xy in people.values()])
        return {"shared": RidgeSystem(x, y, weights)}
    return {name: RidgeSystem(x, y, np.full(len(x), 1 / len(x))) for name, (x, y) in people.items()}


def evaluate_ridge(models: dict, sets: list[dict], shuffled: bool) -> tuple[pd.DataFrame, dict]:
    """Rank fixed candidates by cosine and report true/assigned normalized prediction errors."""
    rows, matrices = [], {}
    for gallery in sets:
        name = "shared" if "shared" in models else f"{gallery['group']}:{gallery['member']}"
        predicted = torch.from_numpy(predict(models[name], gallery["eeg"].numpy()))
        target = gallery["movement"].double()
        require(bool(torch.isfinite(predicted).all()), "Nonfinite predictions")
        scores = functional.normalize(predicted, dim=1) @ functional.normalize(target, dim=1).T
        assigned = target[gallery["mapping"]] if shuffled and gallery["split"] == "train" else target
        true_error = (predicted - target).square().mean(1)
        assigned_error = (predicted - assigned).square().mean(1)
        zero_error = target.square().mean(1)
        matrices[gallery["gallery_id"]] = scores.numpy()
        for condition in ("all", "closed1", "closed2"):
            indices = torch.tensor(
                [i for i, c in enumerate(gallery["conditions"]) if condition == "all" or c == condition],
                dtype=torch.long,
            )
            if len(indices) == 0:
                continue
            rows.append(
                dict(
                    gallery_id=gallery["gallery_id"],
                    group=gallery["group"],
                    member=gallery["member"],
                    split=gallery["split"],
                    offset=gallery["offset"],
                    condition=condition,
                    candidates=len(target),
                    queries=len(indices),
                    mse=float(true_error[indices].mean()),
                    assigned_mse=float(assigned_error[indices].mean()),
                    zero_prediction_mse=float(zero_error[indices].mean()),
                    zero_prediction_fraction=float((predicted[indices].norm(dim=1) < 1e-12).double().mean()),
                    **retrieval_metrics(scores[indices], indices),
                )
            )
    return pd.DataFrame(rows), matrices


def summarize_ridge(frame: pd.DataFrame) -> dict:
    """Match CL's equal-dyad summaries and exclude pair 2020 from selection."""
    result = {}
    for split in ("train", "validation"):
        rows = frame[(frame.split == split) & (frame.condition == "all") & (frame.group != 2020)]
        require(not rows.empty, "No eligible summary dyads")
        for metric in ("top1", "mrr", "chance_top1", "chance_mrr", "mse", "assigned_mse", "zero_prediction_mse"):
            result[f"{split}_{metric}"] = float(rows.groupby("group")[metric].mean().mean())
    return result


def run_ridge(features: Path, run: Path, output: Path, lambdas: tuple[float, ...] = LAMBDAS, seed: int = 42) -> dict:
    """Compare fixed-target ridge families with globally selected regularization and matched controls."""
    from .plotting import plot_ridge_history

    require(bool(lambdas) and len(set(lambdas)) == len(lambdas), "Provide distinct regularization values")
    require(all(np.isfinite(v) and v > 0 for v in lambdas) and seed >= 0, "Invalid regularization/seed")
    grid = sorted(lambdas, reverse=True)
    bank = FeatureBank(features, run)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(
        format_version=1,
        operation="ridge",
        readout="temporal",
        target="self",
        profile="combined-closed",
        lambdas=grid,
        seed=seed,
        offsets=list(OFFSETS),
        features_source=str(features.resolve()),
        split_source=str(run.resolve()),
        features_sha256=digest(features / "features.json"),
        split_sha256=digest(run / "split.json"),
        objective="sum of weighted squared vector residuals + lambda * squared Frobenius norm; weights sum to one",
        shared_weighting="each participant receives equal total weight; each of their training windows equal weight",
        participant_weighting="each person's training windows sum to weight one",
        normalization="participant training-only mean/scalar RMS vector norm, identical to CL",
        selection="one lambda per family/arm by dyad-macro validation MRR; exact ties prefer larger lambda",
        scoring="cosine in normalized movement space; no learned target encoder, no unit-circle renormalization",
        error_units="MSE per component of normalized 400-dimensional movement vector; zero prediction is training mean",
        diagnostic_scope="Known-person held-trial retrieval; not an isolated loss ablation or significance test",
        test_features_accessed=False,
        reserved_features_accessed=False,
        temperature=None,
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    sets, stats = bank.build("temporal", seed + 1042)
    torch.save(stats, output / "normalization.pt")
    manifest = [
        {key: value for key, value in g.items() if key not in ("eeg", "movement", "mapping")}
        | {"shuffled_positive_columns": g["mapping"].tolist()}
        for g in sets
    ]
    (output / "galleries.json").write_text(json.dumps(manifest, indent=2) + "\n")
    comparisons, history = [], []
    for arm in ("true", "shuffled"):
        people = training_examples(sets, arm == "shuffled")
        for family in ("shared", "participant"):
            destination = output / family / arm
            destination.mkdir(parents=True)
            systems = make_systems(people, family)
            best, selected = -np.inf, None
            for value in grid:
                models = {name: system.solve(value) for name, system in systems.items()}
                details, matrices = evaluate_ridge(models, sets, arm == "shuffled")
                summary = dict(family=family, arm=arm, regularization=value, **summarize_ridge(details))
                history.append(summary)
                print(
                    f"Ridge {family}/{arm} lambda={value:g}: val top1={summary['validation_top1']:.3f} "
                    f"MRR={summary['validation_mrr']:.3f}",
                    flush=True,
                )
                pd.DataFrame(history).to_csv(output / "regularization.csv", index=False)
                plot_ridge_history(pd.DataFrame(history), output)
                if summary["validation_mrr"] > best:
                    best, selected = summary["validation_mrr"], summary
                    torch.save(
                        {
                            name: {key: torch.from_numpy(v) for key, v in model.items()}
                            for name, model in models.items()
                        },
                        destination / "best_models.pt",
                    )
                    details.to_csv(destination / "best_gallery_metrics.csv", index=False)
                    np.savez_compressed(destination / "best_scores.npz", **matrices)
                    (destination / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
            comparisons.append(selected)
            pd.DataFrame(comparisons).to_csv(output / "comparison.csv", index=False)
            del systems, models
    result = dict(
        **config,
        complete=True,
        results=comparisons,
        participant_count=len(people),
        training_participant_windows=sum(len(x) for x, _ in people.values()),
        input_array_sha256={name: bank.metadata["artifact_sha256"][name] for name in sorted(bank.verified)},
        versions=dict(numpy=np.__version__, torch=torch.__version__, pandas=pd.__version__),
        code_sha256={
            name: digest(Path(__file__).with_name(name))
            for name in ("ridge.py", "training.py", "tiny_fit.py", "contrastive.py", "plotting.py")
        },
        output_sha256={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    (output / "ridge.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
