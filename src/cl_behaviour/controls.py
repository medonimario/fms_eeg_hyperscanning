"""Reproducible CPU synthetic correspondence controls, independent of real EEG code."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .contrastive import paired_logits, retrieval_metrics, symmetric_loss
from .models import LinearHeads


def synthetic_pairs(seed: int) -> tuple[torch.Tensor, ...]:
    """Make disjoint training/validation trials with shared hidden signal and modality offsets.

    Both modalities are noisy full-rank linear views of a 16-dimensional latent signal.
    Only training rows fit modality means. Validation has 16 fixed galleries of eight trials.
    """
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(384, 16, generator=generator)
    views = []
    for width in (48, 32):
        mapping = torch.randn(16, width, generator=generator) / 4
        offset = 10 * torch.randn(1, width, generator=generator)
        values = latent @ mapping + offset + 0.05 * torch.randn(384, width, generator=generator)
        mean = values[:256].mean(0, keepdim=True)
        views.extend((values[:256] - mean, values[256:] - mean))
    return views[0], views[2], views[1], views[3]


@torch.no_grad()
def score_galleries(eeg: torch.Tensor, movement: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """Score 16 fixed eight-trial validation galleries in both retrieval directions."""
    matrices, forward, reverse = [], [], []
    for start in range(0, len(eeg), 8):
        matrix = paired_logits(eeg[start : start + 8], movement[start : start + 8])
        matrices.append(matrix)
        forward.append(retrieval_metrics(matrix))
        reverse.append(retrieval_metrics(matrix.T))
    result = {f"eeg_to_movement_{key}": float(np.mean([m[key] for m in forward])) for key in forward[0]}
    result.update({f"movement_to_eeg_{key}": float(np.mean([m[key] for m in reverse])) for key in reverse[0]})
    return result, torch.stack(matrices)


def fit_arm(
    initial: LinearHeads,
    data: tuple[torch.Tensor, ...],
    mapping: torch.Tensor,
    steps: int,
) -> tuple[dict, list[dict], torch.Tensor]:
    """Train one matched arm for a fixed budget with no validation-based selection."""
    model = copy.deepcopy(initial)
    eeg, movement, valid_eeg, valid_movement = data
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    before = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
    history = []
    gradient_max = 0.0
    for step in range(steps + 1):
        left, right = model(eeg, movement)
        loss = symmetric_loss(paired_logits(left, right), mapping)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite synthetic loss")
        if step % 25 == 0 or step == steps:
            with torch.no_grad():
                validation, _ = score_galleries(*model(valid_eeg, valid_movement))
            history.append(dict(step=step, loss=float(loss.detach()), **validation))
        if step == steps:
            break
        optimizer.zero_grad()
        loss.backward()
        gradients = [p.grad for p in model.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise ValueError("Missing or nonfinite gradients")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        gradient_max = max(gradient_max, float(norm))
        optimizer.step()
    with torch.no_grad():
        metrics, matrices = score_galleries(*model(valid_eeg, valid_movement))
        train = retrieval_metrics(paired_logits(*model(eeg, movement)), mapping)
        after = torch.cat([p.detach().flatten() for p in model.parameters()])
    metrics.update(
        final_loss=float(loss.detach()),
        initial_loss=history[0]["loss"],
        train_assigned_pair_top1=train["top1"],
        gradient_norm_max=gradient_max,
        parameter_change_norm=float((after - before).norm()),
    )
    return metrics, history, matrices


def independent_pairing_null(matrices: torch.Tensor, seed: int, repeats: int = 512) -> dict:
    """Randomize held-out correspondence within galleries; fixed points are permitted.

    This is an engineering null for label indexing and chance, not a scientific EEG test.
    A training-only shuffle is reported separately and is not required to be at chance.
    """
    generator = torch.Generator().manual_seed(seed)
    draws = []
    for _ in range(repeats):
        values = [retrieval_metrics(matrix, torch.randperm(8, generator=generator)) for matrix in matrices]
        draws.append([np.mean([v[key] for v in values]) for key in ("top1", "mrr")])
    array = np.asarray(draws)
    chance = np.array([1 / 8, sum(1 / i for i in range(1, 9)) / 8])
    error = np.abs(array.mean(0) - chance)
    tolerance = 6 * array.std(0, ddof=1) / np.sqrt(repeats) + 1e-12
    return dict(
        repeats=repeats,
        mean_top1=float(array[:, 0].mean()),
        mean_mrr=float(array[:, 1].mean()),
        chance_top1=float(chance[0]),
        chance_mrr=float(chance[1]),
        six_standard_error_tolerance=tolerance.tolist(),
        passed=bool(np.all(error <= tolerance)),
    )


def run_controls(output: Path, seeds: tuple[int, ...] = (42, 43, 44), steps: int = 250) -> dict:
    """Run matched synthetic arms and write a pass/fail report in a fresh directory."""
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds) or steps < 1:
        raise ValueError("Use distinct nonnegative seeds and a positive step budget")
    output.mkdir(parents=True, exist_ok=False)
    results, histories, mappings = [], [], {}
    for seed in seeds:
        print(f"Synthetic correspondence seed {seed}", flush=True)
        data = synthetic_pairs(seed)
        with torch.random.fork_rng():
            torch.manual_seed(seed + 1000)
            initial = LinearHeads(48, 32, 16)
        shuffled = torch.randperm(256, generator=torch.Generator().manual_seed(seed + 2000))
        mappings[str(seed)] = shuffled.tolist()
        for arm, mapping in (("true", torch.arange(256)), ("shuffled", shuffled)):
            metrics, history, matrices = fit_arm(initial, data, mapping, steps)
            histories.extend(dict(seed=seed, arm=arm, **row) for row in history)
            result = dict(seed=seed, arm=arm, **metrics)
            if arm == "true":
                result["independent_validation_pairing"] = independent_pairing_null(matrices, seed + 3000)
                result["passed"] = bool(
                    metrics["eeg_to_movement_top1"] >= 0.9
                    and metrics["movement_to_eeg_top1"] >= 0.9
                    and metrics["final_loss"] < metrics["initial_loss"]
                    and metrics["parameter_change_norm"] > 0
                    and metrics["gradient_norm_max"] > 0
                    and result["independent_validation_pairing"]["passed"]
                )
            results.append(result)
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(histories[0]))
        writer.writeheader()
        writer.writerows(histories)
    (output / "shuffled_mappings.json").write_text(json.dumps(mappings, indent=2) + "\n")
    report = dict(
        format_version=1,
        operation="synthetic-controls",
        complete=True,
        passed=all(row["passed"] for row in results if row["arm"] == "true"),
        scope="Engineering checks only; no real EEG, movement, LaBraM or cross-project imports",
        seeds=list(seeds),
        steps=steps,
        train_trials=256,
        validation_trials=128,
        validation_gallery_size=8,
        validation_galleries=16,
        negatives_per_training_query=255,
        training="full training gallery; symmetric hard-target cosine CE; AdamW lr=.01 decay=.01; temperature=.1",
        centering="separate modality means fitted on synthetic training trials only",
        selection="fixed final step; no validation checkpoint selection",
        gate="each true seed: bidirectional top1 >= .90, lower loss, nonzero updates/gradients, null passes",
        shuffled_arm="same initialization and budget; fixed training permutation, true validation correspondence",
        results=results,
        versions=dict(torch=torch.__version__, numpy=np.__version__),
        code_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("controls.py", "contrastive.py", "models.py")
        },
        artifact_sha256={
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("history.csv", "shuffled_mappings.json")
        },
    )
    (output / "controls.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError(f"Synthetic control gate failed; see {output / 'controls.json'}")
    return report
