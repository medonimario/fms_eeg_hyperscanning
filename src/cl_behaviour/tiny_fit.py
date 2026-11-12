"""Training-only real-feature memorization diagnostic, independent of adjacent projects."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional

from .contrastive import paired_logits, retrieval_metrics, symmetric_loss
from .models import LinearHeads

READOUTS = ("pooled", "temporal", "full")


def digest(path: Path) -> str:
    """Hash a file without loading large feature arrays into memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    """Reject malformed or out-of-scope inputs explicitly."""
    if not condition:
        raise ValueError(message)


def checked_path(root: Path, name: str, hashes: dict) -> Path:
    """Resolve a simple artifact filename only after verifying its expected digest."""
    require(Path(name).name == name, "Unsafe feature filename")
    path = root / name
    require(digest(path) == hashes.get(name), f"Artifact hash mismatch: {name}")
    return path


def load_tiny(
    features: Path,
    run: Path,
    pairs: int = 8,
    group: int | None = None,
    member: str = "a",
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict, pd.DataFrame, dict]:
    """Load one person's fixed-offset training trials, with no held-out slice materialization.

    Complete files are hashed for integrity; their validation rows are not used for fitting or scoring.
    Metadata is cross-checked against the approved split before any feature array is opened.
    """
    require(pairs >= 2 and member in ("a", "b") and offset in range(0, 15, 2), "Invalid tiny-fit selection")
    metadata = json.loads((features / "features.json").read_text())
    expected = dict(
        format_version=1,
        operation="features",
        complete=True,
        backbone_frozen=True,
        members=["a", "b"],
        token_shape=[2, 64, 4, 200],
        movement_shape=[2, 2, 900],
        token_order="channel-major, time-minor",
        movement_components=["sin", "cos"],
        movement_sfreq=50,
        centered=False,
        statistics_fitted=False,
    )
    for key, value in expected.items():
        require(metadata.get(key) == value, f"Invalid feature contract: {key}")
    protocol = json.loads((run / "split.json").read_text())
    require(protocol.get("format_version") == 2 and protocol.get("operation") == "split", "Expected split format 2")
    require(digest(run / "split.json") == metadata["split_sha256"], "Split identity mismatch")
    require(metadata["cache_sha256"] == protocol["cache_sha256"], "Preparation identity mismatch")
    require(protocol["profile"] == metadata["profile"] == "combined-closed", "Tiny-fit expects combined closed profile")
    source = pd.read_csv(checked_path(run, "splits.csv", protocol["manifest_sha256"]), keep_default_na=False)
    source_windows = pd.read_csv(checked_path(run, "windows.csv", protocol["manifest_sha256"]), keep_default_na=False)
    trials = pd.read_csv(checked_path(features, "trials.csv", metadata["artifact_sha256"]), keep_default_na=False)
    windows = pd.read_csv(checked_path(features, "windows.csv", metadata["artifact_sha256"]), keep_default_na=False)
    require(not trials.trial_id.duplicated().any() and not windows.pair_id.duplicated().any(), "Duplicate identities")
    require(set(trials.trial_id) <= set(source.trial_id), "Unknown feature trials")
    require(set(windows.pair_id) <= set(source_windows.pair_id), "Unknown feature windows")
    for actual, original, key in ((trials, source, "trial_id"), (windows, source_windows, "pair_id")):
        expected_rows = original.set_index(key, drop=False).loc[actual[key]].reset_index(drop=True)
        for column in original.columns:
            require((actual[column].reset_index(drop=True) == expected_rows[column]).all(), f"Split mismatch: {column}")
        require(actual.eligible.all() and actual.split.isin(["train", "validation"]).all(), "Forbidden feature rows")
        require(not actual.cohort_fold.isin([1, 2]).any(), "Reserved feature rows")
    candidates = trials[(trials.split == "train") & trials.main_summary_eligible]
    if group is not None:
        candidates = candidates[candidates.group == group]
    chosen = None
    for _, rows in candidates.groupby("group", sort=True):
        if len(rows) >= pairs:
            chosen = rows.sort_values(["condition_trial", "condition"]).head(pairs).copy()
            break
    require(chosen is not None, "No permitted dyad has enough training trials")
    selected = windows[(windows.trial_id.isin(chosen.trial_id)) & (windows.crop_start_s == offset)].copy()
    require(len(selected) == pairs and not selected.trial_id.duplicated().any(), "Missing fixed-offset windows")
    selected = selected.set_index("trial_id", drop=False).loc[chosen.trial_id].reset_index(drop=True)
    group = int(chosen.group.iloc[0])
    require(selected.group.eq(group).all(), "Mixed dyads")
    require(selected.token_array.eq(f"{group}_tokens.npy").all(), "Unexpected token file")
    require(selected.movement_feature_array.eq(f"{group}_movement.npy").all(), "Unexpected movement file")
    parent = chosen.set_index("trial_id").loc[selected.trial_id]
    require(np.array_equal(selected.movement_feature_index, parent.movement_feature_index), "Movement index mismatch")
    require(selected.movement_feature_start.eq(offset * 50).all(), "Movement start mismatch")
    require(selected.movement_feature_stop.eq((offset + 4) * 50).all(), "Movement stop mismatch")
    selected["member"] = member
    arrays, used_hashes = [], {}
    for name, column, shape in (
        (f"{group}_tokens.npy", "token_array_index", (int((windows.group == group).sum()), 2, 64, 4, 200)),
        (f"{group}_movement.npy", "movement_feature_index", (int((trials.group == group).sum()), 2, 2, 900)),
    ):
        path = checked_path(features, name, metadata["artifact_sha256"])
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        require(array.shape == shape and array.dtype == np.float32, f"Invalid feature array: {name}")
        indices = selected[column].to_numpy()
        require(len(np.unique(indices)) == pairs, "Duplicate selected feature indices")
        require(np.issubdtype(indices.dtype, np.integer), "Noninteger feature indices")
        require(bool(((indices >= 0) & (indices < len(array))).all()), "Feature index out of range")
        values = np.array(array[indices, "ab".index(member)], copy=True)
        if column == "movement_feature_index":
            values = values[:, :, offset * 50 : (offset + 4) * 50]
        require(np.isfinite(values).all(), "Nonfinite selected features")
        arrays.append(torch.from_numpy(values))
        used_hashes[name] = metadata["artifact_sha256"][name]
    norm_path = checked_path(features, "pooling_norm.npz", metadata["artifact_sha256"])
    with np.load(norm_path, allow_pickle=False) as saved:
        norm = {key: np.array(saved[key], copy=True) for key in ("weight", "bias", "eps")}
    require(norm["weight"].shape == norm["bias"].shape == (200,), "Invalid pooling norm shape")
    require(all(np.isfinite(value).all() for value in norm.values()) and float(norm["eps"]) > 0, "Invalid norm values")
    used_hashes["pooling_norm.npz"] = digest(norm_path)
    provenance = dict(
        features_sha256=digest(features / "features.json"),
        split_sha256=digest(run / "split.json"),
        artifact_sha256=used_hashes,
        group=group,
        member=member,
        offset=offset,
    )
    return arrays[0], arrays[1], norm, selected, provenance


def readout(tokens: torch.Tensor, norm: dict, profile: str) -> torch.Tensor:
    """Derive original pooled, ordered temporal or flattened full normalized tokens."""
    require(tokens.ndim == 4 and tokens.shape[1:] == (64, 4, 200), "Invalid token grid")
    require(profile in READOUTS, "Unknown readout")
    values = tokens.mean((1, 2)) if profile == "pooled" else tokens.mean(1) if profile == "temporal" else tokens
    values = functional.layer_norm(
        values, (200,), torch.as_tensor(norm["weight"]), torch.as_tensor(norm["bias"]), float(norm["eps"])
    )
    return values.flatten(1)


def center_scale(values: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Fit a selected-training-set mean and scalar RMS vector norm, without featurewise scaling."""
    mean = values.mean(0, keepdim=True)
    centered = values - mean
    scale = centered.square().sum(1).mean().sqrt()
    require(bool(torch.isfinite(scale)) and float(scale) > 1e-12, "Degenerate selected features")
    return centered / scale, dict(mean=mean, scale=scale)


def fit_tiny_arm(
    initial: LinearHeads,
    eeg: torch.Tensor,
    movement: torch.Tensor,
    mapping: torch.Tensor,
    steps: int = 500,
    learning_rate: float = 0.01,
) -> tuple[dict, list[dict], dict, np.ndarray]:
    """Memorize one gallery using a fixed budget and report both assigned and true correspondence."""
    model = copy.deepcopy(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    before = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
    history, max_gradient = [], 0.0
    for step in range(steps + 1):
        logits = paired_logits(*model(eeg, movement), temperature=0.1)
        loss = symmetric_loss(logits, mapping)
        require(bool(torch.isfinite(loss)), "Nonfinite loss")
        if step % 25 == 0 or step == steps:
            forward, reverse = retrieval_metrics(logits, mapping), retrieval_metrics(logits.T, torch.argsort(mapping))
            history.append(
                dict(
                    step=step,
                    loss=float(loss.detach()),
                    assigned_top1=forward["top1"],
                    assigned_mrr=forward["mrr"],
                    reverse_top1=reverse["top1"],
                    reverse_mrr=reverse["mrr"],
                    true_top1=retrieval_metrics(logits)["top1"],
                )
            )
        if step == steps:
            break
        optimizer.zero_grad()
        loss.backward()
        require(
            all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), "Invalid gradients"
        )
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        max_gradient = max(max_gradient, float(gradient))
        optimizer.step()
    after = torch.cat([p.detach().flatten() for p in model.parameters()])
    result = dict(
        **history[-1],
        initial_loss=history[0]["loss"],
        max_gradient_norm=max_gradient,
        parameter_change_norm=float((after - before).norm()),
        passed=history[-1]["assigned_top1"] == 1.0
        and history[-1]["reverse_top1"] == 1.0
        and history[-1]["loss"] < history[0]["loss"]
        and max_gradient > 0
        and float((after - before).norm()) > 0,
    )
    return result, history, model.state_dict(), logits.detach().numpy()


def run_tiny_fit(
    features: Path,
    run: Path,
    output: Path,
    pairs: int = 8,
    group: int | None = None,
    member: str = "a",
    offset: int = 0,
    profiles: tuple[str, ...] = READOUTS,
    steps: int = 500,
    seed: int = 42,
) -> dict:
    """Run matched true/shuffled tiny fits on one fixed training-only participant gallery."""
    require(steps > 0 and seed >= 0, "Positive steps and nonnegative seed required")
    require(
        bool(profiles) and len(set(profiles)) == len(profiles) and set(profiles) <= set(READOUTS), "Invalid readouts"
    )
    tokens, movement, norm, selected, provenance = load_tiny(features, run, pairs, group, member, offset)
    output.mkdir(parents=True, exist_ok=False)
    selected.to_csv(output / "selected_windows.csv", index=False)
    movement, movement_stats = center_scale(movement.flatten(1))
    mapping = torch.randperm(pairs, generator=torch.Generator().manual_seed(seed + 1042))
    results, histories = [], []
    for profile in profiles:
        print(f"Tiny-fit {profile}: {pairs} trials, pair {provenance['group']}{member}, offset {offset}s", flush=True)
        eeg, eeg_stats = center_scale(readout(tokens, norm, profile))
        np.savez(
            output / f"{profile}_normalization.npz",
            **{
                f"{modality}_{name}": value.numpy()
                for modality, stats in (("eeg", eeg_stats), ("movement", movement_stats))
                for name, value in stats.items()
            },
        )
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            initial = LinearHeads(eeg.shape[1], movement.shape[1], 32)
        for arm, positives in (("true", torch.arange(pairs)), ("shuffled", mapping)):
            metrics, history, state, scores = fit_tiny_arm(initial, eeg, movement, positives, steps)
            results.append(dict(profile=profile, arm=arm, **metrics))
            histories.extend(dict(profile=profile, arm=arm, **row) for row in history)
            torch.save(state, output / f"{profile}_{arm}_heads.pt")
            np.save(output / f"{profile}_{arm}_logits.npy", scores, allow_pickle=False)
    pd.DataFrame(histories).to_csv(output / "history.csv", index=False)
    pd.DataFrame(results).to_csv(output / "comparison.csv", index=False)
    result = dict(
        format_version=1,
        operation="tiny-fit",
        complete=True,
        passed=all(row["passed"] for row in results),
        scope="Training-only memorization; no generalization claim; never reuse these heads for evaluation",
        **provenance,
        profiles=list(profiles),
        steps=steps,
        seed=seed,
        pairs=pairs,
        chance_top1=1 / pairs,
        negatives_per_query=pairs - 1,
        embedding_dim=32,
        training="CPU full gallery; symmetric hard-target loss; temperature .1; AdamW lr=.01 decay=0; clip=5",
        normalization="each modality: selected participant/training trials mean; scalar RMS vector norm",
        shuffled_positive_columns=mapping.tolist(),
        selection="fixed final step, no held-out evaluation",
        gate="100% assigned-pair retrieval in both directions in every arm/readout, decreasing loss, finite updates",
        note="Fixed shuffled pairs may memorize too; this is expected and not a chance test",
        validation_features_materialized=False,
        test_features_accessed=False,
        results=results,
        versions=dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__),
        code_sha256={
            name: digest(Path(__file__).with_name(name)) for name in ("tiny_fit.py", "models.py", "contrastive.py")
        },
        output_sha256={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    )
    (output / "tiny_fit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
