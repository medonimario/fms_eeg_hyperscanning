"""Frozen EEG tokens and time-aligned self-movement features, without model fitting."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.signal import resample_poly
from torch import nn

from hyperscanning_contrastive.data import CHANNELS, sha256
from hyperscanning_contrastive.overfit import ContrastiveLabram

from .splits import build_protocol, require, validate_cache


class FrozenTokens(nn.Module):
    """Expose contextual LaBraM tokens and verify original mean-pooling reconstruction."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        """Load the original pretrained backbone with gradients and dropout disabled."""
        super().__init__()
        self.encoder = ContrastiveLabram("pretrained", checkpoint, False).to(device).eval()
        self.requires_grad_(False)
        self.eval()
        self.max_reconstruction_error = 0.0

    @torch.inference_mode()
    def forward(self, microvolts: torch.Tensor) -> torch.Tensor:
        """Return float32 (batch, channel, time patch, feature) without centering."""
        require(not self.training and not self.encoder.training, "Extractor must remain in evaluation mode")
        captured = []

        def capture(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            """Capture the final contextual sequence including its CLS token."""
            captured.append(output)

        handle = self.encoder.backbone.norm.register_forward_hook(capture)
        try:
            pooled = self.encoder.features(microvolts)
        finally:
            handle.remove()
        require(len(captured) == 1 and captured[0].shape[1:] == (257, 200), "Unexpected LaBraM tokens")
        tokens = captured[0][:, 1:]
        reconstructed = self.encoder.backbone.fc_norm(tokens.mean(1))
        torch.testing.assert_close(reconstructed, pooled, atol=1e-6, rtol=1e-5)
        self.max_reconstruction_error = max(self.max_reconstruction_error, float((reconstructed - pooled).abs().max()))
        require(bool(torch.isfinite(tokens).all()), "Nonfinite LaBraM tokens")
        return tokens.reshape(-1, 64, 4, 200).float()


def phase_features(phase: np.ndarray) -> np.ndarray:
    """Convert complete 18-second phases to sin/cos at 50 Hz before window slicing.

    Args:
        phase: Finite radians with final axis of 4320 samples at 240 Hz.

    Returns:
        Float32 features with appended component/sample axes (2, 900), ordered sin/cos.
        A zero-phase polyphase FIR uses a 5/24 ratio, Kaiser beta 5 and line edge padding.
    """
    require(phase.shape[-1] == 4320 and np.isfinite(phase).all(), "Invalid 18-second movement phase")
    components = np.stack((np.sin(phase), np.cos(phase)), axis=-2)
    result = resample_poly(components, 5, 24, axis=-1, window=("kaiser", 5.0), padtype="line")
    require(result.shape[-2:] == (2, 900) and np.isfinite(result).all(), "Invalid resampled movement")
    return result.astype(np.float32)


def feature_plan(cache: Path, run: Path, smoke: bool = False) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """Verify saved protocol against preparation and select only permitted train/validation rows."""
    metadata, original, original_windows = validate_cache(cache)
    protocol = json.loads((run / "split.json").read_text())
    require(protocol.get("format_version") == 2 and protocol.get("operation") == "split", "Expected split format 2")
    require(sha256(cache / "cache.json") == protocol["cache_sha256"], "Preparation identity mismatch")
    for name in ("splits.csv", "windows.csv", "galleries.csv"):
        require(sha256(run / name) == protocol["manifest_sha256"][name], f"Split hash mismatch: {name}")
    trials, galleries = build_protocol(
        original, {int(k): v for k, v in protocol["group_folds"].items()}, protocol["seed"], protocol["profile"]
    )
    pd.testing.assert_frame_equal(trials, pd.read_csv(run / "splits.csv", keep_default_na=False), check_dtype=False)
    pd.testing.assert_frame_equal(galleries, pd.read_csv(run / "galleries.csv"), check_dtype=False)
    expected_windows = original_windows.drop(columns="split").merge(
        trials[["trial_id", "split", "cohort_fold", "repetition_id", "eligible", "exclusion_reason", "gallery_pool"]],
        on="trial_id",
        validate="many_to_one",
        how="left",
    )
    pd.testing.assert_frame_equal(
        expected_windows, pd.read_csv(run / "windows.csv", keep_default_na=False), check_dtype=False
    )
    selected = trials[trials.eligible & trials.split.isin(["train", "validation"])].copy()
    if smoke:
        selected = selected[selected.split == "train"].sort_values(["group", "condition_trial", "condition"])
        selected = selected[selected.group == selected.group.min()].head(2).copy()
    require(len(selected) > 0 and not selected.cohort_fold.isin([1, 2]).any(), "No permitted development trials")
    windows = expected_windows[expected_windows.trial_id.isin(selected.trial_id)].copy()
    return metadata, protocol, selected, windows


def verified_array(cache: Path, name: str, metadata: dict, expected_shape: tuple[int, ...]) -> np.ndarray:
    """Hash a permitted dyad file, then memory-map it without materializing unused trials."""
    require(Path(name).name == name, "Unsafe signal filename")
    require(sha256(cache / name) == metadata["array_sha256"].get(name), f"Signal hash mismatch: {name}")
    array = np.load(cache / name, mmap_mode="r", allow_pickle=False)
    require(array.shape == expected_shape and np.issubdtype(array.dtype, np.floating), f"Invalid shape/dtype: {name}")
    if name.endswith("_eeg.npy"):
        require(array.dtype == np.float32, "EEG must be float32 microvolts")
    return array


def encode_windows(
    extractor: nn.Module, eeg: np.ndarray, indices: np.ndarray, batch_size: int, device: str, destination: Path
) -> dict:
    """Stream paired windows in bounded batches and run a repeatability gate before writing."""
    require(batch_size > 0 and len(indices) > 0, "Nonempty windows and positive batch size required")
    output = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32, shape=(len(indices), 2, 64, 4, 200))
    maximum = 0.0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = np.array(eeg[indices[start : start + batch_size]], copy=True)
            require(np.isfinite(selected).all(), "Nonfinite selected EEG")
            maximum = max(maximum, float(np.abs(selected).max()))
            signal = torch.from_numpy(selected.reshape(-1, 64, 800)).to(device)
            if start == 0:
                first = extractor(signal[:1])
                repeat = extractor(signal[:1])
                torch.testing.assert_close(first, repeat, atol=1e-6, rtol=1e-5)
            tokens = extractor(signal)
            require(tokens.shape == (len(signal), 64, 4, 200), "Unexpected token output shape")
            require(bool(torch.isfinite(tokens).all()), "Nonfinite feature output")
            output[start : start + len(selected)] = tokens.cpu().numpy().reshape(-1, 2, 64, 4, 200)
    output.flush()
    del output
    return dict(max_abs_eeg_microvolts=maximum, paired_windows=len(indices))


def extract_features(
    cache: Path,
    run: Path,
    output: Path,
    checkpoint: Path,
    device: str = "cpu",
    batch_size: int = 8,
    smoke: bool = False,
) -> dict:
    """Write an immutable paired feature cache, or a training-only smoke report.

    Signal hashes cover complete permitted-dyad files, including unused rows. Only selected
    training/validation slices enter feature computation; reserved-dyad arrays are never opened.
    Centering and all learned transforms remain deferred to training.
    """
    require(batch_size > 0, "Batch size must be positive")
    metadata, protocol, trials, windows = feature_plan(cache, run, smoke)
    require(checkpoint.is_file(), "Missing original pretrained LaBraM checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = sha256(checkpoint)
    torch.manual_seed(42)
    extractor = FrozenTokens(checkpoint, device)
    norm = extractor.encoder.backbone.fc_norm
    np.savez(
        output / "pooling_norm.npz",
        weight=norm.weight.detach().cpu().numpy(),
        bias=norm.bias.detach().cpu().numpy(),
        eps=np.asarray(norm.eps),
    )
    all_trials, all_windows, diagnostics, sources = [], [], [], {}
    cache_trials = pd.read_csv(cache / "trials.csv")
    for group, group_trials in trials.groupby("group", sort=True):
        print(f"Extracting pair {group}: {len(group_trials)} paired trials", flush=True)
        group_trials = group_trials.sort_values("movement_array_index").copy()
        group_windows = windows[windows.group == group].sort_values("eeg_array_index").copy()
        count = int(cache_trials[(cache_trials.group == group) & cache_trials.retained].shape[0])
        eeg_name, phase_name = f"{group}_eeg.npy", f"{group}_phase.npy"
        eeg = verified_array(cache, eeg_name, metadata, (count * 8, 2, 64, 800))
        phase = verified_array(cache, phase_name, metadata, (count, 2, 4320))
        sources.update({name: metadata["array_sha256"][name] for name in (eeg_name, phase_name)})
        movement = phase_features(np.array(phase[group_trials.movement_array_index.to_numpy()], copy=True))
        token_name, movement_name = f"{group}_tokens.npy", f"{group}_movement.npy"
        np.save(output / movement_name, movement, allow_pickle=False)
        diagnostic = encode_windows(
            extractor, eeg, group_windows.eeg_array_index.to_numpy(), batch_size, device, output / token_name
        )
        diagnostic.update(group=int(group), paired_trials=len(group_trials), movement_shape=list(movement.shape))
        diagnostics.append(diagnostic)
        group_trials["movement_feature_array"] = movement_name
        group_trials["movement_feature_index"] = np.arange(len(group_trials))
        index = dict(zip(group_trials.trial_id, group_trials.movement_feature_index, strict=True))
        group_windows["token_array"] = token_name
        group_windows["token_array_index"] = np.arange(len(group_windows))
        group_windows["movement_feature_array"] = movement_name
        group_windows["movement_feature_index"] = group_windows.trial_id.map(index)
        group_windows["movement_feature_start"] = group_windows.crop_start_s * 50
        group_windows["movement_feature_stop"] = (group_windows.crop_start_s + 4) * 50
        require((group_windows.movement_feature_stop <= 900).all(), "Movement window exceeds trial")
        all_trials.append(group_trials)
        all_windows.append(group_windows)
        del eeg, phase, movement
    pd.concat(all_trials, ignore_index=True).to_csv(output / "trials.csv", index=False)
    pd.concat(all_windows, ignore_index=True).to_csv(output / "windows.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "diagnostics.csv", index=False)
    files = sorted(output.iterdir())
    result = dict(
        format_version=1,
        operation="features-smoke" if smoke else "features",
        complete=True,
        cache_sha256=sha256(cache / "cache.json"),
        split_sha256=sha256(run / "split.json"),
        cache_source=str(cache.resolve()),
        split_source=str(run.resolve()),
        profile=protocol["profile"],
        checkpoint_source=str(checkpoint.resolve()),
        checkpoint_sha256=checkpoint_hash,
        initialization="pretrained",
        backbone_frozen=True,
        model_input_divisor=100,
        channels=CHANNELS,
        members=["a", "b"],
        token_axes=["paired_window", "member", "channel", "time_patch", "feature"],
        token_shape=[2, 64, 4, 200],
        token_source="final contextual tokens before fc_norm; CLS excluded",
        token_order="channel-major, time-minor",
        token_dtype="float32",
        movement_axes=["paired_trial", "member", "component", "sample"],
        movement_shape=[2, 2, 900],
        movement_components=["sin", "cos"],
        movement_sfreq=50,
        movement_resampling="scipy resample_poly 5/24; Kaiser beta=5; line padding; whole trial before slicing",
        movement_unit_circle_renormalization=False,
        centered=False,
        statistics_fitted=False,
        selected_splits=sorted(trials.split.unique()),
        paired_trials=len(trials),
        paired_windows=len(windows),
        participant_windows=2 * len(windows),
        reserved_dyad_arrays_accessed=False,
        test_signals_encoded=False,
        integrity_scope="whole permitted-dyad files hashed; only selected rows checked finite and encoded",
        source_array_sha256=sources,
        max_pooled_reconstruction_error=extractor.max_reconstruction_error,
        batch_size_paired_windows=batch_size,
        device=device,
        versions=dict(
            numpy=np.__version__,
            scipy=scipy.__version__,
            torch=torch.__version__,
            braindecode=version("braindecode"),
            mne=version("mne"),
            pandas=pd.__version__,
        ),
        artifact_sha256={p.name: sha256(p) for p in files},
        code_sha256={name: sha256(Path(__file__).with_name(name)) for name in ("features.py", "splits.py", "data.py")},
    )
    source_root = Path(__file__).resolve().parents[1]
    result["encoder_adapter_sha256"] = {
        name: sha256(source_root / name)
        for name in ("hyperscanning_contrastive/overfit.py", "labram_classification/model.py")
    }
    marker = "smoke.json" if smoke else "features.json"
    (output / marker).write_text(json.dumps(result, indent=2) + "\n")
    return result
