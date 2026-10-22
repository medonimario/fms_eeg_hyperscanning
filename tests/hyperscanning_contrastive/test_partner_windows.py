"""All-window partner split, candidate, coverage and optimization regression checks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from test_ssl import TinyEncoder
from test_temporal import temporal_cache

from hyperscanning_contrastive import partner_windows as pw
from hyperscanning_contrastive.data import sha256
from hyperscanning_contrastive.ssl import EEGCache, SSLConfig


def window_cache(path: Path) -> Path:
    """Build multiple intervals per trial plus one rare shifted interval."""
    temporal_cache(path)
    rows = pd.read_csv(path / "windows.csv")
    rare = (rows.fold == 0) & (rows.condition == "open") & (rows.condition_trial == 0) & (rows.crop_start_s == 4)
    rows.loc[rare, "crop_start_s"] = 6
    rows["crop_end_s"] = rows.crop_start_s + 4
    rows.to_csv(path / "windows.csv", index=False)
    trials = rows[["group", "condition", "trial_id", "fold", "condition_trial"]].drop_duplicates()
    trials.to_csv(path / "trials.csv", index=False)
    metadata = json.loads((path / "cache.json").read_text())
    for name in ("windows.csv", "trials.csv"):
        metadata["manifest_sha256"][name] = sha256(path / name)
    (path / "cache.json").write_text(json.dumps(metadata))
    return path


def test_trial_split_and_intervals(tmp_path: Path) -> None:
    """A shifted window stays with its trial; singleton and reserved windows are accounted for."""
    cache = EEGCache(window_cache(tmp_path / "cache"), SSLConfig(pairs=2), all_windows=True)
    trials, rows, census = pw.plan_windows(cache, 2, 2042)
    assert rows.groupby(pw.TRIAL_KEYS).split.nunique().max() == 1
    assert not rows[rows.used].fold.isin([1, 2]).any()
    assert (census.candidates.sum()) == len(rows)
    assert not rows[rows.crop_start_s == 6].used.any()
    assert set(rows[rows.used].crop_start_s) == {0, 4}
    assert (trials[trials.split == "validation"].groupby(["group", "condition"]).size() == 2).all()
    assert rows[rows.used].groupby(pw.SET_KEYS).trial_id.nunique().min() >= 2
    repeated = pw.plan_windows(cache, 2, 2042)[0]
    pd.testing.assert_frame_equal(trials, repeated)


def test_variable_chance_and_weighting() -> None:
    """Different gallery sizes have their own chance baseline and intervals get equal weight."""
    bank = {}
    for offset, size in ((0, 2), (4, 3)):
        bank[(0, "open", offset)] = {"a": torch.eye(size), "b": torch.eye(size)}
    scores = pw.score_windows(torch.nn.Identity(), bank, 0.05)
    assert scores.mrr.tolist() == [1, 1]
    assert scores.candidates.tolist() == [2, 3]
    assert scores.chance_top1.tolist() == [0.5, 1 / 3]
    assert pw.summarize(scores)["macro"]["chance_top1"] == pytest.approx((0.5 + 1 / 3) / 2)
    assert scores.queries.tolist() == [4, 6]


def test_partner_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training forwards both partners at all usable intervals and updates only the shared head."""
    path = window_cache(tmp_path / "cache")
    monkeypatch.setattr(pw, "ContrastiveLabram", TinyEncoder)
    original = EEGCache.signals
    seen = []

    def signals(self: EEGCache, rows: pd.DataFrame) -> torch.Tensor:
        """Audit encoded windows without allowing reserved EEG through."""
        assert set(rows.fold) == {0}
        assert set(rows.condition) == {"open"}
        seen.extend((r.trial_id, r.crop_start_s, r.member) for r in rows.itertuples())
        return original(self, rows)

    monkeypatch.setattr(EEGCache, "signals", signals)
    out = tmp_path / "run"
    report = pw.train_partner_windows(
        path, out, tmp_path / "absent", initialization="random", validation_trials=2, epochs=1
    )
    windows = pd.read_csv(out / "windows.csv")
    used = windows[windows.used]
    assert len(seen) == 2 * len(used)
    assert len(set(seen)) == len(seen)
    assert report["conditions"] == ["open"]
    assert report["coverage"]["train"]["paired_trials"] == 2
    assert report["excluded_condition_windows"] == int((windows.condition != "open").sum())
    assert set(report["baselines"]["initial"]["conditions"]) == {"open"}
    assert report["checkpoint_sha256"] is None
    assert report["training_paired_windows_per_epoch"] == len(used[used.split == "train"])
    assert report["updates_per_epoch"] == int(np.ceil(report["training_sets_per_epoch"] / 4))
    mapping = pd.read_csv(out / "shuffled_training_pairs.csv")
    for row in mapping.itertuples():
        eligible = used[
            (used.split == "train")
            & (used.group == row.group)
            & (used.condition == row.condition)
            & (used.crop_start_s == row.crop_start_s)
        ]
        assert row.source_trial in set(eligible.trial_id)
        assert row.target_trial in set(eligible.trial_id)
    first = torch.load(out / "initial_projection.pt", weights_only=True)
    for arm in ("true", "shuffled"):
        history = pd.read_csv(out / arm / "history.csv")
        assert report["arms"][arm]["best_epoch"] == history.mrr.idxmax()
        assert history.iloc[0].mrr == pytest.approx(report["baselines"]["initial"]["macro"]["mrr"])
        last = torch.load(out / arm / "last_projection.pt", weights_only=True)
        assert torch.equal(first["0.mean"], last["0.mean"])
        assert torch.equal(first["0.scale"], last["0.scale"])
        assert not torch.equal(first["1.weight"], last["1.weight"])
    assert (out / "partner_windows_curves.png").stat().st_size > 1000


def test_condition_filter_preserves_splits(tmp_path: Path) -> None:
    """Filtering supervision preserves the original whole-trial split and open galleries."""
    cache = EEGCache(window_cache(tmp_path / "cache"), SSLConfig(pairs=2), all_windows=True)
    all_trials, all_rows, _ = pw.plan_windows(cache, 2, 2042, ("closed1", "open", "closed2"))
    trials, rows, census = pw.plan_windows(cache, 2, 2042)
    pd.testing.assert_frame_equal(
        trials.drop(columns="selected_condition"), all_trials.drop(columns="selected_condition")
    )
    pd.testing.assert_frame_equal(
        rows[rows.condition == "open"].reset_index(drop=True),
        all_rows[all_rows.condition == "open"].reset_index(drop=True),
    )
    assert set(rows[rows.used].condition) == {"open"}
    assert not census[census.condition != "open"].used.any()
    assert set(rows[rows.condition != "open"].reason) == {"condition_not_selected"}
    for invalid in ((), ("open", "open"), ("missing",)):
        with pytest.raises(ValueError, match="Conditions"):
            pw.plan_windows(cache, 2, 2042, invalid)
