"""Verify held-out refits, candidate rankings, tie-aware chance and dyad aggregation."""

import numpy as np
import pandas as pd
import pytest

from mTRF import eeg, retrieval


def test_held_out_retrieval_reproduces_original_folds(monkeypatch) -> None:
    """Recover strongly paired synthetic trials using the real fork and reject a mismatched reference."""
    monkeypatch.setattr(eeg, "LAGS", (0, 0))
    rng = np.random.default_rng(8)
    features = rng.normal(size=(19, 160, 5))
    responses = features @ rng.normal(size=(5, 3)) + rng.normal(scale=0.01, size=(19, 160, 3))
    keep = [i for i in range(20) if i != 7]
    reference = eeg.nested_crossval(
        eeg.TRF(direction=1, preload=False),
        list(features),
        list(responses),
        eeg.FS,
        *eeg.LAGS,
        np.array([0.1, 1.0]),
        k=5,
        seed=123,
        n_jobs=1,
        verbose=False,
    )
    seen = []
    for f in range(5):
        result = retrieval.evaluate_fold(features, responses, keep, reference, f)
        assert not set(result["train_trials"]) & set(result["test_trials"])
        assert set(result["train_trials"]) | set(result["test_trials"]) == set(keep)
        np.testing.assert_allclose(result["refit_scores"], result["reference_scores"])
        for q, scores in enumerate(result["scores"]):
            metrics = retrieval.query_metrics(scores, q)
            assert metrics["top1"] == metrics["mrr"] == 1
            assert metrics["chance_top1"] == 1 / len(scores)
        seen.extend(result["test_trials"])
    assert sorted(seen) == keep
    broken = (reference[0] + 1, reference[1], reference[2])
    with pytest.raises(ValueError, match="does not reproduce"):
        retrieval.evaluate_fold(features, responses, keep, broken, 0)


def test_scores_and_ties() -> None:
    """Score every candidate against the query EEG and give identical predictions exact chance performance."""
    query = np.array([[[0.0], [1.0], [3.0]], [[3.0], [0.0], [1.0]], [[1.0], [3.0], [0.0]]])
    scores = retrieval.retrieval_scores(query, query)
    np.testing.assert_array_equal(np.diag(scores), 1)
    assert np.all(scores[~np.eye(3, dtype=bool)] < 1)
    tied = retrieval.retrieval_scores(query, np.repeat(query[:1], 3, axis=0))
    for i in range(3):
        metrics = retrieval.query_metrics(tied[i], i)
        assert metrics["top1"] == metrics["chance_top1"]
        assert metrics["mrr"] == metrics["chance_mrr"]
    with pytest.raises(ValueError, match="nonconstant"):
        retrieval.retrieval_scores(np.ones_like(query), query)


def test_collect_variable_galleries_and_exclusion(tmp_path, monkeypatch) -> None:
    """Average per-query chance consistently and retain pair 2020 only in the detailed output."""
    monkeypatch.setattr(retrieval, "PAIRS", (2010, 2020))
    for pair in retrieval.PAIRS:
        folds = []
        for c in eeg.CONDITIONS:
            for member in "ab":
                for f, test in enumerate(np.array_split(np.arange(19), 5), start=1):
                    scores = np.eye(len(test))
                    if pair == 2020:
                        scores = -scores
                    folds.append(
                        {"condition": c, "member": member, "fold": f, "test_trials": test.tolist(), "scores": scores}
                    )
        eeg.save_pickle(tmp_path / f"pair{pair}.pickle", {"pair": pair, "protocol": {"version": 1}, "folds": folds})
    retrieval.collect(tmp_path)
    assert (tmp_path / "retrieval.png").is_file()
    summary = pd.read_csv(tmp_path / "summary.csv")
    np.testing.assert_allclose(summary.top1, 1)
    np.testing.assert_allclose(summary.chance_top1, 5 / 19)
    assert (summary.dyads == 1).all()
    assert set(pd.read_csv(tmp_path / "queries.csv").pair) == {2010, 2020}
    assert set(pd.read_csv(tmp_path / "participants.csv").pair) == {2010}
    eeg.save_pickle(tmp_path / "pair2020.pickle", {"pair": 2020, "protocol": {"version": 2}, "folds": []})
    with pytest.raises(ValueError, match="Incompatible"):
        retrieval.collect(tmp_path)
