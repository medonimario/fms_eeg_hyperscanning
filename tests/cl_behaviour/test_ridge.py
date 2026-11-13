"""Weighted ridge algebra, held-out isolation and fixed-gallery regression tests."""

import copy

import numpy as np
import pytest
import torch
from test_training import training_cache  # noqa: F401

from cl_behaviour.ridge import RidgeSystem, evaluate_ridge, make_systems, predict, run_ridge, training_examples
from cl_behaviour.tiny_fit import digest
from cl_behaviour.training import FeatureBank


@pytest.mark.parametrize("n,d", [(30, 5), (5, 30)])
def test_primal_dual_match_weighted_normal_equations(n, d):
    """Both representations reproduce an independently solved weighted ridge optimum."""
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(n, d)), rng.normal(size=(n, 7))
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()
    system = RidgeSystem(x, y, weights)
    query = rng.normal(size=(3, d))
    for value in (1e-5, 0.01, 0.1):
        expected = np.linalg.solve(x.T @ (weights[:, None] * x) + value * np.eye(d), x.T @ (weights[:, None] * y))
        np.testing.assert_allclose(predict(system.solve(value), query), query @ expected, rtol=1e-7, atol=1e-7)
    with pytest.raises(ValueError, match="positive"):
        system.solve(0)


def test_equal_person_weights_and_shared_model():
    """Duplicating one participant's samples must not increase their total fitting weight."""
    rng = np.random.default_rng(4)
    people = {
        "a": (rng.normal(size=(6, 3)), rng.normal(size=(6, 2))),
        "b": (rng.normal(size=(3, 3)), rng.normal(size=(3, 2))),
    }
    first = make_systems(people, "shared")["shared"].solve(0.01)
    people["a"] = tuple(np.repeat(v, 3, axis=0) for v in people["a"])
    second = make_systems(people, "shared")["shared"].solve(0.01)
    np.testing.assert_allclose(predict(first, np.eye(3)), predict(second, np.eye(3)), atol=1e-10)


def test_holdout_changes_cannot_change_fit(training_cache):
    """Fit only training rows and preserve coherent shuffled trial identities across offsets."""
    features, run = training_cache
    sets, _ = FeatureBank(features, run).build("temporal", 1084)
    for shuffled in (False, True):
        before = training_examples(sets, shuffled)
        changed = copy.deepcopy(sets)
        for g in changed:
            if g["split"] == "validation":
                g["eeg"].fill_(float("nan"))
                g["movement"].fill_(float("nan"))
        after = training_examples(changed, shuffled)
        for person in before:
            for a, b in zip(before[person], after[person], strict=True):
                np.testing.assert_array_equal(a, b)
    g = next(g for g in sets if g["split"] == "train")
    shuffled_y = training_examples(sets, True)[f"{g['group']}:{g['member']}"][1]
    np.testing.assert_array_equal(shuffled_y[: len(g["movement"])], g["movement"][g["mapping"]].numpy())


def test_zero_predictions_tie_at_chance():
    """No usable prediction must produce tie-aware chance rather than index-biased retrieval."""
    gallery = dict(
        gallery_id="x",
        group=2010,
        member="a",
        split="validation",
        offset=0,
        conditions=["closed1", "closed2", "closed1"],
        eeg=torch.eye(3),
        movement=torch.eye(3),
        mapping=torch.arange(3),
    )
    details, _ = evaluate_ridge({"shared": {"coefficients": np.zeros((3, 3))}}, [gallery], False)
    assert details.iloc[0].top1 == pytest.approx(1 / 3)
    assert details.iloc[0].zero_prediction_fraction == 1


def test_ridge_integration_and_global_selection(training_cache, tmp_path):
    """Write both families/arms with a single selected lambda per family and no test data."""
    features, run = training_cache
    output = tmp_path / "ridge"
    result = run_ridge(features, run, output, lambdas=(0.001, 0.01))
    assert result["complete"] and len(result["results"]) == 4
    assert not result["test_features_accessed"]
    assert (output / "regularization_curves.png").is_file()
    assert not list(output.glob("*.pdf"))
    import pandas as pd

    history = pd.read_csv(output / "regularization.csv")
    for selected in result["results"]:
        rows = history[(history.family == selected["family"]) & (history.arm == selected["arm"])]
        winner = rows.sort_values(["validation_mrr", "regularization"], ascending=False).iloc[0]
        assert winner.regularization == selected["regularization"]
    for name, checksum in result["output_sha256"].items():
        assert digest(output / name) == checksum
