"""Regression tests for the random-assignment validation control."""

import numpy as np
import pytest
import torch

from hyperscanning_contrastive.correspondence import shuffle_metrics


def test_shuffle_calibrates_perfect_matches() -> None:
    """A perfect identity embedding returns to ordinary chance under uniform assignment."""
    features = torch.eye(8)
    rng = np.random.default_rng(1042)
    results = [shuffle_metrics(features, features, rng) for _ in range(1000)]
    assert np.mean([r["top1"] for r in results]) == pytest.approx(1 / 8, abs=0.015)
    assert np.mean([r["mrr"] for r in results]) == pytest.approx(sum(1 / i for i in range(1, 9)) / 8, abs=0.015)
    replay = np.random.default_rng(1042)
    assert shuffle_metrics(features, features, replay) == results[0]


def test_shuffle_collapse_stays_at_chance() -> None:
    """Randomization cannot change a constant similarity matrix into a successful result."""
    features = torch.ones(8, 4)
    result = shuffle_metrics(features, features, np.random.default_rng(0))
    assert result["top1"] == 1 / 8
    assert result["mrr"] == pytest.approx(sum(1 / i for i in range(1, 9)) / 8)
