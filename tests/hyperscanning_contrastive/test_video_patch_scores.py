"""Matching-patch scoring, norm effects and tie handling."""

import numpy as np

from hyperscanning_contrastive.video_baseline import score_similarities
from hyperscanning_contrastive.video_patch_scores import patch_scores, winner_counts


def test_matching_patch_scores():
    """Equal patch norms equate mean and flattened cosine; max/top8 use matching positions."""
    rng = np.random.default_rng(4)
    a = rng.normal(size=(4, 256, 200))
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    scores, patches = patch_scores(a, a)
    np.testing.assert_allclose(scores["flattened"], scores["patch_mean"], atol=1e-12)
    for matrix in scores.values():
        assert score_similarities(matrix)["top1"] == 1
    np.testing.assert_allclose(scores["patch_top8"], np.sort(patches, axis=-1)[..., -8:].mean(-1))
    scaled = a.copy()
    scaled[:, 0] *= 100
    changed, _ = patch_scores(scaled, a)
    np.testing.assert_allclose(changed["patch_mean"], scores["patch_mean"], atol=1e-12)
    assert not np.allclose(changed["flattened"], scores["flattened"])


def test_ties_and_zero_vectors():
    """Zero vectors remain finite and all ties allocate retrieval and patch mass fairly."""
    scores, patches = patch_scores(np.zeros((4, 256, 200)), np.zeros((4, 256, 200)))
    for matrix in scores.values():
        result = score_similarities(matrix)
        assert result["top1"] == 0.25
        np.testing.assert_allclose(result["mrr"], 25 / 48)
    counts = winner_counts(patches)
    np.testing.assert_allclose(counts["aligned"], 4 / 256)
    np.testing.assert_allclose(counts["mismatched"], 12 / 256)
    np.testing.assert_allclose(counts["retrieved_correct"].sum(), 2)
    np.testing.assert_allclose(counts["retrieved_incorrect"].sum(), 6)
