"""Check binary training, held-out validation, and best-checkpoint restoration."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from skorch.callbacks import Callback
from torch import nn
from torch.utils.data import TensorDataset

from labram_classification_lessons.train_preview import make_classifier


class RecordWeights(Callback):
    """Keep epoch-end states so checkpoint selection can be checked independently."""

    def initialize(self) -> "RecordWeights":
        """Clear states at the start of a training run."""
        self.states = []
        return self

    def on_epoch_end(self, net, **kwargs) -> None:
        """Capture weights before training-end checkpoint restoration."""
        self.states.append({name: value.clone() for name, value in net.module_.state_dict().items()})


class TrainingPreviewTests(unittest.TestCase):
    """Verify trainer behavior using a small model without a pretrained checkpoint."""

    def test_binary_fit_uses_explicit_validation_and_restores_best(self) -> None:
        """Optimize with integer labels and restore the earliest best-scoring epoch."""
        torch.manual_seed(42)
        training = TensorDataset(torch.tensor([[-2.0], [-1.0], [1.0], [2.0]]), torch.tensor([0, 0, 1, 1]))
        validation = TensorDataset(torch.tensor([[-3.0], [3.0]]), torch.tensor([0, 1]))
        model = nn.Sequential(nn.Linear(1, 1), nn.Flatten(start_dim=0))
        initial = model[0].weight.detach().clone()
        recorder = RecordWeights()
        with tempfile.TemporaryDirectory() as directory:
            net = make_classifier(model, validation, Path(directory), epochs=2, batch_size=3, lr=0.01)
            net.set_params(verbose=0, callbacks=net.callbacks + [("record_weights", recorder)])
            net.initialize()
            self.assertIs(net.module_, model)
            train_split, valid_split = net.get_split_datasets(training, y=None)
            self.assertIs(train_split, training)
            self.assertIs(valid_split, validation)
            net.fit(training, y=None)
            self.assertEqual(len(net.history), 2)
            self.assertEqual(len(net.history[0]["batches"]), 3)
            self.assertFalse(torch.equal(initial, recorder.states[0]["0.weight"]))
            best = max(range(2), key=lambda epoch: net.history[epoch]["valid_balanced_accuracy"])
            saved = torch.load(Path(directory) / "best_model.pt", weights_only=True)
            for name, value in net.module_.state_dict().items():
                torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
                torch.testing.assert_close(value, recorder.states[best][name], rtol=0, atol=0)
            probabilities = net.predict_proba(validation)
            self.assertEqual(probabilities.shape, (2, 2))
            self.assertTrue(np.isfinite(probabilities).all())
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
