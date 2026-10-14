"""Offline integration checks for the local pretrained LaBraM adaptation."""

import unittest

import torch

from labram_classification_lessons.check_labram import DEFAULT_CHECKPOINT, BinaryLabram, load_model, read_student_weights
from labram_classification_lessons.inspect_mirrorgame import STANDARD_CHANNELS




@unittest.skipUnless(DEFAULT_CHECKPOINT.is_file(), "Place labram-base.pth in models/ to test local weights.")
class PretrainedLabramTests(unittest.TestCase):
    """Ensure adaptations preserve pretrained tensors and binary output semantics."""

    def test_pretrained_weights_and_single_window(self) -> None:
        """Only temporal-table slicing, norm renaming, and the new head may differ."""
        model = load_model(STANDARD_CHANNELS)
        saved = read_student_weights(DEFAULT_CHECKPOINT)
        for name, value in model.backbone.state_dict().items():
            if name.startswith("final_layer."):
                continue
            source_name = name.replace("fc_norm.", "norm.")
            source_name = source_name.replace("patch_embed.temporal_conv.", "patch_embed.")
            source_name = source_name.replace(".mlp.0.", ".mlp.fc1.").replace(".mlp.2.", ".mlp.fc2.")
            source_name = {"position_embedding": "pos_embed", "temporal_embedding": "time_embed"}.get(
                source_name, source_name
            )
            expected = saved[source_name]
            if name == "temporal_embedding":
                expected = expected[:, :5]
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        self.assertEqual(tuple(saved["time_embed"].shape), (1, 16, 200))
        self.assertEqual(tuple(model.backbone.temporal_embedding.shape), (1, 5, 200))
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(1, 64, 800))
        self.assertEqual(tuple(logits.shape), (1,))
        self.assertTrue(torch.isfinite(logits).all())
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 64, 801))
        with self.assertRaises(ValueError):
            BinaryLabram(model.backbone, ["Fp1", "FP1"])
        with self.assertRaises(ValueError):
            BinaryLabram(model.backbone, ["unknown-electrode"])


if __name__ == "__main__":
    unittest.main()
