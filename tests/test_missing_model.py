import unittest

import torch
import torch.nn as nn

from models.missing_model import (
    CrossChannelFusion,
    CrossModalTransformation,
    MissingModalityRecognizer,
    SharedSpecificProjector,
    consistency_loss,
    transformation_loss,
)
from utils.evaluation import recognition_rate


class TinyEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, dim))

    def forward(self, x):
        return self.net(x)


class MissingModelTest(unittest.TestCase):
    def test_shared_specific_projector_splits_embedding(self):
        projector = SharedSpecificProjector(dim=256)
        shared, specific = projector(torch.randn(2, 256))
        self.assertEqual(tuple(shared.shape), (2, 128))
        self.assertEqual(tuple(specific.shape), (2, 128))

    def test_cross_modal_transformation_keeps_specific_shape(self):
        cmft = CrossModalTransformation(dim=128, hidden=512)
        out = cmft(torch.randn(2, 128))
        self.assertEqual(tuple(out.shape), (2, 128))

    def test_cross_channel_fusion_outputs_embedding(self):
        fusion = CrossChannelFusion(dim=256, heads=4, reduction=4)
        out = fusion(torch.randn(2, 256), torch.randn(2, 256))
        self.assertEqual(tuple(out.shape), (2, 256))

    def test_recognizer_supports_all_training_scenarios(self):
        model = MissingModalityRecognizer(TinyEncoder(), TinyEncoder(), num_classes=5, dim=256)
        palm = torch.randn(2, 3, 8, 8)
        vein = torch.randn(2, 3, 8, 8)
        labels = torch.tensor([0, 1])
        for scenario in ("complete", "palmprint_missing", "palmvein_missing"):
            output = model(palm, vein, labels=labels, scenario=scenario)
            self.assertEqual(tuple(output["logits"].shape), (2, 5))
            self.assertEqual(tuple(output["z"].shape), (2, 256))
            self.assertEqual(tuple(output["hat_palm_specific"].shape), (2, 128))
            self.assertEqual(tuple(output["hat_vein_specific"].shape), (2, 128))

    def test_losses_are_finite_scalars(self):
        source = torch.randn(2, 256)
        target = torch.randn(2, 256)
        self.assertEqual(transformation_loss(source, target).dim(), 0)
        self.assertTrue(torch.isfinite(transformation_loss(source, target)))
        self.assertEqual(consistency_loss(source, target).dim(), 0)
        self.assertTrue(torch.isfinite(consistency_loss(source, target)))

    def test_recognition_rate_uses_closed_set_predictions(self):
        logits = torch.tensor([[0.1, 0.9], [0.7, 0.3], [0.8, 0.2]])
        labels = torch.tensor([1, 0, 1])
        self.assertAlmostEqual(recognition_rate(logits, labels), 2 / 3)


if __name__ == "__main__":
    unittest.main()
