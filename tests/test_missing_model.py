import unittest

import torch
import torch.nn as nn

from models.missing_model import (
    CrossChannelFusion,
    FeatureRestorer,
    MissingModalityRecognizer,
    consistency_loss,
    recovery_loss,
)
from utils.evaluation import recognition_rate


class TinyEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, dim))

    def forward(self, x):
        return self.net(x)


class MissingModelTest(unittest.TestCase):
    def test_feature_restorer_keeps_embedding_shape(self):
        restorer = FeatureRestorer(dim=256, hidden=512)
        out = restorer(torch.randn(2, 256))
        self.assertEqual(tuple(out.shape), (2, 256))

    def test_cross_channel_fusion_outputs_embedding(self):
        fusion = CrossChannelFusion(dim=256, heads=4, reduction=4)
        out = fusion(torch.randn(2, 256), torch.randn(2, 256))
        self.assertEqual(tuple(out.shape), (2, 256))

    def test_recognizer_supports_all_training_scenarios(self):
        model = MissingModalityRecognizer(TinyEncoder(), TinyEncoder(), num_classes=5, dim=256)
        palm = torch.randn(2, 3, 8, 8)
        vein = torch.randn(2, 3, 8, 8)
        labels = torch.tensor([0, 1])
        for scenario in ("full", "palm_only", "vein_only"):
            output = model(palm, vein, labels=labels, scenario=scenario)
            self.assertEqual(tuple(output["logits"].shape), (2, 5))
            self.assertEqual(tuple(output["z"].shape), (2, 256))

    def test_losses_are_finite_scalars(self):
        source = torch.randn(2, 256)
        target = torch.randn(2, 256)
        self.assertEqual(recovery_loss(source, target).dim(), 0)
        self.assertTrue(torch.isfinite(recovery_loss(source, target)))
        self.assertEqual(consistency_loss(source, target).dim(), 0)
        self.assertTrue(torch.isfinite(consistency_loss(source, target)))

    def test_recognition_rate_uses_closed_set_predictions(self):
        logits = torch.tensor([[0.1, 0.9], [0.7, 0.3], [0.8, 0.2]])
        labels = torch.tensor([1, 0, 1])
        self.assertAlmostEqual(recognition_rate(logits, labels), 2 / 3)


if __name__ == "__main__":
    unittest.main()
