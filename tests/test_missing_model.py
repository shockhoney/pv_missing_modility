import unittest

import torch
import torch.nn as nn

from models.missing_model import (
    AvailableGuidedFusion,
    CrossChannelFusion,
    CrossModalTransformation,
    MissingModalityRecognizer,
    consistency_loss,
    shared_specific_loss,
    transformation_loss,
)
from models.backbones import build_encoder
from utils.evaluation import recognition_rate


class TinyEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, dim))

    def forward(self, x):
        return self.net(x)

    def forward_parts(self, x):
        feat = self.forward(x)
        return feat[:, :128], feat[:, 128:]


class MissingModelTest(unittest.TestCase):
    def test_cross_modal_transformation_restores_embedding_shape(self):
        cmft = CrossModalTransformation(dim=128, hidden=512)
        out = cmft(torch.randn(2, 128))
        self.assertEqual(tuple(out.shape), (2, 128))

    def test_resnet18_encoder_returns_shared_specific_parts(self):
        encoder = build_encoder("palm", input_channel=3, input_size=64, embedding_size=256, pretrained_path=None)
        encoder.eval()
        with torch.no_grad():
            shared, specific = encoder.forward_parts(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(shared.shape), (2, 128))
        self.assertEqual(tuple(specific.shape), (2, 128))

    def test_available_guided_fusion_preserves_available_feature_at_init(self):
        fusion = AvailableGuidedFusion(dim=256, reduction=4)
        available = torch.randn(2, 256)
        restored = torch.randn(2, 256)
        out = fusion(available, restored)
        self.assertTrue(torch.allclose(out, torch.nn.functional.normalize(available, dim=1), atol=1e-6))

    def test_cross_channel_fusion_outputs_embedding(self):
        fusion = CrossChannelFusion(dim=256, heads=4, reduction=4)
        out = fusion(torch.randn(2, 256), torch.randn(2, 256), scenario="complete")
        self.assertEqual(tuple(out.shape), (2, 256))
        out = fusion(torch.randn(2, 256), torch.randn(2, 256), scenario="palmprint_missing")
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
            self.assertEqual(tuple(output["palm_shared"].shape), (2, 128))
            self.assertEqual(tuple(output["palm_specific"].shape), (2, 128))
            self.assertEqual(tuple(output["hat_palm_specific"].shape), (2, 128))
            self.assertEqual(tuple(output["hat_vein_specific"].shape), (2, 128))

    def test_mask_selects_matching_missing_scenario(self):
        model = MissingModalityRecognizer(TinyEncoder(), TinyEncoder(), num_classes=5, dim=256)
        self.assertEqual(model._scenario_from_mask(torch.tensor([[0, 1], [0, 1]]), "complete"), "palmprint_missing")
        self.assertEqual(model._scenario_from_mask(torch.tensor([[1, 0], [1, 0]]), "complete"), "palmvein_missing")
        self.assertEqual(model._scenario_from_mask(torch.tensor([[1, 1], [1, 1]]), "complete"), "complete")

    def test_losses_are_finite_scalars(self):
        source = torch.randn(2, 256)
        target = torch.randn(2, 256)
        self.assertEqual(transformation_loss(source, target).dim(), 0)
        self.assertTrue(torch.isfinite(transformation_loss(source, target)))
        self.assertEqual(consistency_loss(source, target).dim(), 0)
        self.assertTrue(torch.isfinite(consistency_loss(source, target)))
        self.assertEqual(shared_specific_loss(source[:, :128], target[:, :128], source[:, 128:], target[:, 128:]).dim(), 0)

    def test_recognition_rate_uses_closed_set_predictions(self):
        logits = torch.tensor([[0.1, 0.9], [0.7, 0.3], [0.8, 0.2]])
        labels = torch.tensor([1, 0, 1])
        self.assertAlmostEqual(recognition_rate(logits, labels), 2 / 3)


if __name__ == "__main__":
    unittest.main()
