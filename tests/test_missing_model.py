import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from models.missing_model import (
    AvailableGuidedFusion,
    CrossChannelFusion,
    CrossModalTransformation,
    MissingModalityRecognizer,
    consistency_loss,
    transformation_loss,
)
from models.backbones import build_encoder
from utils.checkpoint import load_encoder_from_checkpoint
from utils.evaluation import recognition_rate
from utils.head import ArcFace
import train_missing_model


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

    def test_shared_specific_heads_use_legacy_embedding(self):
        encoder = build_encoder("palm", input_channel=3, input_size=64, embedding_size=256, pretrained_path=None)
        self.assertEqual(encoder.shared_head[0].in_features, 256)
        self.assertEqual(encoder.specific_head[0].in_features, 256)

    def test_forward_parts_preserves_embedding_at_init(self):
        encoder = build_encoder("palm", input_channel=3, input_size=64, embedding_size=16, pretrained_path=None)
        encoder.eval()
        image = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            embedding = encoder(image)
            shared, specific = encoder.forward_parts(image)
        self.assertTrue(torch.allclose(torch.cat([shared, specific], dim=1), embedding, atol=1e-5))

    def test_checkpoint_loader_keeps_identity_part_heads(self):
        encoder = build_encoder("palm", input_channel=3, input_size=64, embedding_size=16, pretrained_path=None)
        state = encoder.state_dict()
        state["shared_head.0.weight"] = torch.zeros_like(state["shared_head.0.weight"])
        path = os.path.join(tempfile.gettempdir(), "pv_identity_head_test.pth")
        torch.save({"encoder": state}, path)
        loaded = load_encoder_from_checkpoint(path, "palm", 64, 16, "cpu")
        self.assertTrue(torch.allclose(loaded.shared_head[0].weight[:, :8], torch.eye(8)))

    def test_missing_model_freezes_encoders(self):
        model = MissingModalityRecognizer(
            build_encoder("palm", input_channel=3, input_size=64, embedding_size=16, pretrained_path=None),
            build_encoder("vein", input_channel=3, input_size=64, embedding_size=16, pretrained_path=None),
            num_classes=5,
            dim=16,
        )
        self.assertFalse(next(model.palm_encoder.backbone.parameters()).requires_grad)
        self.assertFalse(next(model.palm_encoder.shared_head.parameters()).requires_grad)
        self.assertFalse(next(model.vein_encoder.specific_head.parameters()).requires_grad)

    def test_missing_model_balances_teacher_and_fusion_logits_for_missing_case(self):
        palm_teacher = ArcFace(256, 5)
        vein_teacher = ArcFace(256, 5)
        model = MissingModalityRecognizer(
            TinyEncoder(),
            TinyEncoder(),
            num_classes=5,
            dim=256,
            palm_teacher=palm_teacher,
            vein_teacher=vein_teacher,
        )
        palm = torch.randn(2, 3, 8, 8)
        vein = torch.randn(2, 3, 8, 8)
        output = model(palm, vein, scenario="palmprint_missing")
        self.assertIn("teacher_logits", output)
        self.assertIn("fusion_logits", output)
        self.assertAlmostEqual(output["gate_alpha"].item(), 0.5)
        self.assertTrue(torch.allclose(output["logits"], 0.5 * (output["teacher_logits"] + output["fusion_logits"])))
        self.assertFalse(next(model.vein_teacher.parameters()).requires_grad)

    def test_available_guided_fusion_balances_available_and_restored_at_init(self):
        fusion = AvailableGuidedFusion(dim=256, reduction=4)
        available = torch.randn(2, 256)
        restored = torch.randn(2, 256)
        out = fusion(available, restored)
        expected = torch.nn.functional.normalize(available + restored, dim=1)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

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

    def test_transformation_loss_uses_cosine_scale(self):
        source = torch.randn(2, 128)
        self.assertLess(transformation_loss(source, source).item(), 1e-6)
        self.assertGreater(transformation_loss(source, -source).item(), 1.9)

    def test_optimizer_skips_frozen_encoder_params(self):
        model = MissingModalityRecognizer(TinyEncoder(), TinyEncoder(), num_classes=5, dim=256)
        args = SimpleNamespace(lr=1e-3, wd=1e-4)
        optimizer = train_missing_model.make_optimizer(model, args)
        optimized_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
        encoder_ids = {id(param) for param in model.palm_encoder.parameters()}
        self.assertFalse(optimized_ids & encoder_ids)

    def test_recognition_rate_uses_closed_set_predictions(self):
        logits = torch.tensor([[0.1, 0.9], [0.7, 0.3], [0.8, 0.2]])
        labels = torch.tensor([1, 0, 1])
        self.assertAlmostEqual(recognition_rate(logits, labels), 2 / 3)

    def test_missing_training_returns_anchor_and_available_losses(self):
        model = MissingModalityRecognizer(TinyEncoder(), TinyEncoder(), num_classes=5, dim=256)
        args = SimpleNamespace(
            lambda_shared=0.05,
            lambda_trans=0.1,
            lambda_anchor=1.0,
            lambda_avail=1.0,
            lambda_distill=1.0,
        )
        palm = torch.randn(2, 3, 8, 8)
        vein = torch.randn(2, 3, 8, 8)
        labels = torch.tensor([0, 1])
        losses = train_missing_model.batch_losses(model, palm, vein, labels, nn.CrossEntropyLoss(), args)
        self.assertEqual(len(losses), 8)
        self.assertTrue(torch.isfinite(losses[4]))
        self.assertTrue(torch.isfinite(losses[5]))
        self.assertTrue(torch.isfinite(losses[6]))


if __name__ == "__main__":
    unittest.main()
