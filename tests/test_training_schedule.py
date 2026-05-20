import unittest
import warnings

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import test_encoder
import train_encoder
from models.backbones import build_encoder
from utils.preprocess import CLAHE, build_palm_transform, build_vein_transform


class ScheduleAndFusionTest(unittest.TestCase):
    def test_modality_transforms_are_separate(self):
        palm_ops = [type(op) for op in build_palm_transform(224, train=True).transforms]
        vein_ops = [type(op) for op in build_vein_transform(224, train=True).transforms]
        self.assertNotIn(transforms.RandomHorizontalFlip, palm_ops)
        self.assertIn(transforms.RandomAffine, palm_ops)
        self.assertIn(transforms.ColorJitter, palm_ops)
        self.assertIn(CLAHE, vein_ops)
        self.assertIn(transforms.RandomAffine, vein_ops)
        self.assertNotIn(transforms.ColorJitter, vein_ops)

    def test_modality_transforms_keep_three_channels(self):
        image = Image.fromarray(np.full((16, 16), 128, dtype=np.uint8))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertEqual(build_palm_transform(32)(image).shape[0], 3)
            self.assertEqual(build_vein_transform(32)(image).shape[0], 3)

    def test_default_args_train_palm_baseline_only(self):
        args = train_encoder.parse_args([])
        self.assertEqual(args.modality, "palm")
        self.assertEqual(train_encoder.parse_args(["--modality", "vein"]).modality, "vein")

    def test_weighted_fusion_is_normalized(self):
        palm = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        vein = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        fused = test_encoder.weighted_fusion(palm, vein, 0.75)
        np.testing.assert_allclose(np.linalg.norm(fused, axis=1), np.ones(2), rtol=1e-6)

    def test_resnet50_palm_encoder_output_shape(self):
        encoder = build_encoder("palm", input_channel=3, input_size=64, embedding_size=16, pretrained_path=None)
        encoder.eval()
        with torch.no_grad():
            out = encoder(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(out.shape), (2, 16))

    def test_convnextv2_tiny_vein_encoder_output_shape(self):
        encoder = build_encoder("vein", input_channel=3, input_size=64, embedding_size=16, pretrained_path=None)
        self.assertEqual(encoder.__class__.__name__, "ConvNeXtV2TinyVeinEncoder")
        encoder.eval()
        with torch.no_grad():
            out = encoder(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(out.shape), (2, 16))


if __name__ == "__main__":
    unittest.main()
