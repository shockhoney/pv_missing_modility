import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import test_encoder
import train_encoder
from utils.augmentations import UAAAffineAugmenter


class TinyEncoder(nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3))


class TinyHead(nn.Module):
    def forward(self, feat, labels=None):
        return torch.stack([feat[:, 0], -feat[:, 0]], dim=1)


class ScheduleAndFusionTest(unittest.TestCase):
    def test_epoch_settings(self):
        args = SimpleNamespace(
            use_uaa=True,
            use_starmix=True,
            starmix_start_epoch=51,
            uaa_start_epoch=151,
            align_start_epoch=51,
            lambda_align=0.0,
            align_final=0.03,
            arcface_s=16.0,
            arcface_m=0.25,
            arcface_s_final=24.0,
            arcface_m_final=0.35,
        )
        self.assertEqual(train_encoder.epoch_settings(args, 1)["lambda_align"], 0.0)
        self.assertFalse(train_encoder.epoch_settings(args, 1)["use_starmix"])
        self.assertFalse(train_encoder.epoch_settings(args, 1)["use_uaa"])

        mid = train_encoder.epoch_settings(args, 51)
        self.assertTrue(mid["use_starmix"])
        self.assertFalse(mid["use_uaa"])
        self.assertEqual(mid["lambda_align"], 0.03)
        self.assertEqual(mid["arcface_s"], 24.0)

        late = train_encoder.epoch_settings(args, 151)
        self.assertTrue(late["use_starmix"])
        self.assertTrue(late["use_uaa"])

    def test_apply_uaa_keeps_full_batch(self):
        args = SimpleNamespace(
            use_uaa=True,
            uaa_steps=1,
            uaa_step_size=0.01,
            uaa_beta=0.5,
            uaa_gamma=0.5,
        )
        images = torch.randn(4, 2, 8, 8)
        labels = torch.tensor([0, 1, 0, 1])
        out, out_labels, params = train_encoder.apply_uaa(
            images, labels, TinyEncoder(), TinyHead(), UAAAffineAugmenter(), args, None, enabled=True
        )
        self.assertEqual(out.shape, images.shape)
        self.assertTrue(torch.equal(out_labels, labels))
        self.assertEqual(params.shape[-1], 4)

    def test_weighted_fusion_is_normalized(self):
        palm = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        vein = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        fused = test_encoder.weighted_fusion(palm, vein, 0.75)
        np.testing.assert_allclose(np.linalg.norm(fused, axis=1), np.ones(2), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
