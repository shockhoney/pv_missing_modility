import contextlib
import io
import unittest
import warnings

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import test_encoder
import train_encoder
import train_missing_model
from utils import datasets_txt
from models.backbones import build_encoder
from utils.preprocess import CLAHE, VeinIntensityJitter, build_palm_transform, build_vein_transform


class ScheduleAndFusionTest(unittest.TestCase):
    def test_modality_transforms_are_separate(self):
        palm_ops = [type(op) for op in build_palm_transform(224, train=True).transforms]
        vein_ops = [type(op) for op in build_vein_transform(224, train=True).transforms]
        self.assertNotIn(transforms.RandomHorizontalFlip, palm_ops)
        self.assertIn(transforms.RandomAffine, palm_ops)
        self.assertIn(transforms.ColorJitter, palm_ops)
        self.assertIn(CLAHE, vein_ops)
        self.assertIn(VeinIntensityJitter, vein_ops)
        self.assertIn(transforms.RandomAffine, vein_ops)
        self.assertNotIn(transforms.ColorJitter, vein_ops)
        vein_affine = next(op for op in build_vein_transform(224, train=True).transforms if isinstance(op, transforms.RandomAffine))
        self.assertEqual(vein_affine.degrees, [-5.0, 5.0])
        self.assertEqual(vein_affine.translate, (0.03, 0.03))
        self.assertEqual(vein_affine.scale, (0.95, 1.05))
        self.assertNotIn(VeinIntensityJitter, [type(op) for op in build_vein_transform(224, train=False).transforms])

    def test_modality_transforms_keep_three_channels(self):
        image = Image.fromarray(np.full((16, 16), 128, dtype=np.uint8))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertEqual(build_palm_transform(32)(image).shape[0], 3)
            self.assertEqual(build_vein_transform(32)(image).shape[0], 3)

    def test_default_args_train_palm_baseline_only(self):
        args = train_encoder.parse_args([])
        self.assertEqual(args.modality, "palm")
        self.assertEqual(args.train_list, "data_txt/cumt/ssfd_train_full.txt")
        self.assertEqual(args.palm_pretrained, "pretrained/resnet18_imagenet1k_v1.pth")
        self.assertEqual(args.vein_pretrained, "pretrained/resnet18_imagenet1k_v1.pth")
        self.assertEqual(train_encoder.parse_args(["--modality", "vein"]).modality, "vein")

    def test_vein_training_uses_vein_defaults(self):
        args = train_encoder.parse_args(["--modality", "vein"])
        self.assertEqual(args.epochs, 300)
        self.assertEqual(args.lr, 3e-3)
        self.assertEqual(args.wd, 5e-4)
        self.assertEqual(args.arcface_m, 0.15)
        self.assertEqual(args.warmup_epochs, 5)
        self.assertEqual(args.backbone_lr, 3e-4)
        self.assertEqual(args.label_smoothing, 0.05)

        args = train_encoder.parse_args(["--modality", "vein", "--lr", "1e-3"])
        self.assertEqual(args.lr, 1e-3)
        args = train_encoder.parse_args(["--modality", "vein", "--backbone_lr", "1e-4"])
        self.assertEqual(args.backbone_lr, 1e-4)

    def test_vein_optimizer_uses_lower_backbone_lr(self):
        args = train_encoder.parse_args(["--modality", "vein"])
        encoder = train_encoder.make_encoder(args)
        head = train_encoder.ArcFace(args.embedding_size, 2)
        optimizer = train_encoder.make_optimizer(encoder, head, args)
        self.assertEqual(optimizer.param_groups[0]["lr"], args.backbone_lr)
        self.assertEqual(optimizer.param_groups[1]["lr"], args.lr)

    def test_test_encoder_defaults_to_single_modality(self):
        args = test_encoder.parse_args([])
        self.assertEqual(args.modality, "palm")
        self.assertEqual(args.protocol_list, "data_txt/cumt/ssfd_test_protocol.txt")
        self.assertEqual(args.ckpt, "outputs/encoders/palm_best.pth")

        args = test_encoder.parse_args(["--modality", "vein"])
        self.assertEqual(args.ckpt, "outputs/encoders/vein_best.pth")

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            test_encoder.parse_args(["--modality", "joint"])

    def test_missing_model_defaults_use_ssfd_loss_weights(self):
        args = train_missing_model.parse_args([])
        self.assertEqual(args.train_list, "data_txt/cumt/ssfd_train_full.txt")
        self.assertEqual(args.cmft_hidden, 1024)
        self.assertEqual(args.lambda_shared, 0.2)
        self.assertEqual(args.lambda_trans, 0.3)
        self.assertEqual(args.lambda_orth, 0.05)
        self.assertEqual(args.lambda_cons, 0.0)
        self.assertEqual(args.lr, 1e-3)
        self.assertEqual(args.encoder_lr, 1e-5)
        self.assertEqual(args.warmup_epochs, 5)
        self.assertEqual(args.lambda_anchor, 1.0)
        self.assertEqual(args.lambda_avail, 0.5)
        self.assertTrue(args.freeze_backbone)
        self.assertFalse(args.freeze_encoders)

    def test_protocol_generation_is_closed_set_only(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            datasets_txt.parse_args(["--dataset", "generic"])

    def test_recognition_rate_uses_closed_set_predictions(self):
        logits = np.array([[0.1, 0.9], [0.7, 0.3], [0.8, 0.2]], dtype=np.float32)
        labels = np.array([1, 0, 1], dtype=np.int64)
        self.assertAlmostEqual(test_encoder.recognition_rate(logits, labels), 2 / 3)

    def test_eval_metrics_reports_only_recognition_rate(self):
        logits = np.array([[0.1, 0.9], [0.7, 0.3]], dtype=np.float32)
        labels = np.array([1, 0], dtype=np.int64)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            test_encoder.eval_metrics(logits, labels, "closed")
        self.assertEqual(output.getvalue(), "\n===== closed =====\nSamples: 2\nRecognition Rate (%): 100.00\n")

    def test_error_summary_detects_class_concentration(self):
        labels = np.array([1, 1, 1, 2, 3], dtype=np.int64)
        preds = np.array([0, 0, 2, 0, 0], dtype=np.int64)
        rows, summary = test_encoder.error_analysis(preds, labels, [{} for _ in labels], "vein")
        self.assertEqual(len(rows), 5)
        self.assertEqual(summary["num_error_classes"], 3)
        self.assertEqual(summary["top_error_label"], 1)
        self.assertEqual(summary["top_error_count"], 3)
        self.assertTrue(summary["is_concentrated"])

    def test_error_summary_detects_scattered_errors(self):
        labels = np.array([1, 2, 3, 4], dtype=np.int64)
        preds = np.array([0, 0, 0, 0], dtype=np.int64)
        _, summary = test_encoder.error_analysis(preds, labels, [{} for _ in labels], "vein")
        self.assertFalse(summary["is_concentrated"])

    def test_resnet18_palm_encoder_output_shape(self):
        encoder = build_encoder("palm", input_channel=3, input_size=64, embedding_size=16, pretrained_path=None)
        self.assertEqual(encoder.__class__.__name__, "ResNet18Encoder")
        encoder.eval()
        with torch.no_grad():
            out = encoder(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(out.shape), (2, 16))

    def test_resnet18_vein_encoder_output_shape(self):
        encoder = build_encoder("vein", input_channel=3, input_size=64, embedding_size=16, pretrained_path=None)
        self.assertEqual(encoder.__class__.__name__, "ResNet18Encoder")
        self.assertEqual(encoder.se.__class__.__name__, "SEBlock")
        encoder.eval()
        with torch.no_grad():
            out = encoder(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(out.shape), (2, 16))

    def test_resnet18_palm_encoder_does_not_use_se(self):
        encoder = build_encoder("palm", input_channel=3, input_size=64, embedding_size=16, pretrained_path=None)
        self.assertEqual(encoder.se.__class__.__name__, "Identity")


if __name__ == "__main__":
    unittest.main()
