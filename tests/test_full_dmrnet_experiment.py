from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.full_dmrnet_experiment import (
    ARCHITECTURE_VERSION,
    DMRNetImageModel,
    OFFICIAL_COMMIT,
    _best_payload,
    _learning_rate,
    _progress_payload,
    _restore_progress,
    _validate_identity_disjoint_protocol,
    load_checkpoint_model,
    representation_callback,
)


class _TinyEncoder(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(3, embedding_size, kernel_size=1)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(self.project(images), (2, 2))


class _LoaderStub:
    def __init__(self, seed: int) -> None:
        self.generator = torch.Generator().manual_seed(seed)


class FullDMRNetExperimentTest(unittest.TestCase):
    def test_official_learning_rate_schedule_and_tongji_identity_split(self) -> None:
        base = 1e-3
        expected = {
            1: 2e-4,
            5: 1e-3,
            15: 1e-3,
            16: 1e-4,
            32: 1e-4,
            33: 1e-5,
            49: 1e-5,
            50: 1e-6,
            100: 1e-6,
        }
        for epoch, value in expected.items():
            self.assertAlmostEqual(_learning_rate(epoch, base), value, places=12)

        root = Path(__file__).resolve().parents[1]
        args = SimpleNamespace(
            train_list=root / "data_txt/tongji/ssfd_train_full.txt",
            val_gallery_list=root / "data_txt/tongji/ssfd_val_gallery_full.txt",
            val_protocol_list=root / "data_txt/tongji/ssfd_val_protocol.txt",
        )
        _validate_identity_disjoint_protocol(args)

    def test_real_dual_resnet_spatial_map_single_step_and_mu_callback(self) -> None:
        torch.manual_seed(31)
        model = DMRNetImageModel(embedding_dim=256, num_classes=3)
        model.eval()
        with torch.inference_mode():
            maps = model.feature_maps(
                torch.randn(1, 3, 224, 224),
                torch.randn(1, 3, 224, 224),
            )
        self.assertEqual(maps[0].shape, (1, 256, 7, 7))
        self.assertEqual(maps[1].shape, (1, 256, 7, 7))

        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        palm = torch.randn(3, 3, 32, 32)
        vein = torch.randn(3, 3, 32, 32)
        labels = torch.tensor([0, 1, 2])
        output = model.training_step(
            palm,
            vein,
            labels,
            epoch=1,
            generator=torch.Generator().manual_seed(4),
        )
        loss = output["loss_dict"]["total"]
        self.assertTrue(torch.isfinite(loss))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        codes = output["combination_codes"]
        self.assertTrue(torch.any(codes.bitwise_and(1).bool()))
        self.assertTrue(torch.any(codes.bitwise_and(2).bool()))
        self.assertIsNotNone(model.palm_encoder.backbone.conv1.weight.grad)
        self.assertIsNotNone(model.vein_encoder.backbone.conv1.weight.grad)

        model.eval()
        callback = representation_callback(model)
        masks = torch.tensor([[True, True], [True, False], [False, True]])
        with torch.inference_mode():
            first = callback(palm, vein, masks)
            second = callback(palm, vein, masks)
        self.assertEqual(first.shape, (3, 256))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(
            torch.allclose(first.norm(dim=1), torch.ones(3), atol=1e-5)
        )

    def test_self_contained_best_checkpoint_strict_loader(self) -> None:
        def build_tiny(_modality: str, **kwargs):
            return _TinyEncoder(kwargs["embedding_size"])

        args = SimpleNamespace(embedding_size=4, alpha=2e-3, beta=0.25)
        label_ids = [4, 8, 15]
        fingerprints = {"source": "test"}
        with patch("utils.full_dmrnet_experiment.build_encoder", side_effect=build_tiny):
            model = DMRNetImageModel(
                embedding_dim=4,
                num_classes=len(label_ids),
                alpha=args.alpha,
                beta=args.beta,
            ).eval()
            payload = _best_payload(
                model,
                epoch=7,
                validation={"complete": {"fused": {"eer": 0.1}}},
                history=[{"epoch": 7}],
                args=args,
                label_ids=label_ids,
                fingerprints=fingerprints,
            )
            self.assertTrue(
                any(key.startswith("palm_encoder.") for key in payload["model"])
            )
            self.assertTrue(
                any(key.startswith("vein_encoder.") for key in payload["model"])
            )
            self.assertTrue(
                any(key.startswith("dmrnet.") for key in payload["model"])
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "best.pth"
                torch.save(payload, path)
                restored, restored_payload = load_checkpoint_model(path, "cpu")
            self.assertFalse(restored.training)
            self.assertEqual(restored_payload["label_ids"], label_ids)
            self.assertEqual(restored_payload["architecture_version"], ARCHITECTURE_VERSION)
            self.assertEqual(restored_payload["official_commit"], OFFICIAL_COMMIT)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)

    def test_progress_resume_restores_full_state_and_rng(self) -> None:
        def build_tiny(_modality: str, **kwargs):
            return _TinyEncoder(kwargs["embedding_size"])

        args = SimpleNamespace(embedding_size=4, alpha=1e-3, beta=0.5)
        fingerprints = {"source": "same"}
        label_ids = [0, 1, 2]
        with patch("utils.full_dmrnet_experiment.build_encoder", side_effect=build_tiny):
            model = DMRNetImageModel(embedding_dim=4, num_classes=3)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            combinations = torch.Generator().manual_seed(123)
            loader = _LoaderStub(456)
            payload = copy.deepcopy(
                _progress_payload(
                    model,
                    optimizer,
                    scaler,
                    epoch=4,
                    history=[{"epoch": 4}],
                    best={"epoch": 3},
                    args=args,
                    label_ids=label_ids,
                    fingerprints=fingerprints,
                    combination_generator=combinations,
                    train_loader=loader,
                )
            )
            expected_combination = torch.randint(
                1, 4, (8,), generator=combinations
            )
            expected_loader = torch.randint(0, 99, (8,), generator=loader.generator)

            restored = DMRNetImageModel(embedding_dim=4, num_classes=3)
            restored_optimizer = torch.optim.SGD(
                restored.parameters(), lr=9e-3, momentum=0.9
            )
            restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
            restored_combinations = torch.Generator().manual_seed(999)
            restored_loader = _LoaderStub(999)
            start, history, best = _restore_progress(
                payload,
                restored,
                restored_optimizer,
                restored_scaler,
                restored_combinations,
                restored_loader,
                label_ids=label_ids,
                fingerprints=fingerprints,
            )
            actual_combination = torch.randint(
                1, 4, (8,), generator=restored_combinations
            )
            actual_loader = torch.randint(
                0, 99, (8,), generator=restored_loader.generator
            )
        self.assertEqual(start, 5)
        self.assertEqual(history, [{"epoch": 4}])
        self.assertEqual(best, {"epoch": 3})
        self.assertTrue(torch.equal(expected_combination, actual_combination))
        self.assertTrue(torch.equal(expected_loader, actual_loader))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
