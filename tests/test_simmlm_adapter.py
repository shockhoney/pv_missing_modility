import unittest

import torch
import torch.nn.functional as F

from models.comparisons.simmlm import SimMLMAdapter


class SimMLMAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        self.palm = torch.randn(4, 256)
        self.vein = torch.randn(4, 256)

    def test_mask_aware_routing_and_unit_representations(self) -> None:
        model = SimMLMAdapter().eval()
        palm_present = torch.tensor([True, True, False, False])
        vein_present = torch.tensor([True, False, True, True])
        encoded = model.encode(
            self.palm, self.vein, palm_present, vein_present
        )

        self.assertEqual(encoded["representation"].shape, (4, 256))
        self.assertTrue(
            torch.allclose(
                encoded["representation"].norm(dim=1),
                torch.ones(4),
                atol=1e-5,
            )
        )
        weights = encoded["router_weights"]
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(4)))
        self.assertEqual(weights[1, 0].item(), 1.0)
        self.assertEqual(weights[1, 1].item(), 0.0)
        self.assertTrue(torch.equal(weights[2:, 0], torch.zeros(2)))
        self.assertTrue(torch.equal(weights[2:, 1], torch.ones(2)))

    def test_scalar_masks_optional_inputs_and_fp16_cache(self) -> None:
        model = SimMLMAdapter().eval()
        palm_only = model.representation(self.palm.half(), None, True, False)
        vein_only = model.representation(None, self.vein.half(), False, True)
        self.assertEqual(palm_only.dtype, torch.float32)
        self.assertEqual(vein_only.dtype, torch.float32)
        self.assertTrue(torch.isfinite(palm_only).all())
        self.assertTrue(torch.isfinite(vein_only).all())
        with self.assertRaises(ValueError):
            model.representation(self.palm, self.vein, False, False)
        with self.assertRaises(ValueError):
            model.representation(None, self.vein, True, True)

    def test_complete_single_losses_match_per_sample_mofe(self) -> None:
        model = SimMLMAdapter(num_classes=7).train()
        labels = torch.tensor([0, 1, 2, 3])
        output = model(self.palm, self.vein, labels=labels)
        losses = output["loss_dict"]

        per_sample = {
            name: F.cross_entropy(logits, labels, reduction="none")
            for name, logits in output["logits"].items()
        }
        expected_task = (
            2.0 * per_sample["complete"].mean()
            + per_sample["palm"].mean()
            + per_sample["vein"].mean()
        ) / 4.0
        expected_palm = F.relu(
            per_sample["complete"] - per_sample["palm"]
        ).mean()
        expected_vein = F.relu(
            per_sample["complete"] - per_sample["vein"]
        ).mean()
        expected_mofe = (expected_palm + expected_vein) / 2.0
        expected_total = expected_task + 0.1 * expected_mofe

        self.assertTrue(torch.allclose(losses["task"], expected_task))
        self.assertTrue(torch.allclose(losses["more_vs_palm"], expected_palm))
        self.assertTrue(torch.allclose(losses["more_vs_vein"], expected_vein))
        self.assertTrue(torch.allclose(losses["mofe"], expected_mofe))
        self.assertTrue(torch.allclose(losses["total"], expected_total))
        self.assertGreaterEqual(losses["mofe"].item(), 0.0)

    def test_xavier_initialized_adapter_receives_finite_gradients(self) -> None:
        model = SimMLMAdapter(num_classes=7).train()
        output = model(
            self.palm, self.vein, labels=torch.tensor([0, 1, 2, 3])
        )
        output["loss_dict"]["total"].backward()
        for parameter in (
            model.palm_expert.projection.weight,
            model.vein_expert.projection.weight,
            model.router[0].weight,
            model.router[2].weight,
            model.classifier.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))


if __name__ == "__main__":
    unittest.main()
