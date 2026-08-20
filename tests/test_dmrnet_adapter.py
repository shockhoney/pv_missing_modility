import unittest

import torch
import torch.nn.functional as F

from models.comparisons.dmrnet import DMRNetAdapter


class DMRNetAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.palm = torch.randn(3, 256, 7, 7)
        self.vein = torch.randn(3, 256, 7, 7)
        self.palm_present = torch.tensor([True, True, False])
        self.vein_present = torch.tensor([True, False, True])

    def test_official_spatial_heads_and_mean_inference(self) -> None:
        model = DMRNetAdapter().eval()
        self.assertIsInstance(model.mean_head[0], torch.nn.Conv2d)
        self.assertIsInstance(model.mean_head[1], torch.nn.BatchNorm2d)
        self.assertEqual(model.mean_head[0].kernel_size, (1, 1))
        self.assertIs(model.hcr_predictor, model.classifier)

        output = model(
            self.palm,
            self.vein,
            self.palm_present,
            self.vein_present,
            sample=False,
        )
        self.assertEqual(output["mean"].shape, (3, 256, 7, 7))
        self.assertEqual(output["task_logits"].shape, (3, 432))
        self.assertTrue(
            torch.equal(output["combination_codes"], torch.tensor([3, 1, 2]))
        )
        self.assertTrue(
            torch.allclose(
                output["inference_representation"].norm(dim=1),
                torch.ones(3),
                atol=1e-5,
            )
        )
        repeated = model.representation(
            self.palm,
            self.vein,
            self.palm_present,
            self.vein_present,
            sample=False,
        )
        self.assertTrue(
            torch.allclose(output["inference_representation"], repeated)
        )
        # The official predictor receives raw GAP features, not unit retrieval
        # vectors.
        expected_logits = model.classifier(model.pool(output["mean"]))
        self.assertTrue(torch.allclose(output["task_logits"], expected_logits))

    def test_uniform_random_nonempty_combinations(self) -> None:
        generator_a = torch.Generator().manual_seed(13)
        generator_b = torch.Generator().manual_seed(13)
        palm, vein, codes = DMRNetAdapter.random_modality_masks(
            6000, torch.device("cpu"), generator=generator_a
        )
        _, _, repeated = DMRNetAdapter.random_modality_masks(
            6000, torch.device("cpu"), generator=generator_b
        )
        self.assertTrue(torch.equal(codes, repeated))
        self.assertTrue(torch.equal(codes, palm.long() + 2 * vein.long()))
        self.assertFalse(torch.any(~palm & ~vein))
        counts = torch.bincount(codes, minlength=4)[1:].float()
        frequencies = counts / counts.sum()
        self.assertTrue(
            torch.all(torch.abs(frequencies - 1.0 / 3.0) < 0.03), frequencies
        )

    def test_kl_is_official_sum_over_map_then_batch_mean(self) -> None:
        mean = torch.ones(2, 3, 2, 2)
        logvar = torch.zeros_like(mean)
        # Each of 12 elements contributes 0.5 for each sample.
        self.assertAlmostEqual(
            float(DMRNetAdapter.kl_divergence(mean, logvar)), 6.0, places=6
        )

    def test_whole_data_variance_accumulation_and_top_v(self) -> None:
        model = DMRNetAdapter(input_channels=1, embedding_dim=2, num_classes=3)
        logvar = torch.log(
            torch.tensor(
                [
                    [[[1.0, 3.0]]],
                    [[[5.0, 7.0]]],
                    [[[2.0, 2.0]]],
                    [[[9.0, 11.0]]],
                ]
            )
        )
        codes = torch.tensor([1, 1, 2, 3])
        model.update_hcr_statistics(logvar, codes)
        actual = model.variance_by_combination()
        self.assertTrue(torch.isnan(actual[0]))
        self.assertTrue(
            torch.allclose(
                actual[1:], torch.tensor([4.0, 2.0, 10.0], dtype=torch.float64)
            )
        )
        self.assertTrue(
            torch.equal(
                model.combination_observation_count, torch.tensor([0, 2, 1, 1])
            )
        )
        self.assertTrue(
            torch.equal(
                model.combination_variance_element_count,
                torch.tensor([0, 4, 2, 2]),
            )
        )
        self.assertTrue(
            torch.equal(model.hard_combination_codes(), torch.tensor([3, 1]))
        )

    def test_statistics_require_labels_and_all_combinations_before_hcr(self) -> None:
        model = DMRNetAdapter().train()
        model(self.palm, self.vein, True, True, epoch=1)
        self.assertEqual(int(model.combination_observation_count.sum()), 0)

        labels = torch.tensor([0, 1, 2])
        output = model(
            self.palm, self.vein, True, False, labels=labels, epoch=5
        )
        self.assertFalse(output["hcr_statistics_complete"])
        output = model(
            self.palm, self.vein, True, False, labels=labels, epoch=6
        )
        self.assertFalse(output["hcr_active"])

    def test_hcr_schedule_independent_shared_predictor_and_gradients(self) -> None:
        model = DMRNetAdapter().train()
        labels = torch.tensor([0, 1, 2])
        warmup = model(
            self.palm,
            self.vein,
            self.palm_present,
            self.vein_present,
            labels=labels,
            epoch=5,
        )
        self.assertFalse(warmup["hcr_active"])
        self.assertEqual(warmup["loss_dict"]["hcr"].item(), 0.0)
        self.assertTrue(torch.isfinite(warmup["loss_dict"]["kl"]))
        self.assertEqual(model.hard_combination_codes().numel(), 2)

        torch.manual_seed(19)
        active = model(
            self.palm,
            self.vein,
            self.palm_present,
            self.vein_present,
            labels=labels,
            epoch=6,
        )
        self.assertTrue(active["hcr_active"])
        self.assertGreater(active["loss_dict"]["hcr"].item(), 0.0)
        self.assertEqual(
            active["hcr_logits"].size(0),
            int(active["hcr_sample_mask"].sum()),
        )
        selected_labels = labels[active["hcr_sample_mask"]]
        self.assertTrue(
            torch.allclose(
                active["hcr_logits"],
                active["task_logits"][active["hcr_sample_mask"]],
            )
        )
        expected_hcr = F.cross_entropy(
            active["hcr_logits"], selected_labels, reduction="sum"
        ) / labels.numel()
        self.assertTrue(torch.allclose(active["loss_dict"]["hcr"], expected_hcr))
        expected_total = (
            active["loss_dict"]["task"]
            + 1e-3 * active["loss_dict"]["kl"]
            + 0.5 * active["loss_dict"]["hcr"]
        )
        self.assertTrue(torch.allclose(active["loss_dict"]["total"], expected_total))
        active["loss_dict"]["total"].backward()
        self.assertIsNotNone(model.mean_head[0].weight.grad)
        self.assertIsNotNone(model.logvar_head[0].weight.grad)
        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertTrue(torch.isfinite(model.mean_head[0].weight.grad).all())
        self.assertTrue(torch.isfinite(model.logvar_head[0].weight.grad).all())

    def test_training_step_randomizes_complete_pairs(self) -> None:
        model = DMRNetAdapter().train()
        labels = torch.tensor([0, 1, 2])
        generator = torch.Generator().manual_seed(29)
        palm = self.palm.clone().requires_grad_()
        vein = self.vein.clone().requires_grad_()
        output = model.training_step(
            palm, vein, labels, epoch=1, generator=generator
        )
        self.assertTrue(
            torch.all(
                (output["combination_codes"] >= 1)
                & (output["combination_codes"] <= 3)
            )
        )
        self.assertFalse(
            torch.any(~output["palm_present"] & ~output["vein_present"])
        )
        self.assertTrue(torch.isfinite(output["loss_dict"]["total"]))
        output["loss_dict"]["total"].backward()
        # Both end-to-end encoder branches receive gradients whenever their
        # sampled combination contains that modality.
        if torch.any(output["palm_present"]):
            self.assertGreater(float(palm.grad.abs().sum()), 0.0)
        if torch.any(output["vein_present"]):
            self.assertGreater(float(vein.grad.abs().sum()), 0.0)

    def test_sampling_eval_mean_and_invalid_inputs(self) -> None:
        model = DMRNetAdapter().eval()
        mean_rep = model.representation(
            self.palm, self.vein, True, True, sample=False
        )
        sampled_rep = model.representation(
            self.palm, self.vein, True, True, sample=True
        )
        self.assertFalse(torch.allclose(mean_rep, sampled_rep))
        palm_only = model.representation(self.palm, None, True, False, sample=False)
        self.assertEqual(palm_only.shape, (3, 256))
        with self.assertRaises(ValueError):
            model.representation(self.palm, self.vein, False, False)
        with self.assertRaises(ValueError):
            model.training_step(
                self.palm, self.vein[:2], torch.tensor([0, 1, 2]), epoch=1
            )


if __name__ == "__main__":
    unittest.main()
