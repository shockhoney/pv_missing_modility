import unittest

import torch
import torch.nn.functional as F

from models.dcca_specformer import DCCASpecFormerRecovery
from train_dcca_specformer import configure_stage, low_far_pauc_loss
from utils.analytic_cca import RegularizedSharedIdentityProjector


class BalancedRecoveryTest(unittest.TestCase):
    def _model_and_data(self, retrieval_dropout=0.5):
        torch.manual_seed(19)
        model = DCCASpecFormerRecovery(
            input_dim=16,
            shared_dim=8,
            specific_dim=8,
            transformer_layers=1,
            transformer_heads=2,
            dropout=0.0,
            max_gate=0.75,
            min_recovery_weight=0.25,
            retrieval_dropout=retrieval_dropout,
            max_proxy_identities=16,
        )
        palm = torch.randn(16, 16)
        vein = torch.randn(16, 16)
        labels = torch.arange(4).repeat_interleave(4)
        cca = RegularizedSharedIdentityProjector(16)
        cca.fit(palm, vein, eigen_floor=1.0)
        model.initialize_from_cca(cca)
        model.initialize_identity_proxies(palm, vein, labels)
        gallery = {
            "palm": palm,
            "vein": vein,
            "palm_spatial": torch.randn(16, 16, 7, 7),
            "vein_spatial": torch.randn(16, 16, 7, 7),
            "labels": labels,
        }
        return model, gallery
    def test_gate_has_deterministic_balanced_initialization(self):
        model, _ = self._model_and_data()
        values = model.safe_gate(torch.zeros(3, 10))
        self.assertTrue(
            torch.allclose(values, torch.full_like(values, 0.30), atol=1e-6)
        )


    def test_both_fusion_branches_have_bounded_contribution(self):
        model, gallery = self._model_and_data()
        model.eval()
        memory = model.build_gallery_memory(gallery, chunk_size=8)
        output = model.recover_with_gallery(
            gallery["palm"][:8], gallery["palm_spatial"][:8], "palm", memory
        )
        self.assertTrue((output["recovery_weight"] >= 0.25).all())
        self.assertTrue((output["recovery_weight"] <= 0.75).all())
        self.assertTrue((output["shared_weight"] >= 0.25).all())
        self.assertTrue((output["shared_weight"] <= 0.75).all())
        self.assertTrue(
            torch.allclose(
                output["shared_weight"] + output["recovery_weight"],
                torch.ones_like(output["shared_weight"]),
            )
        )

    def test_retrieval_dropout_and_cycle_receive_gradients(self):
        model, gallery = self._model_and_data(retrieval_dropout=0.999)
        model.eval()
        memory = model.build_gallery_memory(gallery, chunk_size=8)
        model.train()
        output = model.recover_with_gallery(
            gallery["palm"][:8], gallery["palm_spatial"][:8], "palm", memory
        )
        self.assertGreater(float(output["retrieval_dropout_fraction"]), 0.0)
        loss = output["fused_scores"].mean() + output["cycle"].mean()
        loss.backward()
        self.assertGreater(
            float(model.recovery_decoder.full_residual.weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(model.cycle_heads[0][-1].weight.grad.abs().sum()), 0.0
        )

    def test_low_far_pauc_is_differentiable(self):
        scores = torch.randn(8, 6, requires_grad=True)
        targets = torch.arange(8) % 6
        loss = low_far_pauc_loss(scores, targets, margin=0.05, temperature=0.05)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertGreater(float(scores.grad.abs().sum()), 0.0)


    def test_shared_stage_backprop_updates_refiner_and_identity_proxies(self):
        model, gallery = self._model_and_data(retrieval_dropout=0.0)
        configure_stage(model, shared_only=True)
        labels = gallery["labels"]
        palm_raw = model.palm_projector.forward_raw(gallery["palm"])
        vein_raw = model.vein_projector.forward_raw(gallery["vein"])
        palm_logits, targets = model.proxy_logits(
            F.normalize(palm_raw, dim=1), labels, temperature=0.05
        )
        proxy_loss = F.cross_entropy(palm_logits, targets)
        pair_loss = 1.0 - F.cosine_similarity(palm_raw, vein_raw, dim=1).mean()
        (proxy_loss + pair_loss).backward()
        self.assertIsNone(model.palm_projector.base.weight.grad)
        self.assertGreater(
            float(model.palm_projector.refiner[-1].weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(float(model.identity_proxies.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
