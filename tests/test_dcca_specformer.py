import unittest

import torch
import torch.nn.functional as F

from models.dcca_specformer import DCCASpecFormerRecovery
from models.recovery_backbone import StableRecoveryBackbone
from utils.analytic_cca import RegularizedSharedIdentityProjector


class HIASRModelTest(unittest.TestCase):
    def _models_and_gallery(self):
        torch.manual_seed(31)
        kwargs = dict(
            input_dim=16,
            shared_dim=8,
            specific_dim=8,
            transformer_layers=1,
            transformer_heads=2,
            dropout=0.0,
            max_gate=0.75,
            min_recovery_weight=0.15,
            retrieval_dropout=0.0,
            max_proxy_identities=16,
        )
        palm = torch.randn(16, 16)
        vein = torch.randn(16, 16)
        labels = torch.arange(4).repeat_interleave(4)
        gallery = {
            "palm": palm,
            "vein": vein,
            "palm_spatial": torch.randn(16, 16, 7, 7),
            "vein_spatial": torch.randn(16, 16, 7, 7),
            "labels": labels,
        }
        cca = RegularizedSharedIdentityProjector(16)
        cca.fit(palm, vein, eigen_floor=1.0)
        baseline = StableRecoveryBackbone(**kwargs)
        baseline.initialize_from_cca(cca)
        baseline.initialize_identity_proxies(palm, vein, labels)
        candidate = DCCASpecFormerRecovery(
            **kwargs,
            topk_candidates=3,
            role_queries=2,
            candidate_dropout=0.0,
        )
        incompatible = candidate.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(
            all(
                name.startswith(
                    (
                        "shared_disentangler.",
                        "hierarchical_decoder.",
                        "refinement_logit",
                    )
                )
                for name in incompatible.missing_keys
            )
        )
        return baseline, candidate, gallery

    def test_zero_initialized_hierarchical_path_preserves_backbone(self):
        baseline, candidate, gallery = self._models_and_gallery()
        baseline.eval()
        candidate.eval()
        base_memory = baseline.build_gallery_memory(gallery, chunk_size=8)
        candidate_memory = candidate.build_gallery_memory(gallery, chunk_size=8)
        base = baseline.recover_with_gallery(
            gallery["palm"][:8],
            gallery["palm_spatial"][:8],
            "palm",
            base_memory,
        )
        upgraded = candidate.recover_with_gallery(
            gallery["palm"][:8],
            gallery["palm_spatial"][:8],
            "palm",
            candidate_memory,
        )
        for name in ("mean", "recovered_scores", "fused_scores", "recovery_weight"):
            self.assertTrue(
                torch.allclose(base[name], upgraded[name], atol=2e-7, rtol=0.0),
                name,
            )

    def test_topk_candidates_remain_independent(self):
        _, candidate, gallery = self._models_and_gallery()
        candidate.eval()
        memory = candidate.build_gallery_memory(gallery, chunk_size=8)
        output = candidate.recover_with_gallery(
            gallery["vein"][:8],
            gallery["vein_spatial"][:8],
            "vein",
            memory,
        )
        self.assertEqual(tuple(output["candidate_indices"].shape), (8, 3))
        self.assertTrue(
            output["candidate_indices"].unique(dim=1).size(1) == 3
        )
        self.assertTrue(
            torch.allclose(
                output["candidate_weights"].sum(dim=1),
                torch.ones(8),
                atol=1e-6,
            )
        )

    def test_two_stage_path_receives_gradients_after_zero_init_step(self):
        _, candidate, gallery = self._models_and_gallery()
        candidate.eval()
        memory = candidate.build_gallery_memory(gallery, chunk_size=8)
        candidate.train()
        trainable = [
            parameter
            for name, parameter in candidate.named_parameters()
            if name.startswith(("shared_disentangler.", "hierarchical_decoder."))
        ]
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        target_specific = memory["vein_hierarchical_specific"][
            torch.arange(4).repeat_interleave(2)
        ]
        for _ in range(2):
            output = candidate.recover_with_gallery(
                gallery["palm"][:8],
                gallery["palm_spatial"][:8],
                "palm",
                memory,
            )
            target_indices = torch.arange(4).repeat_interleave(2)
            loss = F.cross_entropy(output["fused_scores"] / 0.05, target_indices)
            loss = loss + (
                1.0
                - F.cosine_similarity(
                    output["predicted_specific"], target_specific, dim=1
                )
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.assertGreater(
            float(
                candidate.shared_disentangler.spatial_projection.weight.grad.abs().sum()
            ),
            0.0,
        )
        self.assertGreater(
            float(
                candidate.hierarchical_decoder.stage_one_attention.in_proj_weight.grad.abs().sum()
            ),
            0.0,
        )
        self.assertGreater(
            float(
                candidate.hierarchical_decoder.stage_two_attention.in_proj_weight.grad.abs().sum()
            ),
            0.0,
        )
        self.assertGreater(
            float(
                candidate.hierarchical_decoder.embedding_residual[-1].weight.grad.abs().sum()
            ),
            0.0,
        )

    def test_refinement_never_changes_recovered_top1(self):
        _, candidate, gallery = self._models_and_gallery()
        candidate.eval()
        memory = candidate.build_gallery_memory(gallery, chunk_size=8)
        output = candidate.recover_with_gallery(
            gallery["palm"][:8],
            gallery["palm_spatial"][:8],
            "palm",
            memory,
        )
        teacher = output["teacher_fused_scores"].argmax(dim=1)
        student = output["fused_scores"].argmax(dim=1)
        self.assertTrue(torch.equal(teacher, student))


if __name__ == "__main__":
    unittest.main()
