from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.comparison_protocols import get_protocol_spec
from utils.full_ssfd_experiment import (
    SSFDFullConfig,
    make_representation_callback,
    train_ssfd_epoch,
)


class _RoutingModel(nn.Module):
    representation_dim = 4

    def complete_representation(self, palm: torch.Tensor, vein: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (palm[:, 0, 0, 0], vein[:, 0, 0, 0], palm[:, 0, 0, 0] + 10, vein[:, 0, 0, 0] + 10),
            dim=1,
        )

    def missing_representation(self, available: torch.Tensor, modality: str) -> torch.Tensor:
        offset = 20 if modality == "palm" else 30
        value = available[:, 0, 0, 0]
        return torch.stack((value + offset, value + offset + 1, value + offset + 2, value + offset + 3), dim=1)


class _TinySSFD(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def loss_dict(self, palm: torch.Tensor, vein: torch.Tensor, labels: torch.Tensor):
        del labels
        base = ((palm.mean() + vein.mean()) * self.weight - 1.0).square()
        zero = base * 0.0
        return {
            "classification": base,
            "triplet": zero,
            "transformation": zero,
            "inter_consistency": zero,
            "intra_consistency": zero,
            "total": base,
        }


def _loader(*, complete: bool = True) -> DataLoader:
    torch.manual_seed(2)
    palm = torch.randn(6, 3, 2, 2)
    vein = torch.randn(6, 3, 2, 2)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    masks = torch.ones(6, 2)
    if not complete:
        masks[0, 1] = 0
    return DataLoader(TensorDataset(palm, vein, labels, masks), batch_size=2)


def test_config_defaults_match_resnet_reproduction_and_tongji_protocol() -> None:
    config = SSFDFullConfig()
    spec = get_protocol_spec("tongji")
    config.validate()
    assert config.train_list == spec.selection_train_list
    assert config.val_gallery_list == spec.val_gallery_list
    assert config.val_protocol_list == spec.val_protocol_list
    assert (config.input_size, config.batch_size) == (224, 32)
    assert (config.learning_rate, config.weight_decay) == (1e-4, 0.005)
    assert config.epochs == 100
    assert config.embedding_size == 2 * config.feature_dim == 256
    assert (config.cmft_hidden_dim, config.dropout) == (512, 0.5)
    assert config.palm_arcface_s == config.vein_arcface_s == 32.0
    assert (config.palm_arcface_m, config.vein_arcface_m) == (0.25, 0.15)


def test_representation_callback_routes_mixed_complete_and_missing_samples() -> None:
    model = _RoutingModel()
    callback = make_representation_callback(model)  # type: ignore[arg-type]
    palm = torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1, 1)
    vein = torch.tensor([4.0, 5.0, 6.0]).view(3, 1, 1, 1)
    masks = torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.float32)
    output = callback(palm, vein, masks)
    torch.testing.assert_close(output[0], torch.tensor([1.0, 4.0, 11.0, 14.0]))
    torch.testing.assert_close(output[1], torch.tensor([22.0, 23.0, 24.0, 25.0]))
    torch.testing.assert_close(output[2], torch.tensor([36.0, 37.0, 38.0, 39.0]))


def test_ssfd_epoch_helper_stops_after_one_step() -> None:
    model = _TinySSFD()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    metrics = train_ssfd_epoch(
        model,  # type: ignore[arg-type]
        _loader(),
        optimizer,
        device="cpu",
        max_steps=1,
    )
    assert metrics["steps"] == 1
    assert not torch.equal(model.weight, before)


def test_ssfd_epoch_rejects_incomplete_training_pairs() -> None:
    model = _TinySSFD()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    try:
        train_ssfd_epoch(
            model,  # type: ignore[arg-type]
            _loader(complete=False),
            optimizer,
            device="cpu",
            max_steps=1,
        )
    except ValueError as error:
        assert "complete paired" in str(error)
    else:
        raise AssertionError("incomplete main-stage pairs must be rejected")
