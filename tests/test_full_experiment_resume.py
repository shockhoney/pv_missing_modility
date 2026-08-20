from __future__ import annotations

from types import SimpleNamespace

import torch

from utils.checkpoint_io import safe_torch_load, save_checkpoint_atomic
from utils.full_mmanet_experiment import _progress as mmanet_progress
from utils.full_simmlm_experiment import _progress_payload as simmlm_progress


def _trained_linear(optimizer_name: str):
    model = torch.nn.Linear(3, 2)
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    loss = model(torch.randn(4, 3)).square().mean()
    loss.backward()
    optimizer.step()
    return model, optimizer


def _assert_loader_rng_round_trip(saved_state: torch.Tensor) -> None:
    expected_generator = torch.Generator()
    expected_generator.set_state(saved_state)
    expected = torch.randperm(31, generator=expected_generator)
    resumed_generator = torch.Generator()
    resumed_generator.set_state(saved_state)
    actual = torch.randperm(31, generator=resumed_generator)
    torch.testing.assert_close(actual, expected)


def test_simmlm_progress_round_trips_optimizer_scaler_and_loader_rng(tmp_path) -> None:
    model, optimizer = _trained_linear("adam")
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loader = SimpleNamespace(generator=torch.Generator().manual_seed(913))
    payload = simmlm_progress(
        model,
        "cooperative",
        7,
        {"expert": [], "cooperative": []},
        [optimizer],
        {"rank": None},
        scaler,
        loader,
    )
    path = tmp_path / "simmlm_last.pth"
    save_checkpoint_atomic(path, payload)
    restored = safe_torch_load(path, "cpu")
    assert restored["stage"] == "cooperative" and restored["epoch"] == 7
    assert restored["optimizers"][0]["state"]
    assert restored["scaler"] == scaler.state_dict()
    _assert_loader_rng_round_trip(restored["loader_rng_state"])


def test_mmanet_progress_round_trips_optimizer_scaler_and_loader_rng(tmp_path) -> None:
    model, optimizer = _trained_linear("sgd")
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loader = SimpleNamespace(generator=torch.Generator().manual_seed(1201))
    payload = mmanet_progress(
        model,
        "deployment",
        11,
        {"teacher": [], "deployment": []},
        optimizer,
        {"rank": None},
        scaler,
        loader,
    )
    path = tmp_path / "mmanet_last.pth"
    save_checkpoint_atomic(path, payload)
    restored = safe_torch_load(path, "cpu")
    assert restored["stage"] == "deployment" and restored["epoch"] == 11
    assert restored["optimizer"]["state"]
    assert restored["scaler"] == scaler.state_dict()
    _assert_loader_rng_round_trip(restored["loader_rng_state"])
