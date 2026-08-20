from __future__ import annotations

import gc

import pytest
import torch

from models.comparisons.mmanet_full import (
    ARCHITECTURE_VERSION as MMANET_ARCHITECTURE_VERSION,
    OFFICIAL_COMMIT as MMANET_OFFICIAL_COMMIT,
    MMANetImageModel,
)
from models.comparisons.simmlm_full import (
    ARCHITECTURE_VERSION as SIMMLM_ARCHITECTURE_VERSION,
    OFFICIAL_COMMIT as SIMMLM_OFFICIAL_COMMIT,
    SimMLMImageModel,
)
from run_tongji_full_comparisons import (
    METHOD_ORDER,
    _training_metadata,
    parse_args as parse_runner_args,
    train_command,
)
from test_full_comparison import load_method_checkpoint, parse_args as parse_test_args
from train_full_comparison import METHOD_DEFAULTS, parse_args as parse_train_args


@pytest.mark.parametrize("method", METHOD_ORDER)
def test_training_cli_uses_paper_epoch_and_method_defaults(method: str) -> None:
    args = parse_train_args(["--method", method])

    assert args.epochs == 100
    assert args.teacher_epochs == 100
    assert args.expert_epochs == 100
    assert args.generator_epochs == 100
    assert args.batch_size == METHOD_DEFAULTS[method]["batch_size"]
    assert args.eval_batch_size == METHOD_DEFAULTS[method]["eval_batch_size"]
    assert args.learning_rate == METHOD_DEFAULTS[method]["learning_rate"]
    assert args.output_dir.endswith(f"seed_42/{method}")
    assert args.early_stopping_patience == 12
    assert args.min_epochs == 6


def test_runner_parses_comma_subset_in_canonical_order_and_overrides_stages() -> None:
    args = parse_runner_args(
        ["--methods", "mmanet,ssfd", "dmrnet", "--epochs", "7", "--dry-run"]
    )

    assert args.methods == ("ssfd", "dmrnet", "mmanet")
    command = train_command(args, "ssfd")
    assert command[command.index("--epochs") + 1] == "7"
    assert command[command.index("--teacher-epochs") + 1] == "7"
    assert command[command.index("--expert-epochs") + 1] == "7"
    assert command[command.index("--generator-epochs") + 1] == "7"
    assert command[command.index("--early-stopping-patience") + 1] == "12"
    assert command[command.index("--min-epochs") + 1] == "6"


def test_runner_reads_auditable_best_and_actual_epochs(tmp_path) -> None:
    method_dir = tmp_path / "simmlm"
    method_dir.mkdir()
    torch.save({"best_epoch": 9}, method_dir / "best.pth")
    torch.save(
        {
            "epoch": 21,
            "actual_epochs": {"expert": 100, "cooperative": 21},
            "early_stopping": {
                "stopped": True,
                "stop_reason": "metric_rank did not strictly improve",
            },
        },
        method_dir / "last.pth",
    )

    metadata = _training_metadata(tmp_path, "simmlm")

    assert metadata == {
        "best_epoch": 9,
        "actual_epochs": {"expert": 100, "cooperative": 21},
        "early_stopped": True,
        "stop_reason": "metric_rank did not strictly improve",
    }


def test_test_cli_defaults_result_next_to_checkpoint() -> None:
    args = parse_test_args(["--checkpoint", "somewhere/best.pth"])

    assert args.method == "auto"
    assert args.metrics_path == "somewhere/test_results.json"


def test_simmlm_synthetic_checkpoint_is_self_contained(tmp_path) -> None:
    original = SimMLMImageModel(embedding_dim=8, num_classes=3)
    checkpoint = tmp_path / "simmlm.pth"
    torch.save(
        {
            "architecture_version": SIMMLM_ARCHITECTURE_VERSION,
            "official_commit": SIMMLM_OFFICIAL_COMMIT,
            "method": "simmlm",
            "model": original.state_dict(),
            "args": {"embedding_size": 8, "input_size": 32},
            "label_ids": [2, 5, 9],
        },
        checkpoint,
    )
    del original
    gc.collect()

    method, model, payload, callback = load_method_checkpoint(
        checkpoint, method="auto", device="cpu"
    )
    images = torch.rand(2, 3, 32, 32)
    masks = torch.tensor([[True, True], [True, False]])
    with torch.inference_mode():
        representation = callback(images, images, masks)

    assert method == "simmlm"
    assert payload["official_commit"] == SIMMLM_OFFICIAL_COMMIT
    assert representation.shape == (2, 16)
    assert torch.isfinite(representation).all()
    assert model.training is False


def test_mmanet_synthetic_checkpoint_is_self_contained(tmp_path) -> None:
    original = MMANetImageModel(num_classes=3)
    original.mar_observed.fill_(True)
    original.weak_combination.fill_(original.PALM_ONLY)
    checkpoint = tmp_path / "mmanet.pth"
    torch.save(
        {
            "architecture_version": MMANET_ARCHITECTURE_VERSION,
            "official_commit": MMANET_OFFICIAL_COMMIT,
            "method": "mmanet",
            "model": original.state_dict(),
            "args": {"input_size": 32},
            "label_ids": [0, 1, 2],
        },
        checkpoint,
    )
    del original
    gc.collect()

    method, model, payload, callback = load_method_checkpoint(
        checkpoint, method="mmanet", device="cpu"
    )
    images = torch.rand(2, 3, 32, 32)
    masks = torch.tensor([[True, True], [False, True]])
    with torch.inference_mode():
        representation = callback(images, images, masks)

    assert method == "mmanet"
    assert payload["official_commit"] == MMANET_OFFICIAL_COMMIT
    assert representation.shape == (2, 512)
    assert torch.isfinite(representation).all()
    assert model.weak_modality == "palm"
    assert model.training is False


def test_checkpoint_method_mismatch_is_rejected(tmp_path) -> None:
    model = SimMLMImageModel(embedding_dim=4, num_classes=2)
    checkpoint = tmp_path / "wrong_method.pth"
    torch.save(
        {
            "architecture_version": SIMMLM_ARCHITECTURE_VERSION,
            "method": "simmlm",
            "model": model.state_dict(),
            "args": {"embedding_size": 4, "input_size": 32},
            "label_ids": [0, 1],
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="does not match"):
        load_method_checkpoint(checkpoint, method="dmrnet", device="cpu")
