"""Unified entry point for the five paper-level Tongji reproductions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parent
METHODS: Final = ("ssfd", "dmrnet", "hcmig", "simmlm", "mmanet")
DEFAULT_OUTPUT_ROOT: Final = Path(
    "outputs/gipssr/full_comparisons/tongji/seed_42"
)

METHOD_DEFAULTS: Final = {
    # SSFD paper does not report a learning rate; 1e-5 is the fine-tuning
    # rate for the pretrained ResNet-18 replacement encoders.
    "ssfd": {"batch_size": 32, "eval_batch_size": 32, "learning_rate": 1e-5},
    "dmrnet": {"batch_size": 64, "eval_batch_size": 64, "learning_rate": 1e-3},
    "hcmig": {"batch_size": 8, "eval_batch_size": 8, "learning_rate": 1e-4},
    "simmlm": {"batch_size": 64, "eval_batch_size": 64, "learning_rate": 1e-4},
    "mmanet": {"batch_size": 64, "eval_batch_size": 64, "learning_rate": 1e-3},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a resumable, image-level reproduction of one comparison "
            "method on the locked Tongji validation-selection protocol."
        )
    )
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument(
        "--train-list", "--train_list",
        dest="train_list",
        default="data_txt/tongji/ssfd_train_full.txt",
    )
    parser.add_argument(
        "--val-gallery-list", "--val_gallery_list",
        dest="val_gallery_list",
        default="data_txt/tongji/ssfd_val_gallery_full.txt",
    )
    parser.add_argument(
        "--val-protocol-list", "--val_protocol_list",
        dest="val_protocol_list",
        default="data_txt/tongji/ssfd_val_protocol.txt",
    )
    parser.add_argument(
        "--palm-ckpt", "--palm_ckpt",
        dest="palm_ckpt",
        default="outputs/encoders/palm_best.pth",
    )
    parser.add_argument(
        "--vein-ckpt", "--vein_ckpt",
        dest="vein_ckpt",
        default="outputs/encoders/vein_best.pth",
    )
    parser.add_argument(
        "--output-dir", "--output_dir", "--save-dir", "--save_dir",
        dest="output_dir",
        default=None,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", "--input_size", dest="input_size", type=int, default=224)
    parser.add_argument(
        "--embedding-size", "--embedding_size",
        dest="embedding_size",
        type=int,
        default=256,
    )
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int)
    parser.add_argument(
        "--eval-batch-size", "--eval_batch_size",
        dest="eval_batch_size",
        type=int,
    )
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--teacher-epochs", "--teacher_epochs",
        dest="teacher_epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--expert-epochs", "--expert_epochs",
        dest="expert_epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--generator-epochs", "--generator_epochs",
        dest="generator_epochs",
        type=int,
        default=100,
    )
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int, default=1)
    parser.add_argument(
        "--early-stopping-patience", "--early_stopping_patience",
        dest="early_stopping_patience",
        type=int,
        default=12,
        help=(
            "Stop only the validation-selected main stage after this many "
            "consecutive validation checks without a strict metric-rank improvement."
        ),
    )
    parser.add_argument(
        "--min-epochs", "--min_epochs",
        dest="min_epochs",
        type=int,
        default=6,
        help="Minimum main-stage epochs before early stopping is allowed.",
    )
    parser.add_argument(
        "--learning-rate", "--learning_rate",
        dest="learning_rate",
        type=float,
        default=None,
    )
    parser.add_argument("--expert-lr", "--expert_lr", dest="expert_lr", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", "--gradient_clip", dest="gradient_clip", type=float, default=0.0)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable mixed precision (the paper-faithful default is FP32).",
    )
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.5)

    # SSFD-Net paper configuration (ResNet-18 encoder replacement).
    parser.add_argument("--ssfd-weight-decay", type=float, default=0.005)
    parser.add_argument(
        "--ssfd-feature-dim", type=int, default=128,
        help="Per-part shared/specific width; encoder width is twice this.",
    )
    parser.add_argument("--ssfd-cmft-hidden-dim", type=int, default=512)
    parser.add_argument("--ssfd-dropout", type=float, default=0.5)
    parser.add_argument("--ssfd-triplet-margin", type=float, default=0.1)
    parser.add_argument(
        "--ssfd-share-cmft-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # HCMIG paper configuration and memory-safe execution controls.
    parser.add_argument("--hcmig-micro-batch", "--hcmig_micro_batch", dest="hcmig_micro_batch", type=int, default=None)
    parser.add_argument("--hcmig-learning-rate", "--hcmig_learning_rate", dest="hcmig_learning_rate", type=float, default=1e-4)
    parser.add_argument("--hcmig-weight-decay", "--hcmig_weight_decay", dest="hcmig_weight_decay", type=float, default=0.005)
    parser.add_argument("--hcmig-dropout", "--hcmig_dropout", dest="hcmig_dropout", type=float, default=0.5)
    parser.add_argument("--hcmig-generator-optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--hcmig-discriminator-optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--hcmig-base-channels", type=int, default=64)
    parser.add_argument("--hcmig-recognition-embedding-dim", type=int, default=None)
    parser.add_argument("--hcmig-recognition-hidden-dim", type=int, default=None)
    parser.add_argument("--hcmig-fft-radius-ratio", type=float, default=0.1)
    parser.add_argument(
        "--hcmig-generation-epoch-cap",
        dest="hcmig_generation_epoch_cap",
        type=int,
        default=None,
        help=(
            "Cap the generation stage at this epoch count and proceed to "
            "the recognition stage early."
        ),
    )
    parser.add_argument(
        "--hcmig-stochastic-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _positive(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    for name in names:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.seed != 42:
        raise ValueError("Full comparison training is locked to seed 42")
    defaults = METHOD_DEFAULTS[args.method]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.eval_batch_size is None:
        args.eval_batch_size = defaults["eval_batch_size"]
    if args.learning_rate is None:
        args.learning_rate = defaults["learning_rate"]
    if args.hcmig_micro_batch is None:
        args.hcmig_micro_batch = args.batch_size if args.method == "hcmig" else 4
    if args.output_dir is None:
        args.output_dir = str(DEFAULT_OUTPUT_ROOT / args.method)
    _positive(
        args,
        (
            "input_size", "embedding_size", "batch_size", "eval_batch_size",
            "epochs", "teacher_epochs", "expert_epochs", "generator_epochs",
        ),
    )
    if args.num_workers < 0 or args.eval_every <= 0:
        raise ValueError("num_workers must be non-negative and eval_every positive")
    if args.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if args.min_epochs <= 0:
        raise ValueError("min_epochs must be positive")
    if (
        args.hcmig_generation_epoch_cap is not None
        and args.hcmig_generation_epoch_cap < 0
    ):
        raise ValueError("hcmig_generation_epoch_cap must be non-negative")
    if args.learning_rate <= 0.0 or args.expert_lr <= 0.0:
        raise ValueError("learning rates must be positive")
    return args


def _require_inputs(args: argparse.Namespace) -> None:
    paths = [args.train_list, args.val_gallery_list, args.val_protocol_list]
    paths.extend((args.palm_ckpt, args.vein_ckpt))
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing training inputs: {missing}")


def _ssfd_config(args: argparse.Namespace):
    from utils.full_ssfd_experiment import SSFDFullConfig

    return SSFDFullConfig(
        train_list=args.train_list,
        val_gallery_list=args.val_gallery_list,
        val_protocol_list=args.val_protocol_list,
        output_dir=args.output_dir,
        input_size=args.input_size,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.ssfd_weight_decay,
        feature_dim=args.ssfd_feature_dim,
        embedding_size=2 * args.ssfd_feature_dim,
        cmft_hidden_dim=args.ssfd_cmft_hidden_dim,
        dropout=args.ssfd_dropout,
        triplet_margin=args.ssfd_triplet_margin,
        palm_checkpoint=args.palm_ckpt,
        vein_checkpoint=args.vein_ckpt,
        share_cmft_weights=args.ssfd_share_cmft_weights,
        eval_every=args.eval_every,
        seed=args.seed,
        resume=args.resume,
        max_steps_per_epoch=args.max_steps_per_epoch,
        early_stopping_patience=args.early_stopping_patience,
        min_epochs=args.min_epochs,
    )


def train_method(args: argparse.Namespace) -> Path:
    _require_inputs(args)
    if args.method == "ssfd":
        from utils.full_ssfd_experiment import train

        train(_ssfd_config(args), device=args.device)
        checkpoint = Path(args.output_dir) / "best.pth"
    else:
        modules = {
            "dmrnet": "utils.full_dmrnet_experiment",
            "hcmig": "utils.full_hcmig_experiment",
            "simmlm": "utils.full_simmlm_experiment",
            "mmanet": "utils.full_mmanet_experiment",
        }
        module = __import__(modules[args.method], fromlist=["train"])
        checkpoint = Path(module.train(args))
    if not checkpoint.is_file():
        raise RuntimeError(f"Training did not produce best checkpoint: {checkpoint}")
    return checkpoint


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = train_method(args)
    print(f"[Complete] {args.method} best checkpoint: {checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
