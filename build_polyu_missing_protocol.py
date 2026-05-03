import argparse
import os

from utils.datasets_txt import build_polyu_missing_protocols


def main():
    parser = argparse.ArgumentParser(description="Build PolyU missing-modality protocol files.")
    parser.add_argument("--root_dir", type=str, default="data/PolyU")
    parser.add_argument("--output_dir", type=str, default="data_txt")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--palm_dir_name", type=str, default="Red")
    parser.add_argument("--vein_dir_name", type=str, default="NIR")
    args = parser.parse_args()

    summary = build_polyu_missing_protocols(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        palm_dir_name=args.palm_dir_name,
        vein_dir_name=args.vein_dir_name,
    )

    print("PolyU missing-modality protocols generated.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
