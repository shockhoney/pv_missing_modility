# Repository Guidelines

## Project Structure & Module Organization

This repository trains and evaluates palmprint/palm-vein encoders for missing-modality recognition.

- `train_encoder.py`: training entry point for `palm`, `vein`, or `joint` encoders.
- `test_encoder.py`: evaluation entry point for saved checkpoints.
- `models/`: encoder backbones, including ResNet50 and ConvNeXt V2-Tiny.
- `utils/`: datasets, augmentations, metrics, and ArcFace head code.
- `tests/`: lightweight unit tests for schedules, fusion, augmentation, and encoder shapes.
- `data/`, `data_txt/`, `pretrained/`, `outputs/`, and `runs/`: local data, generated protocols, weights, checkpoints, and logs. These are ignored by git.

## Build, Test, and Development Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate protocol files:

```bash
python utils/datasets_txt.py --root_dir data --output_dir data_txt/<dataset_name>
```

Train single-modality baselines:

```bash
python train_encoder.py --modality palm
python train_encoder.py --modality vein
```

Evaluate checkpoints:

```bash
python test_encoder.py --protocol_list data_txt/polyu/test_missing_protocol.txt --palm_ckpt outputs/encoders/palm_best.pth --vein_ckpt outputs/encoders/vein_best.pth
```

Run tests:

```bash
python -m unittest discover -s tests
```

## Coding Style & Naming Conventions

Use Python with 4-space indentation and clear `snake_case` names. Keep changes localized and prefer existing helpers in `models/` and `utils/` over new abstractions. Follow the current argparse-based CLI style for new options. Name modality-specific files and checkpoints with explicit prefixes such as `palm_best.pth`, `vein_best.pth`, and `joint_best.pth`.

## Testing Guidelines

Add tests under `tests/` using `test_*.py` filenames. Existing tests use `unittest`; keep that style unless there is a clear reason to change. For training logic, prefer small tensor or mock-model tests over full training runs. Cover default arguments, schedule changes, tensor shapes, and metric/fusion behavior.

## Commit & Pull Request Guidelines

Recent commits use concise Chinese summaries. Keep commit messages short and action-oriented. Pull requests should describe the change, list the commands run, mention affected datasets/protocols, and include key metrics or screenshots only when training/evaluation behavior changes.

## Security & Configuration Tips

Do not commit datasets, pretrained weights, checkpoints, TensorBoard logs, or local editor files. Keep machine-specific paths out of code; pass paths through CLI arguments instead.
