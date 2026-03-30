from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from factor_latent_wm.config.core import ProjectConfig, default_project_config
from factor_latent_wm.data.generate import generate_default_splits
from factor_latent_wm.training.stages import evaluate_checkpoint, train_stage1, train_stage2
from factor_latent_wm.utils.seeding import seed_everything


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="State-first factorised latent-action world model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_config = subparsers.add_parser("write-config")
    write_config.add_argument("--output", type=Path, default=Path("config.json"))

    gen = subparsers.add_parser("generate-data")
    gen.add_argument("--output-dir", type=Path, required=True)
    gen.add_argument("--passive-transitions", type=int, default=None)
    gen.add_argument("--labelled-transitions", type=int, default=None)

    stage1 = subparsers.add_parser("train-stage1")
    stage1.add_argument("--train", type=Path, required=True)
    stage1.add_argument("--val", type=Path, required=True)
    stage1.add_argument("--output", type=Path, required=True)
    stage1.add_argument("--baseline", choices=["factor", "monolithic", "action"], default="factor")
    stage1.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    stage1.add_argument("--epochs", type=int, default=None)
    stage1.add_argument("--batch-size", type=int, default=None)
    stage1.add_argument("--encoder-type", choices=["state_factor", "pixel_slot", "entity", "slot"], default=None)
    stage1.add_argument("--entity-weight", type=float, default=None)

    stage2 = subparsers.add_parser("train-stage2")
    stage2.add_argument("--checkpoint", type=Path, required=True)
    stage2.add_argument("--labelled", type=Path, required=True)
    stage2.add_argument("--output", type=Path, required=True)
    stage2.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    stage2.add_argument("--epochs", type=int, default=None)
    stage2.add_argument("--batch-size", type=int, default=None)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--episodes", type=int, default=25)
    evaluate.add_argument("--horizon", type=int, default=12)
    evaluate.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = default_project_config()
    seed_everything(config.dataset.seed)

    if args.command == "write-config":
        config.save(args.output)
        print(f"Wrote default config to {args.output}")
        return

    if args.command == "generate-data":
        if args.passive_transitions is not None:
            config.dataset.passive_train_transitions = args.passive_transitions
            config.dataset.passive_val_transitions = max(256, args.passive_transitions // 10)
        if args.labelled_transitions is not None:
            config.dataset.labelled_train_transitions = args.labelled_transitions
            config.dataset.labelled_val_transitions = max(128, args.labelled_transitions // 5)
        outputs = generate_default_splits(args.output_dir, config.env, config.dataset)
        print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
        return

    if args.command == "train-stage1":
        config.train.device = args.device
        if args.epochs is not None:
            config.train.stage1_epochs = args.epochs
        if args.batch_size is not None:
            config.train.batch_size = args.batch_size
        if args.encoder_type is not None:
            config.model.encoder_type = args.encoder_type
        if args.entity_weight is not None:
            config.model.entity_weight = args.entity_weight
        path = train_stage1(
            str(args.train),
            str(args.val),
            str(args.output),
            config.model,
            config.train,
            baseline=args.baseline,
        )
        print(f"Saved stage 1 checkpoint to {path}")
        return

    if args.command == "train-stage2":
        config.train.device = args.device
        if args.epochs is not None:
            config.train.stage2_epochs = args.epochs
        if args.batch_size is not None:
            config.train.batch_size = args.batch_size
        path = train_stage2(str(args.checkpoint), str(args.labelled), str(args.output), config.train)
        print(f"Saved stage 2 checkpoint to {path}")
        return

    if args.command == "evaluate":
        config.eval.episodes = args.episodes
        config.eval.planner_horizon = args.horizon
        metrics = evaluate_checkpoint(str(args.checkpoint), config.env, config.eval, args.device)
        print(json.dumps(metrics, indent=2))
        return

    raise ValueError(f"Unsupported command {args.command}")
