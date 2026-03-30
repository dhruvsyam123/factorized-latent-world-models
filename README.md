# Factorised Latent-Action World Model

This repository implements a full research prototype for learning object-factorised latent actions from mostly passive video in a multi-object 2D world, then aligning the controllable latent to real actions with a small labelled dataset.

## What is included

- A custom Gymnasium-compatible multi-object environment with keys, doors, hazards, blocks, distractors, and multiple task families.
- Dataset generation for passive and labelled transitions from mixed scripted and random policies.
- Three world-model variants:
  - factorised latent-action world model
  - factorised latent-action world model with an optional slot-based pixel encoder
  - monolithic latent-action baseline
  - action-conditioned baseline
- Stage-wise training for passive latent dynamics, labelled action alignment, and planning evaluation.
- A latent-space planner based on CEM.
- Unit tests for the environment, dataset serialization, and model shapes.

## Quick start

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
flwm generate-data --output-dir data/demo --passive-transitions 2000 --labelled-transitions 500
flwm train-stage1 --train data/demo/passive_train.npz --val data/demo/passive_val.npz --output checkpoints/factor_stage1.pt
flwm train-stage2 --checkpoint checkpoints/factor_stage1.pt --labelled data/demo/labelled_train.npz --output checkpoints/factor_stage2.pt
flwm evaluate --checkpoint checkpoints/factor_stage2.pt --episodes 25
pytest
```

## Recommended compute

- Local MacBook: environment work, dataset generation, tests, and tiny smoke runs.
- Colab GPU: the main training runs and baseline comparisons.
- A100-class GPU: optional for larger passive datasets or sweep-heavy ablations.

## Main commands

```bash
flwm write-config --output config.json
flwm generate-data --output-dir data/run1
flwm train-stage1 --baseline factor --train data/run1/passive_train.npz --val data/run1/passive_val.npz --output checkpoints/factor_stage1.pt
flwm train-stage1 --baseline factor --encoder-type slot --entity-weight 1.0 --train data/run1/passive_train.npz --val data/run1/passive_val.npz --output checkpoints/factor_slot_stage1.pt
flwm train-stage1 --baseline monolithic --train data/run1/passive_train.npz --val data/run1/passive_val.npz --output checkpoints/mono_stage1.pt
flwm train-stage1 --baseline action --train data/run1/labelled_train.npz --val data/run1/labelled_val.npz --output checkpoints/action_stage1.pt
flwm train-stage2 --checkpoint checkpoints/factor_stage1.pt --labelled data/run1/labelled_train.npz --output checkpoints/factor_stage2.pt
flwm evaluate --checkpoint checkpoints/factor_stage2.pt --episodes 100 --horizon 12
```

## Project structure

- `src/factor_latent_wm/envs`: simulator and task definitions
- `src/factor_latent_wm/data`: dataset generation and loading
- `src/factor_latent_wm/models`: factorised and baseline models
- `src/factor_latent_wm/training`: losses and training stages
- `src/factor_latent_wm/planning`: latent planner
- `src/factor_latent_wm/cli`: command-line entrypoints
- `notebooks/factor_latent_wm_colab.ipynb`: Colab runner for setup, sweeps, plots, and artifact export
