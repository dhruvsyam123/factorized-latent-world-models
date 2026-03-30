# Factorised Latent-Action World Model

This repository implements a state-first research prototype for learning factorised latent actions in a multi-object 2D world, then aligning the controllable latent to real actions with a small labelled dataset. The default mainline uses simulator object-state tokens as the primary representation. A slot-based pixel encoder path remains available as an extension path, not the main experimental claim.

## What is included

- A custom Gymnasium-compatible multi-object environment with keys, doors, hazards, blocks, distractors, and multiple task families.
- Dataset generation for passive and labelled transitions from mixed scripted and random policies.
- Three matched world-model variants:
  - `factor`: state-factor latent-action world model with separate controllable and exogenous latents
  - `monolithic`: non-factorised latent baseline with the same control bridge and planning budget
  - `action`: action-conditioned baseline with the same token dynamics backbone
- Sequence-based dataset generation for multi-step state rollout training.
- Per-task evaluation metrics for `reach`, `key_door`, and `push`.
- Stage-wise training for passive latent dynamics, labelled action alignment, and planning evaluation.
- A latent-space planner based on CEM.
- Unit tests for the environment, dataset serialization, and model shapes.

## Quick start

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
flwm generate-data --output-dir data/demo --passive-transitions 2000 --labelled-transitions 500
flwm train-stage1 --train data/demo/passive_train.npz --val data/demo/passive_val.npz --output checkpoints/factor_stage1.pt --encoder-type state_factor
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
flwm train-stage1 --baseline factor --encoder-type state_factor --train data/run1/passive_train.npz --val data/run1/passive_val.npz --output checkpoints/factor_stage1.pt
flwm train-stage1 --baseline factor --encoder-type pixel_slot --entity-weight 1.0 --train data/run1/passive_train.npz --val data/run1/passive_val.npz --output checkpoints/factor_slot_stage1.pt
flwm train-stage1 --baseline monolithic --train data/run1/passive_train.npz --val data/run1/passive_val.npz --output checkpoints/mono_stage1.pt
flwm train-stage1 --baseline action --train data/run1/labelled_train.npz --val data/run1/labelled_val.npz --output checkpoints/action_stage1.pt
flwm train-stage2 --checkpoint checkpoints/factor_stage1.pt --labelled data/run1/labelled_train.npz --output checkpoints/factor_stage2.pt
flwm evaluate --checkpoint checkpoints/factor_stage2.pt --episodes 100 --horizon 12
```

## Project structure

- `src/factor_latent_wm/envs`: simulator and task definitions
- `src/factor_latent_wm/data`: dataset generation and loading
- `src/factor_latent_wm/models`: state-factor mainline, matched baselines, and pixel-slot extension path
- `src/factor_latent_wm/training`: multi-step rollout losses and training stages
- `src/factor_latent_wm/planning`: latent planner
- `src/factor_latent_wm/cli`: command-line entrypoints
- `notebooks/factor_latent_wm_colab.ipynb`: Colab runner for setup, sweeps, plots, and artifact export
- `docs/system_redesign_memo.md`: architectural rationale for the state-factor mainline and pixel extension split
