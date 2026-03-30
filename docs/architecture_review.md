# Architecture Review: Factorised Latent-Action World Model

## Bottom line

The current codebase is a runnable prototype, but it is not yet a defensible implementation of the research claim:

> object-factorised latent actions learned from mostly passive video provide a better interface for control and planning in multi-entity worlds than monolithic latent actions.

As written, the system can produce rough reward ordering signals, but the architecture does not cleanly test that claim. The biggest problem is not undertraining alone. The bigger problem is that the model, losses, planner, and evaluation are misaligned with the claim.

The shortest path to a scientifically coherent project is:

1. Narrow the main claim to a state-centric, object-factor world model first.
2. Treat pixels as an auxiliary rendering channel, not the primary source of factor identity.
3. Add a proper multi-step latent dynamics objective and a bidirectional control-alignment interface.
4. Evaluate on a curriculum with per-task metrics before scaling.
5. Keep the slot-based pixel encoder as a second-phase novelty extension, not the main path.

## What the current system actually is

Today the main factor model in `src/factor_latent_wm/models/factor_model.py` does this:

- For the default `entity` mode, it takes privileged per-entity features from the simulator and directly turns them into factors.
- It infers per-factor latent actions from a single `(t, t+1)` pair by concatenating current and next factors.
- It predicts a single next-step factor state.
- It trains mostly with image reconstruction plus next-entity regression.
- It maps only the first factor's latent to environment action logits in stage 2.
- At planning time, it samples controllable latent sequences, rolls the learned model forward, and scores futures using decoded entity predictions.

That means the current system is much closer to:

> a privileged-state object dynamics model with an inverse-dynamics-style latent and a shallow action classifier

than to:

> a latent-action world model learned from passive pixel observations that discovers object-level control abstractions.

## Top 5 architectural issues

### 1. The factorisation claim is not actually being tested

In the default path, the factor encoder consumes simulator entity tensors directly. That means the model does not have to discover object structure from passive video; it is handed object identity, ordering, activity masks, and normalized state variables. The slot variant avoids that input path, but it is not the main path, and it still shares the same downstream training/evaluation assumptions.

Why it matters:

- The current best-performing path is not evidence for factor discovery from video.
- The result can only support a much weaker claim: factorised latent control is useful when object state decomposition is already available.

Minimum redesign:

- Choose one scope explicitly.
- Recommended main scope: make the primary model state-centric and say so clearly. Use object states as the official observation for the main result; keep pixels only for visualization and auxiliary reconstruction.
- Optional second scope: add a real pixel-object-centric model later, with object matching and slot stability as a separate experiment.

### 2. The latent action is too weakly structured to become a robust control interface

The latent action is inferred from a single step using `MLP([f_t, f_{t+1}]) -> z_t`. There is no bottleneck beyond low dimension, no KL or prior, no sparsity, no controllability split, no temporal consistency, and no multi-step consistency in latent space. The non-agent factors are also not tied to meaningful intervention semantics; they are just whatever helps next-step prediction.

Why it matters:

- A latent inferred from one-step factor difference can work as a nuisance variable for prediction without becoming a reusable control primitive.
- The alignment head is then trying to map a fragile inverse-dynamics code to real actions after the fact.

Minimum redesign:

- Introduce a real latent-action objective:
  - controllable agent latent `z_agent`
  - optional exogenous latents `z_exo^i`
  - prior or quantization on controllable latent
  - temporal consistency or contrastive regularization across short windows
- Train multi-step rollouts, not just one-step prediction.
- Add a controllability-aware decomposition: agent latent should explain agent-caused change, while exogenous factors explain non-agent change.

### 3. Training and planning are mismatched

The model is trained mostly on one-step prediction and reconstruction, but evaluated through repeated imagined rollouts inside the planner. This is a classic train-test mismatch. The planner trusts long-horizon decoded entity predictions that the model was never optimized to keep stable.

Why it matters:

- Good one-step MSE does not imply stable long-horizon control rollouts.
- The planner may exploit decoder or probe artifacts rather than real dynamics.
- Degradation can be silent: reward changes without any clean success signal.

Minimum redesign:

- Make stage 1 explicitly multi-step:
  - roll out 5 to 15 steps in latent space during training
  - penalize entity drift over rollout horizon
  - optionally add scheduled sampling or latent overshooting
- Score planning with state variables that match the trained dynamics target, not only decoded proxies.
- Add calibration checks: one-step error, 5-step error, 10-step error, and task-conditioned rollout failure cases.

### 4. The control-alignment module is underspecified

Stage 2 currently trains only `latent -> action logits` on labelled data, and typically freezes the rest of the world model. There is no inverse mapping `action -> latent`, no cycle consistency, and no guarantee that the latent chosen by planning lives on the same manifold as latents induced by real environment transitions.

Why it matters:

- Planning may choose latents that score well under the world model but decode to poor or arbitrary controls.
- Extra labelled data can fail to help, because the alignment head is learning on top of a latent space that was never shaped for executability.

Minimum redesign:

- Replace one-way alignment with a bidirectional control bridge:
  - encoder `g(a_t) -> z_agent`
  - decoder `h(z_agent) -> a_t`
  - consistency loss between inferred latent and action-induced latent on labelled data
- During stage 2, fine-tune at least the controllable latent inference and controllable dynamics path, not only the final classifier head.
- Constrain planning to latents that come from the learned action prior or action encoder manifold.

### 5. The environment and evaluation do not isolate the research question cleanly

The current environment mixes `reach`, `key_door`, and `push`, with moving hazards and distractors, from the beginning. Passive data comes from random and heuristic policies. Evaluation reports only success rate and mean reward over mixed tasks, with no per-task breakdown, no layout split control, and no held-out interaction types. Planning is also extremely expensive and opaque, so it is hard to know whether failure comes from the representation, the alignment, the planner, or task difficulty.

Why it matters:

- When everything fails at once, you cannot tell what is broken.
- A mixed-task score can hide whether factorisation helps only on interaction-heavy tasks or only on easier navigation.

Minimum redesign:

- Introduce a curriculum:
  - Phase A: `reach` with moving hazards
  - Phase B: `key_door`
  - Phase C: `push`
  - Phase D: mixed-task training/eval
- Report per-task metrics and held-out-layout metrics separately.
- Keep planner settings cheap and fixed during model comparison.
- Add qualitative diagnostics: imagined rollout vs real rollout, control decoding accuracy, factor identity stability.

## Recommended redesign

### Recommendation: make the main project state-first

If the goal is a strong project that can actually deliver a coherent claim, the mainline system should be:

- observation for main model: simulator object states
- optional auxiliary channel: rendered pixels for demos and reconstruction
- factors: one factor per simulator object slot
- latent actions:
  - `z_agent` for controllable action
  - `z_exo^i` for exogenous factor changes if needed
- dynamics: object-set transition with cross-object interaction module
- training:
  - multi-step rollout loss on object states
  - optional image reconstruction auxiliary
  - labelled action alignment with bidirectional latent-action consistency
- planning:
  - optimize only over `z_agent`
  - propagate exogenous dynamics through learned transition model
  - score using explicit task state variables

This is much more likely to work and still answers an interesting question:

> does factorising latent control at the object level help planning and label efficiency once object decomposition is available?

That is narrower than the original pixel-passive story, but it is scientifically coherent and achievable.

### Keep the slot-based pixel encoder as a second-phase experiment

If the state-first model works, then add the slot model as a follow-up:

- encoder from pixels to slots
- slot matching or permutation-invariant supervision
- object persistence loss across time
- optional distillation from state-first factors into slots

This should be treated as an extension, not the core result.

## Minimum viable redesign by subsystem

### Environment and data

- Separate training and evaluation by task family.
- Generate dedicated passive datasets per task family, plus one mixed dataset later.
- Add held-out layouts and held-out object-count splits.
- Keep a fixed easier benchmark for fast iteration.

### Representation

- Mainline: entity-state encoder only.
- Optional auxiliary: image encoder reconstructs the rendered frame from factor states.
- Explicitly distinguish controllable factor vs exogenous factors.

### Dynamics

- Replace one-step-only training with multi-step latent rollout training.
- Predict next object states directly as the primary target.
- Use image reconstruction only as a weak auxiliary target.

### Latent action

- Add latent prior, discretization, or regularization for the controllable latent.
- Make the agent latent executable through both directions:
  - inferred from state transitions
  - induced from true actions

### Planner

- Plan over the controllable latent only.
- Reduce planner complexity and batch candidate rollouts.
- Add progress reporting and cheap evaluation presets.
- Score with explicit task features rather than only decoded probes.

### Evaluation

- Report:
  - per-task success
  - mean reward
  - one-step prediction error
  - multi-step state rollout error
  - action decoding accuracy
  - label-efficiency curve
- Compare only at matched planner budgets.

## What I would implement next

1. Reframe the main system as state-first and update the docs/claims to match.
2. Redesign stage 1 around multi-step object-state prediction.
3. Redesign stage 2 as bidirectional action-latent alignment.
4. Simplify evaluation into per-task curricula with cheap planner defaults.
5. Only after that, reintroduce the slot-based pixel encoder.

## Final judgment

Will the current architecture work as a clean demonstration of the intended factorised latent-action claim?

Not really.

Will it work as a toy prototype that can sometimes show ordering between factor and monolithic models?

Yes, possibly.

Will scaling the current code with more data and more GPU time likely fix the core problem?

Not by itself. The current weak points are structural, especially:

- privileged factor input
- weak latent-action shaping
- one-step training versus long-horizon planning
- thin control alignment
- mixed-task evaluation without diagnostics

The project needs a deliberate redesign before larger-scale experiments are worth the cost.
