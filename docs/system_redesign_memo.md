# System Redesign Memo

## Why this memo exists

The current repo runs, but the question is not whether it runs. The question is whether the architecture, training pipeline, and evaluation setup can produce a result that actually supports the intended claim:

> object-factorised latent actions learned from mostly passive video are a better interface for planning and control in multi-entity worlds than monolithic latent actions

My conclusion is:

- the current code is a valid prototype
- the current code is **not** yet a clean test of that claim
- more Colab sweeps on the current architecture are low-value
- the next step should be a structural redesign, not more hyperparameter search

## What the current system really is

The default path is not learning object factors from passive video alone.

What it actually does:

- the main `entity` encoder consumes privileged per-object simulator features
- latent actions are inferred from single-step inverse-dynamics style differences
- stage 1 trains mostly one-step prediction and reconstruction
- stage 2 trains a thin latent-to-action classifier
- planning samples arbitrary latent vectors and assumes those are executable controls

That means the current system is closer to:

> a privileged object-state world model with a shallow inverse-dynamics latent

than to:

> a passive-video latent-action world model that discovers reusable object-level control primitives

This distinction matters. It changes what the experiments can legitimately claim.

## The biggest architectural problems

### 1. The main path uses privileged structure

The current best path uses simulator entity tensors as the primary factor input. That is useful for debugging, but it short-circuits the central “discover factors from passive observations” story.

Implication:

- if the factor model wins, the result supports a narrower claim:
  - factorised latent control helps once object decomposition is already available
- it does **not** support the full passive-video factor-discovery claim

### 2. The latent-control bridge is not valid enough for planning

Right now the model learns a posterior latent from `(o_t, o_{t+1})` and then learns `latent -> action`.

At test time, planning samples free latent vectors and decodes them as actions.

That is a weak bridge because:

- the planner is searching a latent space with no learned action prior
- the alignment module only sees posterior latents induced by real transitions
- there is no guarantee that sampled planning latents lie on the executable manifold

This is one of the main reasons more labelled data is not clearly helping.

### 3. Training and evaluation are mismatched

Stage 1 is almost entirely one-step.

Evaluation is long-horizon model-predictive control.

That mismatch is severe:

- a model can have acceptable one-step losses and still drift badly under rollout
- the planner can exploit short-horizon probe artifacts
- the signal you get in reward or success is noisy and hard to interpret

### 4. Controllable and exogenous dynamics are not properly separated

The architecture implicitly treats the first factor as controllable and everyone else as “whatever the dynamics model does when their latent is zero”.

That is not enough in multi-entity worlds with moving hazards, distractors, and object interactions.

A proper design needs:

- a controllable latent channel for the agent
- a separate exogenous latent or exogenous predictive mechanism for non-agent dynamics

Otherwise the planner is reasoning under the wrong causal model.

### 5. The benchmark is not instrumented for diagnosis

The current setup mixes:

- `reach`
- `key_door`
- `push`
- hazards
- distractors

from the start, and evaluation returns only aggregate `success_rate` and `mean_reward`.

That makes it hard to tell:

- whether the representation is bad
- whether alignment is bad
- whether the planner is bad
- whether the task is just too hard

The current reward is also task-misaligned for `key_door` and `push`, which makes `mean_reward` a weak signal.

## Will the current architecture work?

### For a demo

Yes, probably.

The current repo can produce:

- videos
- rough ordering between variants
- an end-to-end latent-control story

### For a defensible research result

Not reliably.

The current architecture is too structurally loose in the exact places that matter:

- factor discovery
- latent executability
- multi-step stability
- fair baseline parity
- evaluation clarity

If the goal is a strong final project or paper-style result, the current system needs a redesign first.

## What the redesigned project should be

## Recommended framing

Split the project into two explicit tracks.

### Track A: state-factor scaffold

Purpose:

- validate the algorithmic claim in the cleanest possible setting

Observation:

- object state tokens only

Claim:

- factorised controllable latent actions help planning and label-efficiency once object decomposition is available

Role:

- this becomes the mainline result for v1

### Track B: pixel-factor extension

Purpose:

- test whether some of the same gains survive when factors must be inferred from pixels

Observation:

- pixels only, slot/object-centric encoder

Claim:

- optional novelty extension after Track A works

Role:

- second-phase experiment, not the first thing to optimize

## Recommended architecture v2

### Representation

For Track A:

- one object token per entity
- no ambiguity about object identity

For Track B:

- slot/object encoder from pixels
- explicit temporal slot persistence or matching

### Latent structure

Introduce separate latent channels:

- `z_ctrl`
  - controllable latent for the agent
- `z_exo`
  - exogenous latent for non-agent dynamics, hazards, distractors, and interaction residuals

Learn both:

- posterior `q(z_ctrl, z_exo | x_t, x_{t+1})`
- prior `p(z_ctrl, z_exo | x_t)`

Planning should optimize only `z_ctrl`.

The exogenous channel should be propagated under a learned prior, not chosen by the planner.

### Dynamics

The primary target should be object-state rollout, not image reconstruction.

Use:

- multi-step object rollout loss
- interaction-aware token dynamics
- optional image reconstruction as auxiliary only

The world model should be optimized for the thing planning actually uses.

### Action bridge

Replace the current thin `latent -> action` head with a bidirectional bridge:

- `g(a_t, x_t) -> z_ctrl`
- `h(z_ctrl, x_t) -> a_t`

Train consistency between:

- posterior control latents from transitions
- action-induced control latents from labelled actions

This turns the latent into a more constrained executable interface instead of just a classifier input.

## Recommended training pipeline v2

### Stage 1: passive world-model learning

- learn posterior and prior for `z_ctrl` and `z_exo`
- train multi-step object-state rollout
- regularize latent consistency across short windows
- optional image reconstruction auxiliary

### Stage 2: labelled control grounding

- train bidirectional action-latent bridge on labelled data
- fine-tune the controllable latent path, not just the final action head
- keep the world model mostly stable, but do not freeze the entire control pathway

### Stage 3: planning

- optimize `z_ctrl` sequences only
- keep exogenous evolution under model prior
- compare at matched compute budgets across all baselines

## How the baselines should be redesigned

The current baselines are not cleanly matched.

They should be rebuilt so only one scientific axis changes at a time.

### Action-conditioned baseline

- same state/token backbone
- same dynamics depth
- direct planning in action space
- same rollout budget

### Monolithic latent baseline

- same backbone
- same latent dimensional budget
- same alignment bridge idea
- only change:
  - one scene-level control latent instead of factorised control latent

### Factorised baseline

- same everything else
- only change:
  - control latent is object-factorised

That is the only way the comparison becomes interpretable.

## How evaluation should be redesigned

### Evaluate by task family, not only mixed aggregate

Separate:

- `reach`
- `key_door`
- `push`

Then later add mixed-task evaluation.

### Use task-correct rewards

Current dense reward is not aligned to all tasks.

Redesign:

- `reach`
  - agent-to-target shaping
- `key_door`
  - key pickup bonus
  - door opening bonus
  - then target shaping
- `push`
  - block-to-target shaping
  - valid pushing contact bonus

### Add real split structure

Need explicit split families:

- seen layouts
- unseen layouts
- more distractors
- longer horizon
- new task compositions

Without this, “generalization” is not really being tested.

### Add cheap offline diagnostics

Before expensive online planning, always measure:

- one-step prediction error
- 5-step rollout error
- 10-step rollout error
- action decoding accuracy
- per-task representation probes

This makes iteration much faster.

## Practical recommendation

If the goal is to finish a strong project efficiently, the right order is:

1. rebuild the project around the state-factor track
2. make factor vs monolithic comparison clean and stable there
3. fix evaluation and planner instrumentation
4. only then reintroduce the pixel/slot track

That is the best risk-adjusted path.

## Concrete next implementation priorities

1. Reframe the main codepath as `state_factor` rather than pretending it already solves passive pixel factor discovery.
2. Replace one-step-only training with multi-step object-state rollout.
3. Add prior/posterior modeling for controllable vs exogenous latents.
4. Replace the current one-way action classifier with a bidirectional action-latent bridge.
5. Redesign evaluation around per-task metrics and explicit split banks.
6. Only after the above, revisit the slot-based model.

## Final judgment

The current architecture will probably keep producing noisy reward orderings if scaled.

It is unlikely to become a convincing factorised latent-action result just by adding more data or compute.

The project can absolutely work, but it needs a more coherent architecture than the current one:

- state-first mainline
- proper latent control prior
- explicit exogenous dynamics modeling
- matched baselines
- task-correct evaluation

That is the version worth building next.
