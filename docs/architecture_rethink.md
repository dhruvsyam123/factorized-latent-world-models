# Architecture Rethink: Factorised Latent-Action World Model

## Executive Summary

The current repo is a useful end-to-end prototype, but it does **not** cleanly test the intended research claim:

> object-factorised latent actions learned from mostly passive video become a useful interface for control and planning in multi-entity environments

The main issue is not training scale. It is architectural mismatch. The present system mixes privileged entity-state access, weak world-model objectives, an underconstrained latent-to-action bridge, and computationally expensive evaluation. That combination can produce a runnable demo, but it is unlikely to yield an interpretable or scientifically defensible result.

The right next step is **not** more Colab sweeps. The right next step is a redesign that separates:

1. representation learning
2. controllable vs exogenous latent actions
3. action alignment
4. planning/evaluation
5. baseline parity

## What The Current System Is Actually Doing

### Current factor model

- `entity` mode encodes each factor from privileged simulator entity features plus a global image embedding.
- `slot` mode infers slots from pixels, but still uses the same downstream losses and control bridge.
- Stage 1 trains on one-step reconstruction and privileged next-entity prediction.
- Stage 2 freezes most of the model and trains an action classifier on the inferred agent latent.
- Evaluation plans by sampling arbitrary latent sequences with CEM, then maps the first latent to a discrete action.

### What this means scientifically

- In `entity` mode, the model is **not** learning factors from passive video alone. It is being handed object-structured state.
- In both `entity` and `slot` modes, the control bridge is trained only as a posterior action decoder, not as a valid control prior.
- The planner samples latents from a simple Gaussian, not from a learned controllable latent prior.
- Baseline comparison is not fully apples-to-apples.

## Top Architectural Problems

### 1. The factorised claim is short-circuited by privileged entity inputs

In `entity` mode, the factor encoder is an MLP over simulator entity features with a global image embedding added on top. This means the model is not discovering object structure from passive video; it is consuming pre-segmented object descriptors.

Why it matters:

- It makes the factorised result much easier than the claimed problem.
- It blurs whether gains come from factorisation or from privileged structure.
- It weakens any claim about learning latent actions from passive video.

Minimum redesign:

- Split the project into two explicit tracks:
  - `state-factor` track: privileged object-state world model for algorithm development
  - `pixel-factor` track: pixel-to-slot/object representation for the real research claim
- Do not present the state-factor track as the main result. Present it as a debugging scaffold.

### 2. The latent-to-action bridge is underconstrained and likely invalid for planning

The model infers a posterior latent from `(s_t, s_{t+1})`, then Stage 2 trains an action head to predict the real action from that inferred latent. At test time, planning samples arbitrary latent vectors and feeds them through the same action head.

Why it matters:

- The action head is only trained on posterior latents that correspond to real transitions.
- The planner searches a latent space with no learned support constraint.
- The chosen latent may decode to an action label while still being off-manifold for the dynamics model.

Minimum redesign:

- Learn both:
  - a posterior `q(z_ctrl | o_t, o_{t+1})`
  - a prior `p(z_ctrl | o_t)` or state-conditional proposal model
- Train the action bridge on latents sampled from the posterior and regularized toward the prior.
- During planning, sample only from the learned prior or a constrained proposal distribution.
- Keep the controllable latent separate from exogenous latent variables.

### 3. Exogenous dynamics and controllable dynamics are not properly separated

The current factor model predicts a per-factor latent from state differences, but at planning time only the first factor receives a planned latent and every other factor latent is set to zero. This implicitly assumes the model can simulate exogenous factors without an explicit exogenous latent process.

Why it matters:

- It conflates "agent chooses an action" with "world factors evolve stochastically or autonomously."
- It gives the planner an unrealistic rollout model for hazards, distractors, and interactions.
- It prevents a clean causal interpretation of controllable vs non-controllable changes.

Minimum redesign:

- Use two latent channels:
  - `z_ctrl` for the agent-controllable factor
  - `z_exo` for exogenous or interaction-driven changes
- Train a posterior over both on passive data.
- At planning time:
  - optimize only `z_ctrl`
  - propagate `z_exo` via a learned prior or latent predictive model

### 4. Baselines are not matched in a scientifically fair way

The monolithic latent baseline was retrofitted with a control-alignment head, while the action-conditioned baseline is evaluated with a different planning interface. The three systems do not share a truly matched training/evaluation protocol.

Why it matters:

- Differences in performance may come from interface choices rather than factorisation.
- The action-conditioned baseline becomes a weaker-than-necessary comparator.
- The monolithic baseline is not guaranteed to be the exact non-factorised analogue of the factor model.

Minimum redesign:

- Define a common template:
  - same encoder backbone family
  - same dynamics depth
  - same rollout loss family
  - same planner budget
- Then vary only one axis at a time:
  - factorised vs monolithic latent
  - privileged factors vs slot factors
  - labelled-data amount
- For the action-conditioned baseline, plan directly in action space with the same rollout budget.

### 5. The world-model objective is too weak for the claim

Stage 1 currently uses current-image reconstruction, next-image reconstruction, and privileged next-entity MSE. There is no multi-step latent consistency objective, no prior matching, no temporal persistence constraint, and no explicit factor assignment mechanism.

Why it matters:

- One-step prediction can look fine while long-horizon planning fails.
- The latent action may not become stable or reusable as a control interface.
- Slots/factors can drift over time without penalty.

Minimum redesign:

- Add multi-step rollout loss over short imagined horizons.
- Add a latent prior / posterior regularization term.
- Add factor persistence or assignment consistency over time.
- For the pixel track, include slot/object matching consistency across adjacent frames.

## Why The Current Evaluation Is Misleadingly Expensive

The evaluator replans at every environment step using nested Python loops over:

- episodes
- environment steps
- CEM iterations
- population samples
- horizon steps

This makes runtime explode while still producing weak diagnostics.

Why it matters:

- Slow evaluation encourages tiny experiments with noisy conclusions.
- No per-task breakdown means failure modes are hidden.
- Progress is invisible, so it is hard to distinguish slow from broken.

Recommended redesign:

- Use batched planning instead of per-sample Python loops.
- Add progress bars and timing breakdowns.
- Evaluate tasks separately: `reach`, `key_door`, `push`.
- Start with horizon 3 to 5 for development, then scale.

## Recommended Architecture V2

### Track A: Privileged object-state scaffold

Purpose:

- Debug the algorithmic idea without representation learning confounds.

Representation:

- One token per object from simulator state only.

Model:

- factorised object tokens
- posterior `q(z_ctrl, z_exo | x_t, x_{t+1})`
- prior `p(z_ctrl, z_exo | x_t)`
- token dynamics `p(x_{t+1} | x_t, z_ctrl, z_exo)`

Training:

- passive + labelled data together for posterior/dynamics learning
- action alignment on labelled subset only
- multi-step rollout loss on latent dynamics

Use:

- establish whether factorised controllable latents help at all

### Track B: Pixel object-centric model

Purpose:

- Test the real passive-video factorisation claim.

Representation:

- slot/object encoder from pixels only
- slot persistence / matching across time

Model:

- same latent structure as Track A after the encoder

Training:

- first train the state-factor track to validate the algorithm
- then port the same latent structure to the pixel encoder

Use:

- paper-quality claim about factorisation from pixels

## Recommended Training Pipeline V2

### Stage 1: Passive world-model learning

- learn posterior and prior over controllable/exogenous latents
- learn multi-step factor dynamics
- learn reconstruction or state prediction
- for pixels, learn slot persistence as well

### Stage 2: Control grounding

- train `p(a_t | z_ctrl, x_t)` on labelled data
- optionally train `q(z_ctrl | x_t, a_t)` for consistency
- do not rely on a pure frozen-classifier bridge alone

### Stage 3: Planning

- optimize only controllable latent sequences
- keep exogenous dynamics under learned prior
- compare against:
  - monolithic latent planner
  - action-space planner on action-conditioned model

## Recommended Experimental Order

### Phase 1: Make the algorithm work on privileged object state

Success criterion:

- factorised latent beats monolithic latent on at least one multi-object task
- performance improves with more labelled data

### Phase 2: Add rigorous evaluation

- per-task success
- held-out layouts
- more distractors
- longer horizons

### Phase 3: Port to pixels / slots

Success criterion:

- some retained advantage from factorisation survives the harder representation problem

## Concrete Next Implementation Priorities

1. Separate the repo into `state-factor` and `pixel-factor` modes explicitly.
2. Replace the current single posterior-only action bridge with prior + posterior controllable latent modeling.
3. Split controllable and exogenous latent channels.
4. Rebuild the baseline suite so planning interfaces are matched.
5. Replace the evaluator with batched planning and per-task metrics before running larger sweeps.

## Bottom Line

Will the current code work as a clean research demonstration?

Not for the full claim.

It can work as:

- a prototype
- a debugging scaffold
- a demo of an end-to-end latent-control loop

But it is not yet the right architecture for defending the intended factorised latent-action result. To get there, the next version needs a cleaner causal decomposition, a valid control prior, better baseline parity, and a sharper separation between privileged-state debugging and pixel-based scientific evaluation.
