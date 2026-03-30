from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from factor_latent_wm.config.core import EnvConfig, EvalConfig, ModelConfig, TrainConfig
from factor_latent_wm.data.dataset import SequenceDataset, TransitionDataset
from factor_latent_wm.envs.multi_object_env import MultiObjectEnv
from factor_latent_wm.models.factor_model import build_model
from factor_latent_wm.planning.cem import ActionSequencePlanner, LatentCEMPlanner
from factor_latent_wm.training.losses import alignment_loss, world_model_loss


def _to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _make_loader(path: str, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    dataset = TransitionDataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _make_sequence_loader(path: str, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    dataset = SequenceDataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _sequence_step_outputs(model, batch: dict[str, torch.Tensor], rollout_factors: torch.Tensor, step: int, baseline: str):
    image = batch["frames"][:, step]
    next_image = batch["frames"][:, step + 1]
    entity_features = batch["state_tokens"][:, step]
    next_entity_features = batch["state_tokens"][:, step + 1]
    entity_mask = batch["state_masks"][:, step]
    action = batch["actions"][:, step].clamp_min(0)
    labelled = batch["labelled_mask"][:, step]

    next_true_factors = model.encode(next_image, next_entity_features, entity_mask)
    factor_mask = model.factor_mask(entity_mask)

    if baseline == "action":
        ctrl_post = model.action_to_control(action)
        ctrl_prior, exo_prior = model.prior_latents(rollout_factors, entity_mask)
        _, exo_post = model.control_bridge.posterior(
            model.control_anchor(rollout_factors, factor_mask),
            model.control_anchor(next_true_factors, factor_mask),
            rollout_factors,
            next_true_factors,
        )
        posterior_next_factors, posterior_entities = model.rollout_step(rollout_factors, entity_mask, action, exo_post)
        prior_action = torch.argmax(model.action_logits(ctrl_prior), dim=-1)
        prior_next_factors, prior_entities = model.rollout_step(rollout_factors, entity_mask, prior_action, exo_prior)
    else:
        ctrl_post, exo_post = model.control_bridge.posterior(
            model.control_anchor(rollout_factors, factor_mask),
            model.control_anchor(next_true_factors, factor_mask),
            rollout_factors,
            next_true_factors,
        )
        ctrl_prior, exo_prior = model.prior_latents(rollout_factors, entity_mask)
        posterior_next_factors, posterior_entities = model.rollout_step(
            rollout_factors,
            entity_mask,
            ctrl_post,
            exo_post,
        )
        prior_next_factors, prior_entities = model.rollout_step(
            rollout_factors,
            entity_mask,
            ctrl_prior,
            exo_prior,
        )

    outputs = {
        "current_state_pred": model.decode_state(rollout_factors),
        "prior_current_state_pred": model.decode_state(rollout_factors),
        "current_reconstruction": model.decode_image(rollout_factors, entity_mask),
        "next_reconstruction": model.decode_image(posterior_next_factors, entity_mask),
        "prior_next_reconstruction": model.decode_image(prior_next_factors, entity_mask),
        "predicted_next_entities": posterior_entities,
        "prior_predicted_next_entities": prior_entities,
        "factors": rollout_factors,
        "predicted_next_factors": posterior_next_factors,
        "prior_predicted_next_factors": prior_next_factors,
        "ctrl_posterior": ctrl_post,
        "exo_posterior": exo_post,
        "ctrl_prior": ctrl_prior,
        "exo_prior": exo_prior,
        "agent_action_logits": model.action_logits(ctrl_post),
        "action_control_latent": model.action_to_control(action),
        "factor_mask": factor_mask,
    }
    step_batch = {
        "image": image,
        "next_image": next_image,
        "entity_features": entity_features,
        "next_entity_features": next_entity_features,
        "entity_mask": entity_mask,
        "goal_vector": batch["goal_vectors"][:, step],
        "task_id": batch["task_id"],
        "action": action,
        "labelled": labelled,
    }
    return outputs, step_batch, posterior_next_factors


def _sequence_training_loss(model, batch: dict[str, torch.Tensor], baseline: str, config: ModelConfig, rollout_horizon: int):
    horizon = min(rollout_horizon, batch["frames"].shape[1] - 1, int(batch["trajectory_length"].min().item()))
    if horizon <= 0:
        zero = torch.zeros((), device=batch["frames"].device)
        return zero, {"loss": 0.0}

    rollout_factors = model.encode(batch["frames"][:, 0], batch["state_tokens"][:, 0], batch["state_masks"][:, 0])
    total_loss = torch.zeros((), device=batch["frames"].device)
    metrics: dict[str, float] = {"loss": 0.0}
    for step in range(horizon):
        outputs, step_batch, rollout_factors = _sequence_step_outputs(model, batch, rollout_factors, step, baseline)
        step_loss, step_metrics = world_model_loss(
            outputs,
            step_batch,
            config.reconstruction_weight,
            config.entity_weight,
            config.latent_weight,
        )
        if baseline == "action":
            step_loss = step_loss + alignment_loss(outputs, step_batch)
        total_loss = total_loss + step_loss
        for key, value in step_metrics.items():
            metrics[key] = metrics.get(key, 0.0) + value

    total_loss = total_loss / float(horizon)
    for key in list(metrics.keys()):
        metrics[key] /= float(horizon)
    metrics["loss"] = float(total_loss.detach().cpu())
    return total_loss, metrics


def _evaluate_epoch(model, loader: DataLoader, device: str, config: ModelConfig, baseline: str) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "batches": 0}
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            if "frames" in batch:
                loss, _ = _sequence_training_loss(model, batch, baseline, config, rollout_horizon=5)
            else:
                outputs = model(batch)
                loss, _ = world_model_loss(
                    outputs,
                    batch,
                    config.reconstruction_weight,
                    config.entity_weight,
                    config.latent_weight,
                )
            totals["loss"] += float(loss.detach().cpu())
            totals["batches"] += 1
    return {"val_loss": totals["loss"] / max(1, totals["batches"])}


def train_stage1(
    train_path: str,
    val_path: str,
    output_path: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    baseline: str = "factor",
) -> Path:
    device = train_config.device
    model = build_model(baseline, model_config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    train_loader = _make_sequence_loader(train_path, train_config.batch_size, train_config.num_workers, True)
    val_loader = _make_sequence_loader(val_path, train_config.batch_size, train_config.num_workers, False)
    best_state = None
    best_loss = float("inf")

    for _ in tqdm(range(train_config.stage1_epochs), desc=f"stage1-{baseline}"):
        model.train()
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _sequence_training_loss(
                model,
                batch,
                baseline,
                model_config,
                train_config.rollout_horizon,
            )
            loss.backward()
            optimizer.step()
        metrics = _evaluate_epoch(model, val_loader, device, model_config, baseline)
        if metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    checkpoint = {
        "state_dict": best_state or model.state_dict(),
        "config": asdict(model_config),
        "baseline": baseline,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return output


def train_stage2(
    checkpoint_path: str,
    labelled_path: str,
    output_path: str,
    train_config: TrainConfig,
) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = ModelConfig(**checkpoint["config"])
    baseline = checkpoint["baseline"]
    if baseline not in {"factor", "monolithic"}:
        raise ValueError("Stage 2 alignment is only defined for factor and monolithic latent models.")
    model = build_model(baseline, model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(train_config.device)

    if train_config.freeze_world_model_stage2:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("control_bridge")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_config.learning_rate)
    loader = _make_loader(labelled_path, train_config.batch_size, train_config.num_workers, True)

    model.train()
    for _ in tqdm(range(train_config.stage2_epochs), desc=f"stage2-{baseline}"):
        for batch in loader:
            batch = _to_device(batch, train_config.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss = alignment_loss(outputs, batch)
            loss.backward()
            optimizer.step()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": asdict(model_config), "baseline": baseline}, output)
    return output


def evaluate_checkpoint(
    checkpoint_path: str,
    env_config: EnvConfig,
    eval_config: EvalConfig,
    device: str,
) -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = ModelConfig(**checkpoint["config"])
    model = build_model(checkpoint["baseline"], model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    env = MultiObjectEnv(env_config, seed=0)
    successes = 0
    rewards = []
    by_task = defaultdict(lambda: {"success": 0, "count": 0, "reward": 0.0})
    planner = None
    if checkpoint["baseline"] in {"factor", "monolithic"}:
        planner = LatentCEMPlanner(
            model,
            horizon=eval_config.planner_horizon,
            population=eval_config.planner_population,
            elite_count=eval_config.planner_elite_count,
            iterations=eval_config.planner_iterations,
            device=device,
        )
    elif checkpoint["baseline"] == "action":
        planner = ActionSequencePlanner(
            model,
            horizon=eval_config.planner_horizon,
            population=eval_config.planner_population,
            iterations=eval_config.planner_iterations,
            device=device,
        )

    for episode in range(eval_config.episodes):
        obs, info = env.reset(seed=episode)
        done = False
        episode_reward = 0.0
        while not done:
            batch = {
                "image": torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0,
                "entity_features": torch.from_numpy(info["entity_features"]).unsqueeze(0).float().to(device),
                "entity_mask": torch.from_numpy(info["entity_mask"]).unsqueeze(0).float().to(device),
                "goal_vector": torch.from_numpy(info["goal_vector"]).unsqueeze(0).float().to(device),
                "task_id": torch.tensor([info["task_id"]], dtype=torch.long, device=device),
            }
            if planner is None:
                chosen_action = int(env.action_space.sample())
            else:
                chosen_action = int(planner.plan(batch).item())
            obs, reward, terminated, truncated, info = env.step(chosen_action)
            episode_reward += reward
            done = terminated or truncated

        successes += int(info["success"])
        rewards.append(episode_reward)
        task_stats = by_task[info["task"]]
        task_stats["success"] += int(info["success"])
        task_stats["count"] += 1
        task_stats["reward"] += episode_reward

    metrics = {
        "success_rate": successes / max(1, eval_config.episodes),
        "mean_reward": sum(rewards) / max(1, len(rewards)),
    }
    for task, stats in by_task.items():
        metrics[f"{task}_success_rate"] = stats["success"] / max(1, stats["count"])
        metrics[f"{task}_mean_reward"] = stats["reward"] / max(1, stats["count"])
    return metrics
