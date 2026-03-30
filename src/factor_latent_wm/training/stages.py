from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from factor_latent_wm.config.core import EnvConfig, EvalConfig, ModelConfig, TrainConfig
from factor_latent_wm.data.dataset import TransitionDataset
from factor_latent_wm.envs.multi_object_env import MultiObjectEnv
from factor_latent_wm.models.factor_model import build_model
from factor_latent_wm.planning.cem import ActionSequencePlanner, LatentCEMPlanner
from factor_latent_wm.training.losses import alignment_loss, world_model_loss


def _to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _make_loader(path: str, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    dataset = TransitionDataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _evaluate_epoch(model, loader: DataLoader, device: str, config: ModelConfig) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "batches": 0}
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            outputs = model(batch)
            loss, _ = world_model_loss(outputs, batch, config.reconstruction_weight, config.entity_weight)
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
    train_loader = _make_loader(train_path, train_config.batch_size, train_config.num_workers, True)
    val_loader = _make_loader(val_path, train_config.batch_size, train_config.num_workers, False)
    best_state = None
    best_loss = float("inf")

    for _ in tqdm(range(train_config.stage1_epochs), desc=f"stage1-{baseline}"):
        model.train()
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, _ = world_model_loss(outputs, batch, model_config.reconstruction_weight, model_config.entity_weight)
            loss.backward()
            optimizer.step()
        metrics = _evaluate_epoch(model, val_loader, device, model_config)
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
            param.requires_grad = name.startswith("action_head")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_config.learning_rate)
    loader = _make_loader(labelled_path, train_config.batch_size, train_config.num_workers, True)

    model.train()
    for _ in tqdm(range(train_config.stage2_epochs), desc=f"stage2-{baseline}"):
        for batch in loader:
            batch = _to_device(batch, train_config.device)
            optimizer.zero_grad(set_to_none=True)
            factors = model.encode(batch["image"], batch["entity_features"], batch["entity_mask"])
            next_factors = model.encode(batch["next_image"], batch["next_entity_features"], batch["entity_mask"])
            control_latent = model.infer_control_latent(factors, next_factors)
            logits = model.action_logits(control_latent)
            loss = alignment_loss(logits, batch["action"], batch["labelled"])
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
            device=device,
        )

    for episode in range(eval_config.episodes):
        obs, info = env.reset(seed=episode)
        done = False
        episode_reward = 0.0
        while not done:
            if isinstance(planner, LatentCEMPlanner):
                batch = {
                    "image": torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0,
                    "entity_features": torch.from_numpy(info["entity_features"]).unsqueeze(0).float().to(device),
                    "entity_mask": torch.from_numpy(info["entity_mask"]).unsqueeze(0).float().to(device),
                    "goal_vector": torch.from_numpy(info["goal_vector"]).unsqueeze(0).float().to(device),
                    "task_id": torch.tensor([info["task_id"]], dtype=torch.long, device=device),
                }
                _, action = planner.plan(batch)
                chosen_action = int(action.item())
            elif isinstance(planner, ActionSequencePlanner):
                batch = {
                    "image": torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0,
                    "entity_features": torch.from_numpy(info["entity_features"]).unsqueeze(0).float().to(device),
                    "entity_mask": torch.from_numpy(info["entity_mask"]).unsqueeze(0).float().to(device),
                    "goal_vector": torch.from_numpy(info["goal_vector"]).unsqueeze(0).float().to(device),
                    "task_id": torch.tensor([info["task_id"]], dtype=torch.long, device=device),
                }
                action = planner.plan(batch)
                chosen_action = int(action.item())
            else:
                chosen_action = int(env.action_space.sample())
            obs, reward, terminated, truncated, info = env.step(chosen_action)
            episode_reward += reward
            done = terminated or truncated
        successes += int(info["success"])
        rewards.append(episode_reward)

    return {
        "success_rate": successes / max(1, eval_config.episodes),
        "mean_reward": sum(rewards) / max(1, len(rewards)),
    }
