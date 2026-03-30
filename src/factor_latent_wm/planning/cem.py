from __future__ import annotations

import torch


def score_goal(predicted_entities: torch.Tensor, goal_vector: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
    agent_pos = predicted_entities[:, 0, 7:9]
    block_pos = predicted_entities[:, 3, 7:9]
    target_pos = goal_vector[:, :2]
    door_pos = goal_vector[:, 4:6]

    reach_score = -torch.norm(agent_pos - target_pos, p=1, dim=-1)
    key_door_score = -torch.norm(agent_pos - target_pos, p=1, dim=-1) - 0.5 * torch.norm(agent_pos - door_pos, p=1, dim=-1)
    push_score = -torch.norm(block_pos - target_pos, p=1, dim=-1)

    return torch.where(task_id == 0, reach_score, torch.where(task_id == 1, key_door_score, push_score))


class LatentCEMPlanner:
    def __init__(
        self,
        model,
        horizon: int,
        population: int,
        elite_count: int,
        iterations: int,
        device: str,
    ):
        self.model = model
        self.horizon = horizon
        self.population = population
        self.elite_count = elite_count
        self.iterations = iterations
        self.device = device

    @torch.inference_mode()
    def plan(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        factors = self.model.encode(batch["image"], batch["entity_features"], batch["entity_mask"])
        factor_mask = self.model.factor_mask(batch["entity_mask"])
        goal_vector = batch["goal_vector"]
        task_id = batch["task_id"]
        latent_dim = self.model.config.latent_action_dim
        mean = torch.zeros(self.horizon, latent_dim, device=self.device)
        std = torch.ones(self.horizon, latent_dim, device=self.device)

        factors = factors.expand(self.population, -1, -1).contiguous()
        factor_mask = factor_mask.expand(self.population, -1).contiguous()
        goal_vector = goal_vector.expand(self.population, -1).contiguous()
        task_id = task_id.expand(self.population).contiguous()

        for _ in range(self.iterations):
            samples = torch.randn(self.population, self.horizon, latent_dim, device=self.device) * std + mean
            rollout = factors
            predicted_entities = self.model.decode_state(rollout)
            for t in range(self.horizon):
                rollout, predicted_entities = self.model.rollout_step(rollout, factor_mask, samples[:, t])
            scores = score_goal(predicted_entities, goal_vector, task_id)
            elite_idx = torch.topk(scores, k=min(self.elite_count, self.population)).indices
            elite = samples[elite_idx]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0).clamp_min(1e-3)

        best_action = torch.argmax(self.model.action_logits(mean[0].unsqueeze(0)), dim=-1)
        return best_action


class ActionSequencePlanner:
    def __init__(self, model, horizon: int, population: int, iterations: int, device: str):
        self.model = model
        self.horizon = horizon
        self.population = population
        self.iterations = iterations
        self.device = device

    @torch.inference_mode()
    def plan(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        factors = self.model.encode(batch["image"], batch["entity_features"], batch["entity_mask"])
        factor_mask = self.model.factor_mask(batch["entity_mask"])
        goal_vector = batch["goal_vector"]
        task_id = batch["task_id"]
        num_actions = self.model.config.num_actions

        factors = factors.expand(self.population, -1, -1).contiguous()
        factor_mask = factor_mask.expand(self.population, -1).contiguous()
        goal_vector = goal_vector.expand(self.population, -1).contiguous()
        task_id = task_id.expand(self.population).contiguous()

        probs = torch.full((self.horizon, num_actions), 1.0 / num_actions, device=self.device)
        for _ in range(self.iterations):
            samples = torch.stack(
                [torch.multinomial(probs[t], self.population, replacement=True) for t in range(self.horizon)],
                dim=1,
            )
            rollout = factors
            predicted_entities = self.model.decode_state(rollout)
            for t in range(self.horizon):
                rollout, predicted_entities = self.model.rollout_step(rollout, factor_mask, samples[:, t])
            scores = score_goal(predicted_entities, goal_vector, task_id)
            elite_idx = torch.topk(scores, k=min(self.population, max(1, self.population // 4))).indices
            elite = samples[elite_idx]
            for t in range(self.horizon):
                counts = torch.bincount(elite[:, t], minlength=num_actions).float()
                probs[t] = (counts + 1.0) / (counts.sum() + float(num_actions))

        best_action = torch.argmax(probs[0], dim=-1)
        return best_action

