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

    return torch.where(
        task_id == 0,
        reach_score,
        torch.where(task_id == 1, key_door_score, push_score),
    )


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

    @torch.no_grad()
    def plan(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        factors = self.model.encode(batch["image"], batch["entity_features"], batch["entity_mask"])
        factor_mask = self.model.factor_mask(batch["entity_mask"])
        goal_vector = batch["goal_vector"]
        task_id = batch["task_id"]
        latent_dim = self.model.config.latent_action_dim
        mean = torch.zeros(self.horizon, latent_dim, device=self.device)
        std = torch.ones(self.horizon, latent_dim, device=self.device)

        for _ in range(self.iterations):
            samples = torch.randn(self.population, self.horizon, latent_dim, device=self.device) * std + mean
            scores = torch.zeros(self.population, device=self.device)
            for sample_idx in range(self.population):
                rollout_factors = factors.clone()
                predicted_entities = batch["entity_features"]
                for t in range(self.horizon):
                    rollout_factors, predicted_entities = self.model.rollout_step(
                        rollout_factors,
                        factor_mask,
                        samples[sample_idx, t].unsqueeze(0),
                    )
                scores[sample_idx] = score_goal(predicted_entities, goal_vector, task_id).squeeze()

            elite_idx = torch.topk(scores, k=min(self.elite_count, self.population)).indices
            elite = samples[elite_idx]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0).clamp_min(1e-3)

        logits = self.model.action_logits(mean[0].unsqueeze(0))
        return mean[0], torch.argmax(logits, dim=-1)


class ActionSequencePlanner:
    def __init__(self, model, horizon: int, population: int, device: str):
        self.model = model
        self.horizon = horizon
        self.population = population
        self.device = device

    @torch.no_grad()
    def plan(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        factors = self.model.encode(batch["image"], batch["entity_features"], batch["entity_mask"])
        factor_mask = self.model.factor_mask(batch["entity_mask"])
        goal_vector = batch["goal_vector"]
        task_id = batch["task_id"]

        sequences = torch.randint(
            low=0,
            high=self.model.config.num_actions,
            size=(self.population, self.horizon),
            device=self.device,
        )
        scores = torch.zeros(self.population, device=self.device)
        for sample_idx in range(self.population):
            rollout_factors = factors.clone()
            predicted_entities = batch["entity_features"]
            for t in range(self.horizon):
                rollout_factors, predicted_entities = self.model.rollout_step(
                    rollout_factors,
                    factor_mask,
                    sequences[sample_idx, t].unsqueeze(0),
                )
            scores[sample_idx] = score_goal(predicted_entities, goal_vector, task_id).squeeze()

        best_idx = torch.argmax(scores)
        return sequences[best_idx, 0].unsqueeze(0)
