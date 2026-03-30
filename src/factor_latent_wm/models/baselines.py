from __future__ import annotations

import torch
from torch import nn

from factor_latent_wm.config.core import ModelConfig
from factor_latent_wm.models.components import EntityDecoder, EntityFactorEncoder, FactorDynamics, ImageDecoder, ImageEncoder


class MonolithicLatentWorldModel(nn.Module):
    def __init__(self, config: ModelConfig, image_size: int = 84):
        super().__init__()
        self.config = config
        self.image_encoder = ImageEncoder(config.image_channels, config.hidden_dim, config.factor_dim)
        self.entity_encoder = EntityFactorEncoder(config.entity_feature_dim, config.factor_dim)
        self.scene_latent = nn.Sequential(
            nn.Linear(config.factor_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_action_dim),
        )
        self.scene_to_factor = nn.Linear(config.latent_action_dim, config.factor_dim)
        self.dynamics = FactorDynamics(config.factor_dim, config.factor_dim, config.num_attention_heads)
        self.decoder = ImageDecoder(config.factor_dim, config.image_channels, image_size, config.decoder_channels)
        self.entity_decoder = EntityDecoder(config.factor_dim, config.entity_feature_dim)
        self.action_head = nn.Sequential(
            nn.Linear(config.latent_action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_actions),
        )

    def factor_mask(self, entity_mask: torch.Tensor) -> torch.Tensor:
        return entity_mask

    def encode(self, image: torch.Tensor, entity_features: torch.Tensor, entity_mask: torch.Tensor) -> torch.Tensor:
        global_context = self.image_encoder(image)
        factors = self.entity_encoder(entity_features, global_context)
        return factors * entity_mask.unsqueeze(-1)

    def infer_control_latent(self, factors: torch.Tensor, next_factors: torch.Tensor) -> torch.Tensor:
        scene = (factors).sum(dim=1)
        next_scene = (next_factors).sum(dim=1)
        return self.scene_latent(torch.cat([scene, next_scene], dim=-1))

    def action_logits(self, latent: torch.Tensor) -> torch.Tensor:
        return self.action_head(latent)

    def rollout_step(
        self,
        factors: torch.Tensor,
        entity_mask: torch.Tensor,
        control_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_per_factor = self.scene_to_factor(control_latent).unsqueeze(1).expand(-1, factors.shape[1], -1)
        next_factors = self.dynamics(factors, latent_per_factor, entity_mask)
        predicted_entities = self.entity_decoder(next_factors)
        return next_factors, predicted_entities

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        factors = self.encode(batch["image"], batch["entity_features"], batch["entity_mask"])
        next_factors = self.encode(batch["next_image"], batch["next_entity_features"], batch["entity_mask"])
        latent = self.infer_control_latent(factors, next_factors)
        predicted_next_factors, predicted_entities = self.rollout_step(factors, batch["entity_mask"], latent)
        return {
            "current_reconstruction": self.decoder(factors, batch["entity_mask"]),
            "next_reconstruction": self.decoder(predicted_next_factors, batch["entity_mask"]),
            "predicted_next_entities": predicted_entities,
            "agent_action_logits": self.action_logits(latent),
        }


class ActionConditionedWorldModel(nn.Module):
    def __init__(self, config: ModelConfig, image_size: int = 84):
        super().__init__()
        self.config = config
        self.image_encoder = ImageEncoder(config.image_channels, config.hidden_dim, config.factor_dim)
        self.entity_encoder = EntityFactorEncoder(config.entity_feature_dim, config.factor_dim)
        self.action_embedding = nn.Embedding(config.num_actions, config.factor_dim)
        self.dynamics = FactorDynamics(config.factor_dim, config.factor_dim, config.num_attention_heads)
        self.decoder = ImageDecoder(config.factor_dim, config.image_channels, image_size, config.decoder_channels)
        self.entity_decoder = EntityDecoder(config.factor_dim, config.entity_feature_dim)

    def factor_mask(self, entity_mask: torch.Tensor) -> torch.Tensor:
        return entity_mask

    def encode(self, image: torch.Tensor, entity_features: torch.Tensor, entity_mask: torch.Tensor) -> torch.Tensor:
        global_context = self.image_encoder(image)
        factors = self.entity_encoder(entity_features, global_context)
        return factors * entity_mask.unsqueeze(-1)

    def rollout_step(
        self,
        factors: torch.Tensor,
        entity_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_embed = self.action_embedding(actions.clamp_min(0))
        latent = torch.zeros_like(factors)
        latent[:, 0, :] = action_embed
        predicted_next_factors = self.dynamics(factors, latent, entity_mask)
        predicted_entities = self.entity_decoder(predicted_next_factors)
        return predicted_next_factors, predicted_entities

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        factors = self.encode(batch["image"], batch["entity_features"], batch["entity_mask"])
        predicted_next_factors, predicted_entities = self.rollout_step(factors, batch["entity_mask"], batch["action"])
        return {
            "current_reconstruction": self.decoder(factors, batch["entity_mask"]),
            "next_reconstruction": self.decoder(predicted_next_factors, batch["entity_mask"]),
            "predicted_next_entities": predicted_entities,
        }
