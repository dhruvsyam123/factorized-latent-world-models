from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from factor_latent_wm.config.core import ModelConfig
from factor_latent_wm.models.components import (
    EntityDecoder,
    EntityFactorEncoder,
    FactorDynamics,
    ImageDecoder,
    ImageEncoder,
    SlotAttentionEncoder,
    SpatialImageEncoder,
)


class FactorisedLatentActionModel(nn.Module):
    def __init__(self, config: ModelConfig, image_size: int = 84):
        super().__init__()
        self.config = config
        self.image_encoder = ImageEncoder(config.image_channels, config.hidden_dim, config.factor_dim)
        self.entity_encoder = None
        self.slot_feature_encoder = None
        self.slot_encoder = None
        if config.encoder_type == "entity":
            self.entity_encoder = EntityFactorEncoder(config.entity_feature_dim, config.factor_dim)
        elif config.encoder_type == "slot":
            self.slot_feature_encoder = SpatialImageEncoder(config.image_channels, config.hidden_dim, config.factor_dim)
            self.slot_encoder = SlotAttentionEncoder(
                num_slots=config.max_entities,
                input_dim=config.factor_dim,
                slot_dim=config.factor_dim,
                iterations=config.slot_iterations,
                mlp_dim=config.slot_mlp_dim,
            )
        else:
            raise ValueError(f"Unknown encoder_type {config.encoder_type}")
        self.latent_inference = nn.Sequential(
            nn.Linear(config.factor_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_action_dim),
        )
        self.dynamics = FactorDynamics(config.factor_dim, config.latent_action_dim, config.num_attention_heads)
        self.decoder = ImageDecoder(config.factor_dim, config.image_channels, image_size, config.decoder_channels)
        self.entity_decoder = EntityDecoder(config.factor_dim, config.entity_feature_dim)
        self.action_head = nn.Sequential(
            nn.Linear(config.latent_action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_actions),
        )

    def factor_mask(self, entity_mask: torch.Tensor) -> torch.Tensor:
        if self.config.encoder_type == "slot":
            return torch.ones(
                entity_mask.shape[0],
                self.config.max_entities,
                device=entity_mask.device,
                dtype=entity_mask.dtype,
            )
        return entity_mask

    def encode(self, image: torch.Tensor, entity_features: torch.Tensor, entity_mask: torch.Tensor) -> torch.Tensor:
        factor_mask = self.factor_mask(entity_mask)
        if self.config.encoder_type == "entity":
            global_context = self.image_encoder(image)
            assert self.entity_encoder is not None
            factors = self.entity_encoder(entity_features, global_context)
            return factors * factor_mask.unsqueeze(-1)

        assert self.slot_feature_encoder is not None
        assert self.slot_encoder is not None
        slots = self.slot_encoder(self.slot_feature_encoder(image))
        return slots * factor_mask.unsqueeze(-1)

    def infer_latent_actions(self, factors: torch.Tensor, next_factors: torch.Tensor) -> torch.Tensor:
        return self.latent_inference(torch.cat([factors, next_factors], dim=-1))

    def infer_control_latent(self, factors: torch.Tensor, next_factors: torch.Tensor) -> torch.Tensor:
        return self.infer_latent_actions(factors, next_factors)[:, 0]

    def predict_next_factors(
        self, factors: torch.Tensor, latent_actions: torch.Tensor, entity_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.dynamics(factors, latent_actions, entity_mask)

    def decode_image(self, factors: torch.Tensor, entity_mask: torch.Tensor) -> torch.Tensor:
        return self.decoder(factors, entity_mask)

    def predict_entities(self, factors: torch.Tensor) -> torch.Tensor:
        return self.entity_decoder(factors)

    def action_logits(self, agent_latent: torch.Tensor) -> torch.Tensor:
        return self.action_head(agent_latent)

    def rollout_step(
        self,
        factors: torch.Tensor,
        entity_mask: torch.Tensor,
        agent_latent_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        factor_mask = self.factor_mask(entity_mask)
        latents = torch.zeros(
            factors.shape[0],
            factors.shape[1],
            self.config.latent_action_dim,
            device=factors.device,
            dtype=factors.dtype,
        )
        latents[:, 0] = agent_latent_action
        next_factors = self.predict_next_factors(factors, latents, factor_mask)
        predicted_entities = self.predict_entities(next_factors)
        return next_factors, predicted_entities

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        factor_mask = self.factor_mask(batch["entity_mask"])
        factors = self.encode(batch["image"], batch["entity_features"], batch["entity_mask"])
        next_factors_true = self.encode(batch["next_image"], batch["next_entity_features"], batch["entity_mask"])
        latent_actions = self.infer_latent_actions(factors, next_factors_true)
        predicted_next_factors = self.predict_next_factors(factors, latent_actions, factor_mask)

        return {
            "current_reconstruction": self.decode_image(factors, factor_mask),
            "next_reconstruction": self.decode_image(predicted_next_factors, factor_mask),
            "predicted_next_entities": self.predict_entities(predicted_next_factors),
            "true_latent_actions": latent_actions,
            "agent_action_logits": self.action_logits(latent_actions[:, 0]),
            "factors": factors,
            "predicted_next_factors": predicted_next_factors,
            "factor_mask": factor_mask,
        }

    def save_checkpoint(self, path: str | Path, baseline: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": asdict(self.config), "baseline": baseline}, path)


def build_model(baseline: str, config: ModelConfig, image_size: int = 84) -> nn.Module:
    from factor_latent_wm.models.baselines import ActionConditionedWorldModel, MonolithicLatentWorldModel

    if baseline == "factor":
        return FactorisedLatentActionModel(config, image_size=image_size)
    if baseline == "monolithic":
        return MonolithicLatentWorldModel(config, image_size=image_size)
    if baseline == "action":
        return ActionConditionedWorldModel(config, image_size=image_size)
    raise ValueError(f"Unknown baseline {baseline}")
