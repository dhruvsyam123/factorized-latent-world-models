from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from factor_latent_wm.config.core import ModelConfig
from factor_latent_wm.models.components import (
    ControlBridge,
    EntityDecoder,
    EntityFactorEncoder,
    ImageDecoder,
    ImageEncoder,
    SpatialImageEncoder,
    SlotAttentionEncoder,
    StateDecoder,
    StateTokenEncoder,
    TokenDynamics,
    masked_mean,
)


class FactorisedLatentActionModel(nn.Module):
    """State-first object-factor world model with an optional slot-based pixel encoder."""

    def __init__(self, config: ModelConfig, image_size: int = 84):
        super().__init__()
        self.config = config
        if self.config.encoder_type == "entity":
            self.config.encoder_type = "state_factor"
        if self.config.encoder_type == "slot":
            self.config.encoder_type = "pixel_slot"
        self.image_encoder = ImageEncoder(config.image_channels, config.hidden_dim, config.factor_dim)
        self.image_decoder = ImageDecoder(config.factor_dim, config.image_channels, image_size, config.decoder_channels)
        self.state_decoder = StateDecoder(config.factor_dim, config.entity_feature_dim)

        self.state_encoder = None
        self.slot_feature_encoder = None
        self.slot_encoder = None
        self.token_count = config.max_entities
        self.anchor_mode = "agent"
        if config.encoder_type == "state_factor":
            self.state_encoder = StateTokenEncoder(config.entity_feature_dim, config.factor_dim)
        elif config.encoder_type == "pixel_slot":
            self.slot_feature_encoder = SpatialImageEncoder(config.image_channels, config.hidden_dim, config.factor_dim)
            self.slot_encoder = SlotAttentionEncoder(
                num_slots=config.max_entities,
                input_dim=config.factor_dim,
                slot_dim=config.factor_dim,
                iterations=config.slot_iterations,
                mlp_dim=config.slot_mlp_dim,
            )
            self.anchor_mode = "pooled"
        else:
            raise ValueError(f"Unknown encoder_type {config.encoder_type}")

        self.control_bridge = ControlBridge(
            factor_dim=config.factor_dim,
            latent_dim=config.latent_action_dim,
            num_actions=config.num_actions,
            hidden_dim=config.hidden_dim,
        )
        self.dynamics = TokenDynamics(config.factor_dim, config.latent_action_dim, config.num_attention_heads)

    def factor_mask(self, entity_mask: torch.Tensor) -> torch.Tensor:
        if self.config.encoder_type == "pixel_slot":
            return torch.ones(
                entity_mask.shape[0],
                self.config.max_entities,
                device=entity_mask.device,
                dtype=entity_mask.dtype,
            )
        return entity_mask

    def encode(self, image: torch.Tensor, entity_features: torch.Tensor, entity_mask: torch.Tensor) -> torch.Tensor:
        if self.config.encoder_type == "state_factor":
            assert self.state_encoder is not None
            tokens = self.state_encoder(entity_features)
            return tokens * entity_mask.unsqueeze(-1)

        assert self.slot_feature_encoder is not None
        assert self.slot_encoder is not None
        tokens = self.slot_encoder(self.slot_feature_encoder(image))
        return tokens * self.factor_mask(entity_mask).unsqueeze(-1)

    def control_anchor(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.anchor_mode == "agent":
            return tokens[:, 0]
        return masked_mean(tokens, mask)

    def infer_control_latent(self, factors: torch.Tensor, next_factors: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(factors.shape[:2], device=factors.device, dtype=factors.dtype)
        ctrl, _ = self.control_bridge.posterior(
            self.control_anchor(factors, mask),
            self.control_anchor(next_factors, mask),
            factors,
            next_factors,
        )
        return ctrl

    def infer_latent_actions(self, factors: torch.Tensor, next_factors: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(factors.shape[:2], device=factors.device, dtype=factors.dtype)
        ctrl, exo = self.control_bridge.posterior(
            self.control_anchor(factors, mask),
            self.control_anchor(next_factors, mask),
            factors,
            next_factors,
        )
        return torch.cat([ctrl, exo.flatten(1)], dim=-1)

    def prior_latents(self, factors: torch.Tensor, entity_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        factor_mask = self.factor_mask(entity_mask)
        return self.control_bridge.prior(self.control_anchor(factors, factor_mask), factors)

    def action_logits(self, control_latent: torch.Tensor) -> torch.Tensor:
        return self.control_bridge.action_logits(control_latent)

    def action_to_control(self, actions: torch.Tensor) -> torch.Tensor:
        return self.control_bridge.action_to_control(actions)

    def predict_next_factors(
        self,
        factors: torch.Tensor,
        control_latent: torch.Tensor,
        exo_latent: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> torch.Tensor:
        factor_mask = self.factor_mask(entity_mask)
        return self.dynamics(factors, control_latent, exo_latent, factor_mask)

    def decode_image(self, factors: torch.Tensor, entity_mask: torch.Tensor) -> torch.Tensor:
        factor_mask = self.factor_mask(entity_mask)
        return self.image_decoder(factors, factor_mask)

    def decode_state(self, factors: torch.Tensor) -> torch.Tensor:
        return self.state_decoder(factors)

    def rollout_step(
        self,
        factors: torch.Tensor,
        entity_mask: torch.Tensor,
        control_latent: torch.Tensor,
        exo_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        factor_mask = self.factor_mask(entity_mask)
        if exo_latent is None:
            _, exo_latent = self.prior_latents(factors, entity_mask)
        next_factors = self.dynamics(factors, control_latent, exo_latent, factor_mask)
        predicted_entities = self.decode_state(next_factors)
        return next_factors, predicted_entities

    def rollout_sequence(
        self,
        factors: torch.Tensor,
        entity_mask: torch.Tensor,
        control_sequence: torch.Tensor,
        exo_sequence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rollout_factors = factors
        predicted_entities = self.decode_state(factors)
        for step in range(control_sequence.shape[1]):
            exo_step = None if exo_sequence is None else exo_sequence[:, step]
            rollout_factors, predicted_entities = self.rollout_step(
                rollout_factors,
                entity_mask,
                control_sequence[:, step],
                exo_step,
            )
        return rollout_factors, predicted_entities

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        entity_mask = batch["entity_mask"]
        factor_mask = self.factor_mask(entity_mask)
        factors = self.encode(batch["image"], batch["entity_features"], entity_mask)
        next_factors_true = self.encode(batch["next_image"], batch["next_entity_features"], entity_mask)

        ctrl_post, exo_post = self.control_bridge.posterior(
            self.control_anchor(factors, factor_mask),
            self.control_anchor(next_factors_true, factor_mask),
            factors,
            next_factors_true,
        )
        ctrl_prior, exo_prior = self.prior_latents(factors, entity_mask)

        posterior_next_factors = self.predict_next_factors(factors, ctrl_post, exo_post, entity_mask)
        prior_next_factors = self.predict_next_factors(factors, ctrl_prior, exo_prior, entity_mask)

        return {
            "current_state_pred": self.decode_state(factors),
            "prior_current_state_pred": self.decode_state(factors),
            "current_reconstruction": self.decode_image(factors, entity_mask),
            "next_reconstruction": self.decode_image(posterior_next_factors, entity_mask),
            "prior_next_reconstruction": self.decode_image(prior_next_factors, entity_mask),
            "predicted_next_entities": self.decode_state(posterior_next_factors),
            "prior_predicted_next_entities": self.decode_state(prior_next_factors),
            "factors": factors,
            "predicted_next_factors": posterior_next_factors,
            "prior_predicted_next_factors": prior_next_factors,
            "ctrl_posterior": ctrl_post,
            "exo_posterior": exo_post,
            "ctrl_prior": ctrl_prior,
            "exo_prior": exo_prior,
            "agent_action_logits": self.action_logits(ctrl_post),
            "action_control_latent": self.action_to_control(batch["action"].clamp_min(0)),
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
