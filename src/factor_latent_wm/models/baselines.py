from __future__ import annotations

import torch
from torch import nn

from factor_latent_wm.config.core import ModelConfig
from factor_latent_wm.models.components import (
    ControlBridge,
    ImageDecoder,
    ImageEncoder,
    SpatialImageEncoder,
    SlotAttentionEncoder,
    StateDecoder,
    StateTokenEncoder,
    TokenDynamics,
    masked_mean,
)


class _TokenWorldModelBase(nn.Module):
    def __init__(self, config: ModelConfig, image_size: int = 84):
        super().__init__()
        self.config = config
        if self.config.encoder_type == "entity":
            self.config.encoder_type = "state_factor"
        if self.config.encoder_type == "slot":
            self.config.encoder_type = "pixel_slot"
        self.token_count = config.max_entities
        self.anchor_mode = "agent"
        self.image_encoder = ImageEncoder(config.image_channels, config.hidden_dim, config.factor_dim)
        self.image_decoder = ImageDecoder(config.factor_dim, config.image_channels, image_size, config.decoder_channels)
        self.state_decoder = StateDecoder(config.factor_dim, config.entity_feature_dim)
        self.state_encoder = None
        self.slot_feature_encoder = None
        self.slot_encoder = None
        if config.encoder_type == "state_factor":
            self.state_encoder = StateTokenEncoder(config.entity_feature_dim, config.factor_dim)
        elif config.encoder_type == "pixel_slot":
            self.slot_feature_encoder = SpatialImageEncoder(
                config.image_channels,
                config.hidden_dim,
                config.factor_dim,
            )
            self.slot_encoder = SlotAttentionEncoder(
                num_slots=config.max_entities,
                input_dim=config.factor_dim,
                slot_dim=config.factor_dim,
                iterations=config.slot_iterations,
                mlp_dim=config.slot_mlp_dim,
            )
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

    def prior_latents(self, factors: torch.Tensor, entity_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        factor_mask = self.factor_mask(entity_mask)
        return self.control_bridge.prior(self.control_anchor(factors, factor_mask), factors)

    def infer_control_latent(self, factors: torch.Tensor, next_factors: torch.Tensor) -> torch.Tensor:
        factor_mask = torch.ones(factors.shape[:2], device=factors.device, dtype=factors.dtype)
        ctrl, _ = self.control_bridge.posterior(
            self.control_anchor(factors, factor_mask),
            self.control_anchor(next_factors, factor_mask),
            factors,
            next_factors,
        )
        return ctrl

    def infer_latent_actions(self, factors: torch.Tensor, next_factors: torch.Tensor) -> torch.Tensor:
        factor_mask = torch.ones(factors.shape[:2], device=factors.device, dtype=factors.dtype)
        ctrl, exo = self.control_bridge.posterior(
            self.control_anchor(factors, factor_mask),
            self.control_anchor(next_factors, factor_mask),
            factors,
            next_factors,
        )
        return torch.cat([ctrl, exo.flatten(1)], dim=-1)

    def action_logits(self, control_latent: torch.Tensor) -> torch.Tensor:
        return self.control_bridge.action_logits(control_latent)

    def action_to_control(self, actions: torch.Tensor) -> torch.Tensor:
        return self.control_bridge.action_to_control(actions)

    def decode_image(self, tokens: torch.Tensor, entity_mask: torch.Tensor) -> torch.Tensor:
        return self.image_decoder(tokens, self.factor_mask(entity_mask))

    def decode_state(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.state_decoder(tokens)

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
        return next_factors, self.decode_state(next_factors)

    def rollout_sequence(
        self,
        factors: torch.Tensor,
        entity_mask: torch.Tensor,
        control_sequence: torch.Tensor,
        exo_sequence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = factors
        decoded = self.decode_state(factors)
        for step in range(control_sequence.shape[1]):
            exo = None if exo_sequence is None else exo_sequence[:, step]
            current, decoded = self.rollout_step(current, entity_mask, control_sequence[:, step], exo)
        return current, decoded

    def _forward_common(self, batch: dict[str, torch.Tensor], predicted_next: torch.Tensor, prior_next: torch.Tensor, ctrl_post: torch.Tensor, exo_post: torch.Tensor, ctrl_prior: torch.Tensor, exo_prior: torch.Tensor, factors: torch.Tensor, factor_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "current_state_pred": self.decode_state(factors),
            "prior_current_state_pred": self.decode_state(factors),
            "current_reconstruction": self.decode_image(factors, batch["entity_mask"]),
            "next_reconstruction": self.decode_image(predicted_next, batch["entity_mask"]),
            "prior_next_reconstruction": self.decode_image(prior_next, batch["entity_mask"]),
            "predicted_next_entities": self.decode_state(predicted_next),
            "prior_predicted_next_entities": self.decode_state(prior_next),
            "factors": factors,
            "predicted_next_factors": predicted_next,
            "prior_predicted_next_factors": prior_next,
            "ctrl_posterior": ctrl_post,
            "exo_posterior": exo_post,
            "ctrl_prior": ctrl_prior,
            "exo_prior": exo_prior,
            "agent_action_logits": self.action_logits(ctrl_post),
            "action_control_latent": self.action_to_control(batch["action"].clamp_min(0)),
            "factor_mask": factor_mask,
        }


class MonolithicLatentWorldModel(_TokenWorldModelBase):
    def __init__(self, config: ModelConfig, image_size: int = 84):
        super().__init__(config, image_size=image_size)
        self.anchor_mode = "pooled"
        self.scene_decoder = StateDecoder(config.factor_dim, config.entity_feature_dim)

    def control_anchor(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_mean(tokens, mask)

    def prior_latents(self, factors: torch.Tensor, entity_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        factor_mask = self.factor_mask(entity_mask)
        scene = self.control_anchor(factors, factor_mask)
        ctrl, exo = self.control_bridge.prior(scene, factors)
        scene_exo = masked_mean(exo, factor_mask).unsqueeze(1).expand(-1, factors.shape[1], -1)
        return ctrl, scene_exo

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
        if exo_latent.dim() == 2:
            exo_latent = exo_latent.unsqueeze(1).expand(-1, factors.shape[1], -1)
        next_factors = self.dynamics(factors, control_latent, exo_latent, factor_mask)
        return next_factors, self.decode_state(next_factors)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        factors = self.encode(batch["image"], batch["entity_features"], batch["entity_mask"])
        next_factors_true = self.encode(batch["next_image"], batch["next_entity_features"], batch["entity_mask"])
        mask = self.factor_mask(batch["entity_mask"])
        ctrl_post, exo_post = self.control_bridge.posterior(
            self.control_anchor(factors, mask),
            self.control_anchor(next_factors_true, mask),
            factors,
            next_factors_true,
        )
        ctrl_prior, exo_prior = self.prior_latents(factors, batch["entity_mask"])
        posterior_next, _ = self.rollout_step(factors, batch["entity_mask"], ctrl_post, exo_post)
        prior_next, _ = self.rollout_step(factors, batch["entity_mask"], ctrl_prior, exo_prior)
        return self._forward_common(
            batch,
            posterior_next,
            prior_next,
            ctrl_post,
            exo_post,
            ctrl_prior,
            exo_prior,
            factors,
            mask,
        )


class ActionConditionedWorldModel(_TokenWorldModelBase):
    def __init__(self, config: ModelConfig, image_size: int = 84):
        super().__init__(config, image_size=image_size)

    def rollout_step(
        self,
        factors: torch.Tensor,
        entity_mask: torch.Tensor,
        actions: torch.Tensor,
        exo_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        factor_mask = self.factor_mask(entity_mask)
        control_latent = self.action_to_control(actions)
        if exo_latent is None:
            _, exo_latent = self.prior_latents(factors, entity_mask)
        next_factors = self.dynamics(factors, control_latent, exo_latent, factor_mask)
        return next_factors, self.decode_state(next_factors)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        factors = self.encode(batch["image"], batch["entity_features"], batch["entity_mask"])
        next_factors_true = self.encode(batch["next_image"], batch["next_entity_features"], batch["entity_mask"])
        mask = self.factor_mask(batch["entity_mask"])
        ctrl_post, exo_post = self.control_bridge.posterior(
            self.control_anchor(factors, mask),
            self.control_anchor(next_factors_true, mask),
            factors,
            next_factors_true,
        )
        ctrl_prior, exo_prior = self.prior_latents(factors, batch["entity_mask"])
        posterior_next, _ = self.rollout_step(factors, batch["entity_mask"], batch["action"], exo_post)
        prior_actions = torch.argmax(self.action_logits(ctrl_prior), dim=-1)
        prior_next, _ = self.rollout_step(factors, batch["entity_mask"], prior_actions, exo_prior)
        return self._forward_common(
            batch,
            posterior_next,
            prior_next,
            ctrl_post,
            exo_post,
            ctrl_prior,
            exo_prior,
            factors,
            mask,
        )
