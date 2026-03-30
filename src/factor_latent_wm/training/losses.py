from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_entity_mse(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = (predicted - target) ** 2
    loss = loss.mean(dim=-1) * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def latent_consistency_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    ctrl_loss = F.mse_loss(outputs["ctrl_posterior"], outputs["ctrl_prior"])
    exo_loss = F.mse_loss(outputs["exo_posterior"], outputs["exo_prior"])
    return ctrl_loss + exo_loss


def world_model_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    reconstruction_weight: float,
    entity_weight: float,
    latent_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    current_state = masked_entity_mse(outputs["current_state_pred"], batch["entity_features"], batch["entity_mask"])
    next_state = masked_entity_mse(outputs["predicted_next_entities"], batch["next_entity_features"], batch["entity_mask"])
    prior_next_state = masked_entity_mse(
        outputs["prior_predicted_next_entities"], batch["next_entity_features"], batch["entity_mask"]
    )
    recon_current = F.mse_loss(outputs["current_reconstruction"], batch["image"])
    recon_next = F.mse_loss(outputs["next_reconstruction"], batch["next_image"])
    prior_recon_next = F.mse_loss(outputs["prior_next_reconstruction"], batch["next_image"])
    latent = latent_consistency_loss(outputs)

    total = (
        entity_weight * (current_state + next_state + prior_next_state)
        + reconstruction_weight * (recon_current + recon_next + prior_recon_next)
        + latent_weight * latent
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "current_state": float(current_state.detach().cpu()),
        "next_state": float(next_state.detach().cpu()),
        "prior_next_state": float(prior_next_state.detach().cpu()),
        "recon_current": float(recon_current.detach().cpu()),
        "recon_next": float(recon_next.detach().cpu()),
        "prior_recon_next": float(prior_recon_next.detach().cpu()),
        "latent": float(latent.detach().cpu()),
    }
    return total, metrics


def alignment_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    labelled_mask = batch["labelled"]
    if labelled_mask.sum() <= 0:
        return torch.zeros((), device=outputs["agent_action_logits"].device)

    per_item = F.cross_entropy(outputs["agent_action_logits"], batch["action"], reduction="none")
    weighted_ce = (per_item * labelled_mask).sum() / labelled_mask.sum().clamp_min(1.0)
    latent_target = outputs["action_control_latent"].detach()
    latent_mse = ((outputs["ctrl_posterior"] - latent_target) ** 2).mean(dim=-1)
    weighted_latent = (latent_mse * labelled_mask).sum() / labelled_mask.sum().clamp_min(1.0)
    prior_latent = ((outputs["ctrl_prior"] - latent_target) ** 2).mean(dim=-1)
    weighted_prior = (prior_latent * labelled_mask).sum() / labelled_mask.sum().clamp_min(1.0)
    return weighted_ce + 0.5 * weighted_latent + 0.25 * weighted_prior

