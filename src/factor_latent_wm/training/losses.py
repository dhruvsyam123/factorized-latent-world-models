from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_entity_mse(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = (predicted - target) ** 2
    loss = loss.mean(dim=-1) * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def world_model_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    reconstruction_weight: float,
    entity_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    recon_current = F.mse_loss(outputs["current_reconstruction"], batch["image"])
    recon_next = F.mse_loss(outputs["next_reconstruction"], batch["next_image"])
    entity_loss = masked_entity_mse(
        outputs["predicted_next_entities"], batch["next_entity_features"], batch["entity_mask"]
    )
    total = reconstruction_weight * (recon_current + recon_next) + entity_weight * entity_loss
    metrics = {
        "loss": float(total.detach().cpu()),
        "recon_current": float(recon_current.detach().cpu()),
        "recon_next": float(recon_next.detach().cpu()),
        "entity": float(entity_loss.detach().cpu()),
    }
    return total, metrics


def alignment_loss(logits: torch.Tensor, actions: torch.Tensor, labelled_mask: torch.Tensor) -> torch.Tensor:
    per_item = F.cross_entropy(logits, actions, reduction="none")
    weighted = per_item * labelled_mask
    return weighted.sum() / labelled_mask.sum().clamp_min(1.0)
