from __future__ import annotations

import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(image).flatten(1)
        return self.proj(hidden)


class SpatialImageEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.conv(image)
        return feats.flatten(2).transpose(1, 2)


class EntityFactorEncoder(nn.Module):
    def __init__(self, entity_feature_dim: int, factor_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(entity_feature_dim, factor_dim),
            nn.ReLU(),
            nn.Linear(factor_dim, factor_dim),
        )

    def forward(self, entity_features: torch.Tensor, global_context: torch.Tensor) -> torch.Tensor:
        local = self.mlp(entity_features)
        return local + global_context.unsqueeze(1)


class SlotAttentionEncoder(nn.Module):
    def __init__(self, num_slots: int, input_dim: int, slot_dim: int, iterations: int, mlp_dim: int):
        super().__init__()
        self.num_slots = num_slots
        self.iterations = iterations
        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(input_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, slot_dim),
        )
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.scale = slot_dim**-0.5

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        inputs = self.norm_inputs(inputs)
        keys = self.project_k(inputs)
        values = self.project_v(inputs)

        mu = self.slot_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slot_log_sigma.exp().expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        for _ in range(self.iterations):
            prev_slots = slots
            query = self.project_q(self.norm_slots(slots))
            attn_logits = torch.einsum("bid,bjd->bij", keys, query) * self.scale
            attn = attn_logits.softmax(dim=-1) + 1e-8
            attn = attn / attn.sum(dim=1, keepdim=True)
            updates = torch.einsum("bij,bid->bjd", attn, values)

            slots = self.gru(
                updates.reshape(-1, updates.shape[-1]),
                prev_slots.reshape(-1, prev_slots.shape[-1]),
            ).view(batch_size, self.num_slots, -1)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


class FactorDynamics(nn.Module):
    def __init__(self, factor_dim: int, latent_dim: int, num_heads: int):
        super().__init__()
        self.action_proj = nn.Linear(latent_dim, factor_dim)
        self.attn = nn.MultiheadAttention(factor_dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(factor_dim * 2, factor_dim),
            nn.ReLU(),
            nn.Linear(factor_dim, factor_dim),
        )

    def forward(self, factors: torch.Tensor, latent_actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        action_emb = self.action_proj(latent_actions)
        fused = factors + action_emb
        key_padding_mask = mask < 0.5
        attended, _ = self.attn(fused, fused, fused, key_padding_mask=key_padding_mask)
        output = self.mlp(torch.cat([factors, attended], dim=-1))
        return output * mask.unsqueeze(-1)


class ImageDecoder(nn.Module):
    def __init__(self, factor_dim: int, out_channels: int, image_size: int, hidden_channels: int):
        super().__init__()
        self.image_size = image_size
        self.proj = nn.Sequential(
            nn.Linear(factor_dim, hidden_channels * 11 * 11),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels // 4, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, factors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = (factors * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        hidden = self.proj(pooled).view(pooled.shape[0], -1, 11, 11)
        image = self.deconv(hidden)
        if image.shape[-1] != self.image_size:
            image = torch.nn.functional.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear")
        return image


class EntityDecoder(nn.Module):
    def __init__(self, factor_dim: int, entity_feature_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(factor_dim, factor_dim),
            nn.ReLU(),
            nn.Linear(factor_dim, entity_feature_dim),
        )

    def forward(self, factors: torch.Tensor) -> torch.Tensor:
        return self.mlp(factors)
