"""Training objectives for information-preserving context latents."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PreservationLosses:
    total: torch.Tensor
    reconstruction: torch.Tensor
    contrastive: torch.Tensor


class ContextPreservationObjective(nn.Module):
    """Reconstruct source features and align their global semantics."""

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        num_heads: int,
        reconstruction_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.queries = nn.Linear(output_dim, embed_dim, bias=False)
        self.reconstruction_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, bias=False
        )
        self.output = nn.Linear(embed_dim, output_dim, bias=False)
        self.latent_summary = nn.Linear(embed_dim, output_dim, bias=False)
        self.reconstruction_weight = reconstruction_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

    def forward(self, latents, source_features, attention_mask=None):
        query = self.queries(source_features)
        reconstructed, _ = self.reconstruction_attention(
            query, latents, latents, need_weights=False
        )
        reconstructed = self.output(reconstructed)
        if attention_mask is None:
            attention_mask = torch.ones(
                source_features.shape[:2], device=source_features.device, dtype=torch.bool
            )
        valid = attention_mask.bool().unsqueeze(-1)
        squared_error = (reconstructed - source_features).float().square() * valid
        reconstruction = squared_error.sum() / valid.sum().clamp_min(1) / source_features.size(-1)

        source_sum = (source_features * valid).sum(1)
        source_mean = source_sum / valid.sum(1).clamp_min(1)
        latent_mean = self.latent_summary(latents.mean(1))
        source_mean = F.normalize(source_mean.float(), dim=-1)
        latent_mean = F.normalize(latent_mean.float(), dim=-1)
        similarity = latent_mean @ source_mean.transpose(0, 1) / self.temperature
        labels = torch.arange(similarity.size(0), device=similarity.device)
        contrastive = F.cross_entropy(similarity, labels)
        total = (
            self.reconstruction_weight * reconstruction
            + self.contrastive_weight * contrastive
        )
        return PreservationLosses(total, reconstruction, contrastive)
