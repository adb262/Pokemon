import torch
import torch.nn as nn


class PatchEncoder(nn.Module):
    def __init__(
            self, num_heads: int, num_layers: int, patch_embed_dim: int, d_model: int, output_dim: int,
            attn_dropout: float, ff_dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_embed_dim = patch_embed_dim
        self.d_model = d_model
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout

        self.qkv_proj = nn.Sequential(
            nn.LayerNorm(patch_embed_dim),
            nn.Linear(patch_embed_dim, 3 * d_model),
            nn.Dropout(attn_dropout),
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_layers):
            # Tensor is of shape (batch_size, num_patches, patch_embed_dim)
            qkv: torch.Tensor = self.qkv_proj(x)
            q, k, v = qkv.split(qkv.shape[2] // 3, dim=2)

            self.positional_encoding(x.shape[1], device=x.device)

            q = (q @ k.transpose(-2, -1)) / (self.d_model ** -0.5)

        return q
