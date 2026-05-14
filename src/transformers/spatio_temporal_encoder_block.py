import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers.x_transformers import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
)

from activations.swiglu import SwiGLU

logger = logging.getLogger(__name__)


class _SDPAttention(nn.Module):
    """Self-attention using F.scaled_dot_product_attention for Flash/mem-efficient backends.

    RoPE is applied to Q and K *after* the linear projection and *before*
    the attention score computation.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: RotaryEmbedding | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, S, D = x.shape
        H = self.num_heads

        qkv = self.qkv(x).view(B, S, 3, H, -1)
        q, k, v = qkv.unbind(dim=2)  # each (B, S, H, head_dim)

        # Move heads ahead of the sequence axis so axis -2 is the sequence
        # dimension that ``apply_rotary_pos_emb`` and SDPA both expect.
        q = q.transpose(1, 2)  # (B, H, S, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if rotary_emb is not None:
            # ``apply_rotary_pos_emb`` handles 4D (B, H, S, head_dim) inputs by
            # broadcasting the (1, S, head_dim) freqs across heads. The previous
            # ``q.reshape(-1, S, head_dim)`` collapsed (B, S, H) into the leading
            # axis, scrambling the (h, s) pairing within each row and applying
            # rotary frequencies to a position+head mix instead of pure positions
            # — which silently degraded attention quality on every run since
            # ``_SDPAttention`` was introduced.
            freqs, scale = rotary_emb.forward_from_seq_len(S)
            q = apply_rotary_pos_emb(q, freqs, int(scale))
            k = apply_rotary_pos_emb(k, freqs, int(scale))

        if is_causal:
            attn_mask = torch.triu(
                torch.full((S, S), float("-inf"), device=x.device, dtype=q.dtype),
                diagonal=1,
            )
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class SpatioTemporalEncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        num_images_in_video: int,
        num_heads: int,
        d_model: int,
        use_spatial_transformer: bool,
        use_temporal_transformer: bool,
    ):
        super(SpatioTemporalEncoderBlock, self).__init__()
        self.num_images_in_video = num_images_in_video
        self.d_model = d_model
        self.num_heads = num_heads

        self.use_spatial_transformer = use_spatial_transformer
        self.use_temporal_transformer = use_temporal_transformer

        if use_spatial_transformer:
            self.spatial_transformer_attention = _SDPAttention(d_model, num_heads)
        if use_temporal_transformer:
            self.temporal_transformer_attention = _SDPAttention(d_model, num_heads)

        self.pre_spatial_attn_norm = nn.RMSNorm(d_model)
        self.pre_spatial_ffn_norm = nn.RMSNorm(d_model)
        self.pre_temporal_attn_norm = nn.RMSNorm(d_model)
        self.pre_temporal_ffn_norm = nn.RMSNorm(d_model)

        self.spatial_ffn = SwiGLU(d_model, 4 * d_model)
        self.temporal_ffn = SwiGLU(d_model, 4 * d_model)

        self.spatial_rotary_emb = (
            RotaryEmbedding(dim=d_model // num_heads)
            if use_spatial_transformer
            else None
        )
        self.time_rotary_emb = (
            RotaryEmbedding(dim=d_model // num_heads)
            if use_temporal_transformer
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention over patches and temporal attention over timesteps.

        Args:
            x: Tensor with shape (batch_size, num_images_in_video, num_patches, d_model)

        Returns:
            Tensor with the same shape as the input after spatial- and/or temporal-attention.
        """
        batch_size, num_frames, num_patches, d_model = x.shape

        if self.use_spatial_transformer:
            x_spatial_in = self.pre_spatial_attn_norm(x)
            x_spatial_in = x_spatial_in.reshape(
                batch_size * num_frames, num_patches, d_model
            )

            x_spatial_out = self.spatial_transformer_attention(
                x_spatial_in, rotary_emb=self.spatial_rotary_emb, is_causal=False
            )
            x = x + x_spatial_out.view(batch_size, num_frames, num_patches, d_model)
            x = x + self.spatial_ffn(self.pre_spatial_ffn_norm(x))

        if self.use_temporal_transformer:
            x_temp_in = self.pre_temporal_attn_norm(x)
            x_temp = x_temp_in.permute(0, 2, 1, 3).reshape(
                batch_size * num_patches, num_frames, d_model
            )

            x_temp_out = self.temporal_transformer_attention(
                x_temp, rotary_emb=self.time_rotary_emb, is_causal=True
            )

            temporal_attention_output = x_temp_out.view(
                batch_size, num_patches, num_frames, d_model
            ).permute(0, 2, 1, 3)
            x = x + temporal_attention_output
            x = x + self.temporal_ffn(self.pre_temporal_ffn_norm(x))

        return x
