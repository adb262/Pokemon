import logging

import torch
import torch.nn as nn

# Rotary position embedding utilities from x_transformers
from x_transformers.x_transformers import (
    RotaryEmbedding,  # type: ignore
    apply_rotary_pos_emb,  # type: ignore
)

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


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

        # This will project the output back to the original patch space
        self.use_spatial_transformer = use_spatial_transformer
        self.use_temporal_transformer = use_temporal_transformer

        if use_spatial_transformer:
            self.spatial_transformer_attention = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, batch_first=True
            )
        if use_temporal_transformer:
            self.temporal_transformer_attention = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, batch_first=True
            )

        logger.info(
            f"Creating spatial and temporal transformer attention layers. Using {d_model} as the dimension."
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.spatial_ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.temporal_ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
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
        logger.debug(
            f"x shape: {x.shape}. num_frames: {num_frames}, num_patches: {num_patches}, d_model: {d_model}. self.num_images_in_video: {self.num_images_in_video}"
        )
        x = self.norm1(x)

        if self.use_spatial_transformer:
            # Issue with batch size > 1. Move to reshape
            x_spatial_in = x.reshape(
                batch_size * num_frames, num_patches, d_model
            ).contiguous()
            logger.debug(f"x_spatial_in shape: {x_spatial_in.shape}")

            q = x_spatial_in
            k = x_spatial_in
            v = x_spatial_in

            if self.spatial_rotary_emb is not None:
                freqs, scale = self.spatial_rotary_emb.forward_from_seq_len(num_patches)
                q = apply_rotary_pos_emb(q, freqs, scale)
                k = apply_rotary_pos_emb(k, freqs, scale)

            x_spatial_out, _ = self.spatial_transformer_attention(q, k, v)
            logger.debug(f"x_spatial_out shape: {x_spatial_out.shape}")
            x = x + x_spatial_out.view(batch_size, num_frames, num_patches, d_model)
            logger.debug(f"spatial_attention_output shape: {x.shape}")
            x = x + self.spatial_ffn(x)

        if self.use_temporal_transformer:
            x = self.norm2(x)
            x_temp = x.permute(0, 2, 1, 3).contiguous()
            logger.debug(f"x_temp (after permute) shape: {x_temp.shape}")
            x_temp = x_temp.reshape(batch_size * num_patches, num_frames, d_model)
            logger.debug(f"x_temp (after view) shape: {x_temp.shape}")

            q = x_temp
            k = x_temp
            v = x_temp

            if self.time_rotary_emb is not None:
                freqs, scale = self.time_rotary_emb.forward_from_seq_len(num_frames)
                q = apply_rotary_pos_emb(q, freqs, scale)
                k = apply_rotary_pos_emb(k, freqs, scale)

            causal_mask = torch.triu(
                torch.ones(num_frames, num_frames, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            logger.debug(f"causal_mask shape: {causal_mask.shape}")

            x_temp_out, _ = self.temporal_transformer_attention(
                q, k, v, attn_mask=causal_mask
            )
            logger.debug(f"x_temp_out shape: {x_temp_out.shape}")

            temporal_attention_output = x_temp_out.view(
                batch_size, num_patches, num_frames, d_model
            ).permute(0, 2, 1, 3)
            logger.debug(
                f"temporal_attention_output (before permute) shape: {temporal_attention_output.shape}"
            )
            x = x + temporal_attention_output
            x = x + self.temporal_ffn(x)

        return x
