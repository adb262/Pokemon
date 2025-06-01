import torch.nn as nn
import logging
import torch

# Rotary position embedding utilities from x_transformers
from x_transformers.x_transformers import (
    RotaryEmbedding,  # type: ignore
    apply_rotary_pos_emb  # type: ignore
)

logger = logging.getLogger(__name__)


class SpatioTemporalEncoderBlock(nn.Module):
    def __init__(self, *, num_images_in_video: int, num_heads: int, d_model: int,
                 use_spatial_transformer: bool, use_temporal_transformer: bool):
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

        self.projection_out = nn.Linear(d_model, d_model)
        self.rotary_emb = RotaryEmbedding(dim=d_model // num_heads) if use_temporal_transformer else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention over patches and temporal attention over timesteps.

        Args:
            x: Tensor with shape (batch_size, num_images_in_video, num_patches, d_model)

        Returns:
            Tensor with the same shape as the input after spatial- and/or temporal-attention.
        """

        batch_size, num_frames, num_patches, d_model = x.shape
        logger.info(f"x shape: {x.shape}")

        spatial_attention_output = torch.zeros_like(x, device=x.device)
        temporal_attention_output = torch.zeros_like(x, device=x.device)
        if self.use_spatial_transformer:
            x_spatial_in = x.view(batch_size * num_frames, num_patches, d_model)
            logger.info(f"x_spatial_in shape: {x_spatial_in.shape}")
            x_spatial_out, _ = self.spatial_transformer_attention(
                x_spatial_in,
                x_spatial_in,
                x_spatial_in
            )
            logger.info(f"x_spatial_out shape: {x_spatial_out.shape}")
            spatial_attention_output = x_spatial_out.view(batch_size, num_frames, num_patches, d_model)
            logger.info(f"spatial_attention_output shape: {spatial_attention_output.shape}")

        if self.use_temporal_transformer:
            x_temp = x.permute(0, 2, 1, 3).contiguous()
            logger.info(f"x_temp (after permute) shape: {x_temp.shape}")
            x_temp = x_temp.view(batch_size * num_patches, num_frames, d_model)
            logger.info(f"x_temp (after view) shape: {x_temp.shape}")

            if self.rotary_emb is not None:
                freqs, scale = self.rotary_emb.forward_from_seq_len(num_frames)
                logger.info(f"freqs shape: {freqs.shape}")
                if isinstance(scale, torch.Tensor):
                    logger.info(f"scale shape: {scale.shape}")
                freqs = freqs.repeat(x_temp.shape[0], 1, 1)
                if isinstance(scale, torch.Tensor):
                    scale = scale.repeat(x_temp.shape[0], 1, 1)
                logger.info(f"freqs (after repeat) shape: {freqs.shape}")
                if isinstance(scale, torch.Tensor):
                    logger.info(f"scale (after repeat) shape: {scale.shape}")
                x_temp = apply_rotary_pos_emb(x_temp, freqs, scale)  # type: ignore[arg-type]
                logger.info(f"x_temp (after rotary) shape: {x_temp.shape}")

            causal_mask = torch.triu(
                torch.ones(num_frames, num_frames, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            logger.info(f"causal_mask shape: {causal_mask.shape}")

            x_temp_out, _ = self.temporal_transformer_attention(
                x_temp, x_temp, x_temp, attn_mask=causal_mask
            )
            logger.info(f"x_temp_out shape: {x_temp_out.shape}")

            temporal_attention_output = x_temp_out.view(
                batch_size, num_patches, num_frames, d_model).permute(
                0, 2, 1, 3)
            logger.info(f"temporal_attention_output (before permute) shape: {temporal_attention_output.shape}")

        logger.info(f"spatial_attention_output shape: {spatial_attention_output.shape}")
        logger.info(f"temporal_attention_output shape: {temporal_attention_output.shape}")
        out = spatial_attention_output + temporal_attention_output
        logger.info(f"out shape: {out.shape}")
        projected_out = self.projection_out(out)
        logger.info(f"projected_out shape: {projected_out.shape}")
        result = x + projected_out
        logger.info(f"result shape: {result.shape}")
        return result
