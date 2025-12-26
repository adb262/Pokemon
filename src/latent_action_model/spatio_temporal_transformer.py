import logging

import torch
import torch.nn as nn

from latent_action_model.spatio_temporal_encoder_block import SpatioTemporalEncoderBlock

logger = logging.getLogger(__name__)


class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self, *, num_images_in_video: int, num_heads: int, d_model: int, num_layers: int, use_spatial_transformer: bool,
            use_temporal_transformer: bool):
        super(SpatioTemporalTransformer, self).__init__()
        self.num_images_in_video = num_images_in_video
        self.d_model = d_model
        self.num_heads = num_heads

        self.encoder_blocks = nn.Sequential(
            *[
                SpatioTemporalEncoderBlock(
                    num_images_in_video=num_images_in_video, num_heads=num_heads, d_model=d_model,
                    use_spatial_transformer=use_spatial_transformer, use_temporal_transformer=use_temporal_transformer
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # outputs shape (batch_size, num_images_in_video, num_patches, d_model)
        return self.encoder_blocks(x)
