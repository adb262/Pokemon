from einops.layers.torch import Rearrange
import torch
import torch.nn as nn

import logging

from latent_action_model.spatio_temporal_transformer import SpatioTemporalTransformer

logger = logging.getLogger(__name__)


class VideoDecoder(nn.Module):
    def __init__(
            self, *, num_images_in_video: int, image_height: int, image_width: int, channels: int, patch_height: int,
            patch_width: int, d_model: int, num_heads: int, num_layers: int, use_spatial_transformer: bool,
            use_temporal_transformer: bool):
        super(VideoDecoder, self).__init__()
        self.num_images_in_video = num_images_in_video
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.patch_height = patch_height
        self.patch_width = patch_width

        # Need a convolution that collapses our time dimension
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=num_images_in_video - 1)

        # Need our spatial and temporal embeddings
        self.st_transformer = SpatioTemporalTransformer(
            num_images_in_video=num_images_in_video,
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            use_spatial_transformer=use_spatial_transformer,
            use_temporal_transformer=use_temporal_transformer,
        )

        # Need a cross attention layer that takes in the latent action and the collapsed x
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

        h_patches = self.image_height // self.patch_height
        w_patches = self.image_width // self.patch_width
        patch_dim = self.channels * self.patch_height * self.patch_width

        # Now an out projection from d_model to patch_dim
        self.out_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h_patches, w=w_patches,
                      p1=self.patch_height, p2=self.patch_width, c=3)
        )

    def forward(self, x: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        # This takes in x (batch_size, num_images_in_video - 1, num_patches, d_model) and latent_action (batch_size, num_images_in_video, d_model)
        # It outputs a tensor of shape (batch_size, channels, image_height, image_width)
        # Representing the reconstructed pixels of the next frame in the video given the previous frames and the latent action

        logger.info(f"x shape: {x.shape}")

        # Perform self attention (spatial and temporal)
        x = self.st_transformer(x)
        logger.info(f"x shape after st_transformer: {x.shape}")

        # Now, perform a cross attention between x and latent_action
        # However, st_transformer outputs (batch_size, num_frames, num_patches, d_model)
        # And latent action is (batch_size, num_frame - 1, d_model)
        # So, we take the last frame from x as our "prediction" token
        x = x[:, -1, :, :].squeeze(1)
        logger.info(f"x shape after taking last patch: {x.shape} and latent_action shape: {latent_action.shape}")

        # Then, we need to perform cross attention between x and latent_action
        x, _ = self.attention(x, latent_action, latent_action)
        logger.info(f"x shape after attention: {x.shape}")
        # Now we need to decode the patches into images
        return self.out_projection(x)
