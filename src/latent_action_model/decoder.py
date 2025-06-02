from einops.layers.torch import Rearrange
import torch
import torch.nn as nn

import logging

from latent_action_model.spatio_temporal_transformer import SpatioTemporalTransformer

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


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
            nn.Sigmoid(),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h_patches, w=w_patches,
                      p1=self.patch_height, p2=self.patch_width, c=3)
        )

    def forward(self, x: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        # This takes in x (batch_size, num_images_in_video, num_patches, d_model) and latent_action (batch_size, num_images_in_video, d_model)
        # It outputs a tensor of shape (batch_size, channels, image_height, image_width)
        # Representing the reconstructed pixels of the next frame in the video given the previous frames and the latent action

        logger.debug(f"x shape: {x.shape}")

        # Perform self attention (spatial and temporal)
        x = self.st_transformer(x)
        logger.debug(f"x shape after st_transformer: {x.shape}")

        # Now, perform a cross attention between x and latent_action
        # However, st_transformer outputs (batch_size, num_frames, num_patches, d_model)
        # And latent action is (batch_size, num_frame, d_model)
        # So, we take the last frame from x as our "prediction" token
        x = x[:, -1, :, :].squeeze(1)
        logger.debug(f"x shape after taking last patch: {x.shape} and latent_action shape: {latent_action.shape}")

        # Then, we need to perform cross attention between x and latent_action
        # x is of shape (batch_size, num_patches, d_model)
        # latent_action is of shape (batch_size, d_model)
        # We need to expand latent_action to match the shape of x
        latent_action = latent_action.unsqueeze(1).expand(-1, x.shape[1], -1)
        logger.debug(f"latent_action shape after unsqueeze and expand: {latent_action.shape}")
        x, _ = self.attention(latent_action, x, x)
        logger.debug(f"x shape after attention: {x.shape}")
        # Now we need to decode the patches into images
        return self.out_projection(x)
