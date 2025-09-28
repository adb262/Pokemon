import logging

import torch
import torch.nn as nn

from latent_action_model.patch_embedding import PatchEmbedding
from latent_action_model.spatio_temporal_transformer import SpatioTemporalTransformer

logger = logging.getLogger(__name__)

class VideoTokenizerEncoder(nn.Module):
    def __init__(
            self, *, num_images_in_video: int, image_height: int, image_width: int, channels: int, patch_height: int,
            patch_width: int, d_model: int, num_heads: int, num_layers: int, embedding_dim: int):
        super(VideoTokenizerEncoder, self).__init__()
        self.num_images_in_video = num_images_in_video
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.d_model = d_model

        self.embed_image_patches = PatchEmbedding(
            num_images_in_video=num_images_in_video,
            channels=channels,
            patch_height=patch_height,
            patch_width=patch_width,
            d_model=d_model,
        )

        self.encoder = SpatioTemporalTransformer(
            num_images_in_video=num_images_in_video,
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            use_spatial_transformer=True,
            use_temporal_transformer=True,
        )

        self.latent_projection = nn.Linear(d_model, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape (batch_size, num_images_in_video, c, h, w)
        x = self.embed_image_patches(x)
        logger.debug(f"x shape after embed_image_patches: {x.shape}")
        x = self.encoder(x)
        logger.debug(f"x shape after encoder: {x.shape}")
        x = self.latent_projection(x)
        logger.debug(f"x shape after latent_projection: {x.shape}")
        return x

class VideoTokenizerDecoder(nn.Module):
    def __init__(
            self, *, num_images_in_video: int, image_height: int, image_width: int, channels: int, patch_height: int,
            patch_width: int, d_model: int, num_heads: int, num_layers: int, embedding_dim: int):
        super(VideoTokenizerDecoder, self).__init__()
        self.num_images_in_video = num_images_in_video
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.d_model = d_model

        self.embed_image_patches = PatchEmbedding(
            num_images_in_video=num_images_in_video,
            channels=channels,
            patch_height=patch_height,
            patch_width=patch_width,
            d_model=d_model,
        )

        self.encoder = SpatioTemporalTransformer(
            num_images_in_video=num_images_in_video,
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            use_spatial_transformer=True,
            use_temporal_transformer=True,
        )

        self.latent_projection = nn.Linear(embedding_dim, d_model)
        self.patch_to_pixels = nn.Conv2d(in_channels=d_model, out_channels=channels * patch_height * patch_width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape (batch_size, num_images_in_video, num_patches, embedding_dim)
        x = self.embed_image_patches(x)
        logger.debug(f"x shape after embed_image_patches: {x.shape}")
        x = self.encoder(x)
        logger.debug(f"x shape after encoder: {x.shape}")
        x = self.latent_projection(x)
        logger.debug(f"x shape after latent_projection: {x.shape}")
        x = self.patch_to_pixels(x)
        logger.debug(f"x shape after patch_to_pixels: {x.shape}")
        return x


class VideoTokenizer(nn.Module):
    def __init__(self, *, num_images_in_video: int, image_height: int, image_width: int, channels: int, patch_height: int,
            patch_width: int, d_model: int, num_heads: int, num_layers: int, embedding_dim: int):
        super(VideoTokenizer, self).__init__()
        self.encoder = VideoTokenizerEncoder(
            num_images_in_video=num_images_in_video, image_height=image_height, image_width=image_width, channels=channels,
            patch_height=patch_height, patch_width=patch_width, d_model=d_model, num_heads=num_heads, num_layers=num_layers, embedding_dim=embedding_dim)
        self.decoder = VideoTokenizerDecoder(
            num_images_in_video=num_images_in_video, image_height=image_height, image_width=image_width, channels=channels,
            patch_height=patch_height, patch_width=patch_width, d_model=d_model, num_heads=num_heads, num_layers=num_layers, embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))