import torch
from vector_quantize_pytorch import ResidualVQ
import torch.nn as nn

import logging

from latent_action_model.decoder import VideoDecoder
from latent_action_model.patch_embedding import PatchEmbedding
from latent_action_model.spatio_temporal_transformer import SpatioTemporalTransformer
logger = logging.getLogger(__name__)


class LatentActionVQVAE(nn.Module):
    def __init__(
            self, *, num_images_in_video: int, image_height: int, image_width: int, channels: int, patch_height: int,
            patch_width: int, d_model: int, num_heads: int, num_layers: int, use_spatial_transformer: bool,
            use_temporal_transformer: bool, num_embeddings: int, embedding_dim: int):
        super(LatentActionVQVAE, self).__init__()
        self.num_images_in_video = num_images_in_video
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.patch_height = patch_height
        self.patch_width = patch_width

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
            use_spatial_transformer=use_spatial_transformer,
            use_temporal_transformer=use_temporal_transformer,
        )

        self.quantizer = ResidualVQ(
            dim=d_model,
            num_quantizers=8,
            codebook_size=num_embeddings,
            codebook_dim=embedding_dim,
            shared_codebook=True,
        )

        self.decoder = VideoDecoder(
            num_images_in_video=num_images_in_video,
            image_height=image_height,
            image_width=image_width,
            channels=channels,
            patch_height=patch_height,
            patch_width=patch_width,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            use_spatial_transformer=use_spatial_transformer,
            use_temporal_transformer=use_temporal_transformer,
        )

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # video is of shape (batch_size, num_images_in_video, channels, image_height, image_width)
        # We want to encode the video into a sequence of patches
        logger.info(f"video shape: {video.shape}")
        patched_video = self.embed_image_patches(video)
        logger.info(f"encoded_patches shape: {patched_video.shape}")
        # Now, we need to encode the video into a sequence of patches
        x = self.encoder(patched_video)

        # x is now of shape (batch_size, num_frames, num_patches, d_model). Take the last patch for each frame
        x = x[:, :, -1, :].squeeze(2)
        # x is now of shape (batch_size, num_frames, d_model)
        logger.info(f"encoded_patches shape after encoder: {x.shape}")
        # Now, we need to quantize the encoded patches.
        # Importantly, we only want 1 latent action per video, so we need to fix the dimensionality
        quantized, indices, commitment_loss = self.quantizer(x)
        logger.info(f"quantized shape: {quantized.shape}")
        logger.info(f"indices shape: {indices.shape}")
        logger.info(f"commitment_loss: {commitment_loss}")

        # Trim off the final image from the video
        patch_video = patched_video[:, :-1, :, :]

        # Now, we need to decode the quantized patches
        decoded = self.decoder(patch_video, quantized)
        logger.info(f"decoded shape: {decoded.shape}")
        return decoded, commitment_loss
