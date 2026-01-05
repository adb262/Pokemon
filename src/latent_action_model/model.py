import logging
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange

from latent_action_model.patch_embedding import PatchEmbeddingConv
from quantization.base import BaseQuantizer
from quantization.fsq import FiniteScalarQuantizer
from quantization.nsvq import NSVQ
from torch_utilities.crop_center_patches import get_center_patch_indices
from torch_utilities.initialize import init_weights
from torch_utilities.pixel_shuffle_frame_reconstruction import PixelShuffleFrameHead
from transformers.spatio_temporal_transformer import SpatioTemporalTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LatentActionVQVAE(nn.Module):
    quantizer: BaseQuantizer
    action_vocab_size: int

    def __init__(
        self,
        *,
        num_images_in_video: int,
        image_height: int,
        image_width: int,
        channels: int,
        patch_height: int,
        patch_width: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        use_spatial_transformer: bool,
        use_temporal_transformer: bool,
        embedding_dim: int,
        quantizer_type: Literal["nsvq", "fsq"] = "fsq",
        bins: list[int] = [8],  # vocab size of 8
    ):
        super(LatentActionVQVAE, self).__init__()
        self.num_images_in_video = num_images_in_video
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.patch_height = patch_height
        self.patch_width = patch_width

        self.embed_image_patches = PatchEmbeddingConv(
            channels=channels,
            patch_size=patch_height,
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
        self.layer_norm = nn.LayerNorm(d_model)

        num_patches_h = image_height // patch_height
        num_patches_w = image_width // patch_width
        self.num_patches_per_image = num_patches_h * num_patches_w

        device = torch.device("cpu")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        if torch.cuda.is_available():
            device = torch.device("cuda")

        self.quantizer_type = quantizer_type

        if quantizer_type == "nsvq":
            self.quantizer = NSVQ(
                dim=d_model,
                num_embeddings=len(bins),
                embedding_dim=embedding_dim,
                patch_size=patch_height,
                image_size=image_height,
                device=device,
                discarding_threshold=0.1,
                initialization="normal",
                code_seq_len=1,
            )
        elif quantizer_type == "fsq":
            self.quantizer = FiniteScalarQuantizer(
                levels=bins,
                embedding_dim=d_model,
                device=device,
            )
        else:
            raise ValueError(f"Invalid quantizer type: {quantizer_type}")

        self.action_vocab_size = self.quantizer.codebook_size
        self.decoder = SpatioTemporalTransformer(
            num_images_in_video=num_images_in_video,
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            use_spatial_transformer=True,
            use_temporal_transformer=True,
        )
        self.patch_to_pixels = PixelShuffleFrameHead(
            embed_dim=d_model,
            patch_size=patch_height,
            channels=channels,
            H=image_height,
            W=image_width,
        )
        self.reconstruct = nn.Sequential(
            nn.LayerNorm(d_model),
            self.decoder,
            nn.LayerNorm(d_model),
            self.patch_to_pixels,
        )

        # This will act as a glorified embedding layer for the action
        self.action_projection = nn.Linear(1, d_model)
        self.center_indices = get_center_patch_indices(
            image_height, patch_height, image_width, patch_width, device
        )

        params = sum(p.numel() for p in self.parameters())
        logger.debug(f"LatentActionVQVAE initialized. Num params: {params}")
        init_weights(self)

    def decode(self, quantized: torch.Tensor, patch_video_for_decoder: torch.Tensor) -> torch.Tensor:
        # Trim off the final image from the video for decoder input
        # Patches for frames 0 to T-2

        # Now, we need to decode. The decoder aims to predict frame T-1 using frames 0 to T-2 and the quantized_action.
        # decoded = self.decoder(patch_video_for_decoder)

        # Shape is (B, N_images_in_video - 1, num_patches, d_model)
        # logger.debug(
        #     f"decoded shape: {decoded.shape}"
        # )  # Decoder outputs a single frame (B, C, H, W)
        return self.reconstruct(self.project_quantized_actions_fsq(quantized, patch_video_for_decoder))

    def encode_with_fsq(self, video: torch.Tensor) ->tuple[torch.Tensor, torch.Tensor]:
        # video is of shape (batch_size, num_images_in_video, channels, image_height, image_width)
        # We want to encode the video into a sequence of patches
        logger.debug(f"video shape: {video.shape}")
        # patched_video_from_embedder has shape (B, T*P, D) then reshaped to (B, T, P, D) by PatchEmbedding
        patched_video_from_embedder = self.embed_image_patches(video)
        logger.debug(
            f"patched_video_from_embedder shape: {patched_video_from_embedder.shape}"
        )

        # x_encoded_full has shape (batch_size, num_frames, num_patches, d_model)
        x_encoded_full = self.encoder(patched_video_from_embedder)
        logger.debug(f"x_encoded_full shape after encoder: {x_encoded_full.shape}")

        # Compute temporal action differential but only over the selected central patches.
        action_differential = (
            x_encoded_full[:, 1:, self.center_indices, :]
            - x_encoded_full[:, :-1, self.center_indices, :]
        )
        x_encoded_mean = action_differential.mean(dim=2)
        x_encoded_mean = self.layer_norm(x_encoded_mean)

        logger.debug(f"x_encoded_mean shape: {x_encoded_mean.shape}")
        quantized = self.quantizer(x_encoded_mean)

        # quantized, indices, commitment_loss = self.quantizer(action_continuous)
        logger.debug(f"quantized shape: {quantized.shape}")
        logger.debug(f"quantized values: {quantized}")
        return quantized, patched_video_from_embedder

    def project_quantized_actions_fsq(self, quantized: torch.Tensor, patched_video_from_embedder: torch.Tensor) -> torch.Tensor:
        # Project the quantized action to the d_model space
        action_projected = self.action_projection(quantized)
        logger.debug(f"action_projected shape: {action_projected.shape}")

        return patched_video_from_embedder[:, :-1, :, :] + action_projected.unsqueeze(2)

    def project_quantized_actions_nsvq(self, quantized: torch.Tensor, patched_video_from_embedder: torch.Tensor) -> torch.Tensor:
        # quantized, indices, commitment_loss = self.quantizer(action_continuous)
        logger.debug(f"quantized shape: {quantized.shape}")

        actions = rearrange(
            quantized,
            "(b t) p d -> b t p d",
            b=patched_video_from_embedder.shape[0],
            t=patched_video_from_embedder.shape[1] - 1,
        )

        # Trim off the final image from the video for decoder input
        # Patches for frames 0 to T-2
        return patched_video_from_embedder[:, :-1, :, :] + actions

    def encode_with_nsvq(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # video is of shape (batch_size, num_images_in_video, channels, image_height, image_width)
        # We want to encode the video into a sequence of patches
        logger.debug(f"video shape: {video.shape}")
        # patched_video_from_embedder has shape (B, T*P, D) then reshaped to (B, T, P, D) by PatchEmbedding
        patched_video_from_embedder = self.embed_image_patches(video)
        logger.debug(
            f"patched_video_from_embedder shape: {patched_video_from_embedder.shape}"
        )

        # x_encoded_full has shape (batch_size, num_frames, num_patches, d_model)
        x_encoded_full = self.encoder(patched_video_from_embedder)
        logger.debug(f"x_encoded_full shape after encoder: {x_encoded_full.shape}")

        # We now have a sequnce of (B, num_frames, num_patches, d_model)
        # We want to quantize frames (0, 1), (1, 2), etc.
        # Quantizer expects first image, target image, each of shape (B, num_patches, d_model)
        # So, we need to reshape x_encoded_full to (B * (num_frames - 1), num_patches, d_model)
        # 1) Form per-batch consecutive pairs along time
        first_images = x_encoded_full[:, :-1, :, :]  # (B, T-1, P, D) → frames 0..T-2
        target_images = x_encoded_full[:, 1:, :, :]  # (B, T-1, P, D) → frames 1..T-1

        # 2) Now flatten (B, T-1) into a single batch dimension
        first_images = rearrange(
            first_images, "b t p d -> (b t) p d"
        )  # (B*(T-1), P, D)
        target_images = rearrange(
            target_images, "b t p d -> (b t) p d"
        )  # (B*(T-1), P, D)

        # pass to NSVQ
        quantized = self.quantizer(first_images, target_images)

        return quantized, patched_video_from_embedder

    def encode(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Return the quantized values and the patched video from the embedder
        if self.quantizer_type == "nsvq":
            return self.encode_with_nsvq(video)
        elif self.quantizer_type == "fsq":
            return self.encode_with_fsq(video)
        else:
            raise ValueError(f"Invalid quantizer type: {self.quantizer_type}")

    def get_action_sequence(self, quantized: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantized_value_to_codes(quantized)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        actions, patched_video_from_embedder = self.encode(video)
        return self.decode(actions, patched_video_from_embedder)
