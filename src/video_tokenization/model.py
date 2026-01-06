import logging

import torch
import torch.nn as nn

from latent_action_model.patch_embedding import PatchEmbedding
from quantization.fsq import FiniteScalarQuantizer
from torch_utilities.initialize import init_weights
from torch_utilities.pixel_shuffle_frame_reconstruction import PixelShuffleFrameHead
from transformers.spatio_temporal_transformer import SpatioTemporalTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VideoTokenizerEncoder(nn.Module):
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
        embedding_dim: int,
    ):
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
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape (batch_size, num_images_in_video, c, h, w)
        logger.debug(f"x shape before embed_image_patches: {x.shape}")
        x = self.embed_image_patches(x)
        logger.debug(f"x shape after embed_image_patches: {x.shape}")
        x = self.encoder(x)
        logger.debug(f"x shape after encoder: {x.shape}")
        x = self.latent_projection(x)
        logger.debug(f"x shape after latent_projection: {x.shape}")
        return x


class VideoTokenizerDecoder(nn.Module):
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
        embedding_dim: int,
    ):
        super(VideoTokenizerDecoder, self).__init__()
        self.num_images_in_video = num_images_in_video
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.d_model = d_model

        self.latent_projection = nn.Linear(embedding_dim, d_model)

        self.encoder = SpatioTemporalTransformer(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape (batch_size, num_images_in_video, num_patches, embedding_dim)
        logger.debug(f"x shape before latent_projection in decoder: {x.shape}")
        x = self.latent_projection(x)
        logger.debug(f"x shape before encoder in decoder: {x.shape}")
        x = self.encoder(x)
        logger.debug(f"x shape after latent_projection in decoder: {x.shape}")

        x = self.patch_to_pixels(x)
        logger.debug(f"x shape after patch_to_pixels in decoder: {x.shape}")
        return x


class VideoTokenizer(nn.Module):
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
        embedding_dim: int,
        bins: list[int],
    ):
        super(VideoTokenizer, self).__init__()
        self.encoder = VideoTokenizerEncoder(
            num_images_in_video=num_images_in_video,
            image_height=image_height,
            image_width=image_width,
            channels=channels,
            patch_height=patch_height,
            patch_width=patch_width,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
        )

        device = torch.device("cpu")
        if torch.backends.mps.is_available():
            device = torch.device("mps")

        if torch.cuda.is_available():
            device = torch.device("cuda")

        self.fsq = FiniteScalarQuantizer(
            levels=bins,
            embedding_dim=embedding_dim,
            device=device,
        )

        self.decoder = VideoTokenizerDecoder(
            num_images_in_video=num_images_in_video,
            image_height=image_height,
            image_width=image_width,
            channels=channels,
            patch_height=patch_height,
            patch_width=patch_width,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            embedding_dim=len(bins),
        )

    def get_vocab_size(self) -> int:
        return self.fsq.codebook_size

    def get_mask_token_idx(self) -> int:
        return self.fsq.mask_token_idx

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def quantized_value_to_codes(self, x: torch.Tensor) -> torch.Tensor:
        return self.fsq.quantized_value_to_codes(x)

    def mask_latent_tokens(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x[mask] = self.fsq.mask_token_embedding
        return x

    def mask_codebook_tokens(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x[mask] = torch.tensor(self.fsq.mask_token_idx, dtype=torch.long, device=x.device)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fsq(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
