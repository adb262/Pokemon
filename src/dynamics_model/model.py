import torch
import torch.nn as nn
from torch.distributions import uniform

from latent_action_model.model import LatentActionVQVAE
from loss.loss_fns import reconstruction_loss
from torch_utilities.initialize import init_weights
from transformers.spatio_temporal_transformer import SpatioTemporalTransformer
from video_tokenization.model import VideoTokenizer


class DynamicsModel(nn.Module):
    def __init__(
        self,
        *,
        mask_ratio_lower_bound: float,
        mask_ratio_upper_bound: float,
        num_images_in_video: int,
        num_heads: int,
        num_layers: int,
        d_model: int,
        tokenizer: VideoTokenizer,
        action_model: LatentActionVQVAE,
    ):
        super(DynamicsModel, self).__init__()
        self.mask_ratio_lower_bound = mask_ratio_lower_bound
        self.mask_ratio_upper_bound = mask_ratio_upper_bound
        self.d_model = d_model
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.mask_ratio_distribution = uniform.Uniform(
            self.mask_ratio_lower_bound, self.mask_ratio_upper_bound
        )

        self.tokenizer_vocab_size = tokenizer.get_vocab_size()
        self.action_vocab_size = action_model.action_vocab_size

        self.tokenizer = tokenizer
        self.action_model = action_model

        # +1 for the mask token
        self.tokenizer_embedding = nn.Embedding(self.tokenizer_vocab_size + 1, d_model)
        self.action_embedding = nn.Embedding(self.action_vocab_size + 1, d_model)
        self.decoder = SpatioTemporalTransformer(
            num_images_in_video=num_images_in_video,
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            use_spatial_transformer=True,
            use_temporal_transformer=True,
        )
        self.vocab_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, self.tokenizer.get_vocab_size()),
        )
        self.token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.action_loss_fn = reconstruction_loss

        init_weights(self)

    def forward(
        self, video: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input is of shape (batch_size, num_images_in_video, channels, image_height, image_width)
        # Return our predictions (batch_size, num_images_in_video, num_patches, vocab_size)
        # And the masked targets

        # x is of shape (batch_size, num_images_in_video, num_patches)
        with torch.no_grad():
            # x is of shape (batch_size, num_images_in_video, num_patches)
            targets = self.tokenizer.quantized_value_to_codes(
                self.tokenizer.encode(video)
            )

        batch_size, num_images_in_video, num_patches = targets.shape
        mask = (
            torch.rand(batch_size, num_images_in_video - 1, num_patches)
            < self.mask_ratio_distribution.sample()
        )

        targets[:, 1:, :] = self.tokenizer.mask_codebook_tokens(targets[:, 1:, :], mask)

        # TODO: Should this also use the mask?
        # (batch_size, num_images_in_video - 1, num_patches, d_model)
        action_video_encoded = self.action_model.encode(video)
        action_tokens = self.action_model.get_action_sequence(
            action_video_encoded.detach()
        )

        # targets is of shape (batch_size, num_images_in_video, num_patches)
        x = self.tokenizer_embedding(targets)
        x[:, :-1, :] += self.action_embedding(action_tokens)

        x = self.decoder(x)
        x = self.vocab_head(x)

        reconstructed_action_video = self.action_model.decode(action_tokens)
        action_loss = self.action_loss_fn(
            video[:, 1:, :, :], reconstructed_action_video
        )

        targets[:, 1:, :][~mask] = -100
        token_loss = self.token_loss_fn(x[:, 1:, :, :], targets[:, 1:, :])

        return x, token_loss, action_loss
