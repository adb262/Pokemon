import torch
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)


class NaiveCodebook(nn.Module):
    def __init__(self, num_embeddings: int, input_dim: int, embedding_dim: int):
        super(NaiveCodebook, self).__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._input_dim = input_dim

        # Create the codebook
        # Coming in is (batch_size, num_images, num_patches, patch_embed_dim)
        # Need CNN into (batch_size, num_images, embedding_dim)
        self._project_in = nn.Conv2d(self._input_dim, self._embedding_dim, kernel_size=3, padding=1)
        self._project_out = nn.Conv2d(self._embedding_dim, self._input_dim, kernel_size=3, padding=1)
        self._book = nn.Parameter(torch.randn(self._num_embeddings, self._embedding_dim), requires_grad=True)

    def forward(self, image_1: torch.Tensor, image_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Project the images into the embedding space
        logger.info(f"Image 1 shape: {image_1.shape}")
        logger.info(f"Image 2 shape: {image_2.shape}")
        image_1 = self._project_in(image_1)
        image_2 = self._project_in(image_2)

        input_data = image_1 - image_2

        # Compute the distances between the images and the codebook
        distances = torch.cdist(input_data, self._book)

        # Get the indices of the closest codebook vectors
        indices = distances.argmin(dim=1)

        hard_quantized_input = self._book[indices]

        random_vector = torch.normal(torch.zeros_like(input_data), torch.ones_like(input_data))

        norm_quantization_residual = (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()

        # defining vector quantization error
        vq_error = (norm_quantization_residual / norm_random_vector + 1e-6) * random_vector

        quantized_input = input_data + vq_error

        return self._project_out(quantized_input), indices

    @torch.inference_mode()
    def inference(self, image_1: torch.Tensor, image_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Project the images into the embedding space
        image_1 = self._project_in(image_1.reshape(image_1.shape[0], -1))
        image_2 = self._project_in(image_2.reshape(image_2.shape[0], -1))

        input_data = image_1 - image_2

        # Compute the distances between the images and the codebook
        distances = torch.cdist(input_data, self._book)

        # Get the indices of the closest codebook vectors
        indices = distances.argmin(dim=1)

        hard_quantized_input = self._book[indices]

        return self._project_out(hard_quantized_input), indices
