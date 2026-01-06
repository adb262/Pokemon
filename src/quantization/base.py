from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseQuantizer(nn.Module, ABC):
    codebook_size: int
    mask_token_idx: int
    mask_token_embedding: nn.Parameter

    @abstractmethod
    def replace_unused_codebooks(self, num_batches: int):
        pass

    @abstractmethod
    def quantized_value_to_codes(self, quantized_value: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass
