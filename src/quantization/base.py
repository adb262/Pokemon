from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseQuantizer(nn.Module, ABC):
    @abstractmethod
    def replace_unused_codebooks(self, num_batches: int):
        pass

    @abstractmethod
    def quantized_value_to_codes(self, quantized_value: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass
