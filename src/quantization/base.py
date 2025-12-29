from abc import ABC, abstractmethod

from torch import nn


class BaseQuantizer(nn.Module, ABC):
    @abstractmethod
    def replace_unused_codebooks(self, num_batches: int):
        pass
