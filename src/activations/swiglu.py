import torch
import torch.nn.functional as F
from torch import nn

class SwiGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim_in, dim_out)
        self.value_proj = nn.Linear(dim_in, dim_out)
        self.out_proj = nn.Linear(dim_out, dim_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.value_proj(x)
        return self.out_proj(hidden)