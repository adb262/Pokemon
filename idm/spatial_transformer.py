import torch
import torch.nn as nn
import torch.nn.functional as F
from lapa.positional_bias import ContinuousPositionBias


class SpatialMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Continuous position bias
        self.pos_bias = ContinuousPositionBias(
            dim=d_model // 4,  # Smaller dim for efficiency
            heads=num_heads,
            num_dims=2,  # 2D for spatial positions
        )

    def forward(self, query, key, value, mask=None, patch_height=None, patch_width=None):
        batch_size, seq_len, d_model = query.shape

        # Linear transformations
        Q = self.w_q(query)  # (batch_size, seq_len, d_model)
        K = self.w_k(key)    # (batch_size, seq_len, d_model)
        V = self.w_v(value)  # (batch_size, seq_len, d_model)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(
            1, 2)  # (batch_size, num_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(
            1, 2)  # (batch_size, num_heads, seq_len, d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(
            1, 2)  # (batch_size, num_heads, seq_len, d_k)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)

        # Add positional bias if patch dimensions are provided
        if patch_height is not None and patch_width is not None:
            pos_bias = self.pos_bias(patch_height, patch_width, device=query.device)  # (num_heads, seq_len, seq_len)
            scores = scores + pos_bias.unsqueeze(0)  # Broadcast across batch dimension

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, d_k)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear transformation
        output = self.w_o(context)

        return output


class SpatialTransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.self_attn = SpatialMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, patch_height=None, patch_width=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, mask, patch_height, patch_width)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


class SpatialTransformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            SpatialTransformerLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None, patch_height=None, patch_width=None):
        for layer in self.layers:
            x = layer(x, mask, patch_height, patch_width)
        return x
