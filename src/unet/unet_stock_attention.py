# unet_stock_attention.py
"""
Transformer for Stock Prediction U-Net

This implements the transformer module used in the U-Net adapted for stock prediction.
It provides conditional attention using historical data.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class SpatialTransformer(nn.Module):
    """
    ## Spatial Transformer for 1D Sequences
    Adapted for stock data: operates on [B, C, L] sequences.
    """

    def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int):
        """
        :param channels: is the number of channels in the feature map (C)
        :param n_heads: is the number of attention heads
        :param n_layers: is the number of transformer layers
        :param d_cond: is the size of the conditional embedding (last dim of cond)
        """
        super().__init__()
        # Initial group normalization
        self.norm = torch.nn.GroupNorm(num_groups=min(32, channels), num_channels=channels, eps=1e-6, affine=True)
        # Initial 1x1 convolution
        self.proj_in = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, n_heads, channels // n_heads, d_cond=d_cond) for _ in range(n_layers)]
        )

        # Final 1x1 convolution
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: is the feature map of shape `[batch_size, channels, sequence_length]`
        :param cond: is the conditional embeddings of shape `[batch_size, seq_cond, d_cond]`
        """
        # Get shape `[batch_size, channels, sequence_length]`
        b, c, l = x.shape
        # For residual connection
        x_in = x
        # Normalize
        x = self.norm(x)
        # Initial 1x1 convolution
        x = self.proj_in(x)
        # Transpose and reshape from `[batch_size, channels, sequence_length]`
        # to `[batch_size, sequence_length, channels]`
        x = x.permute(0, 2, 1).contiguous() # [B, L, C]
        # Apply the transformer layers
        for block in self.transformer_blocks:
            x = block(x, cond)
        # Reshape and transpose from `[batch_size, sequence_length, channels]`
        # to `[batch_size, channels, sequence_length]`
        x = x.permute(0, 2, 1).contiguous() # [B, C, L]
        # Final 1x1 convolution
        x = self.proj_out(x)
        # Add residual
        return x + x_in


class BasicTransformerBlock(nn.Module):
    """
    ### Transformer Layer
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int):
        """
        :param d_model: is the input embedding size (channels)
        :param n_heads: is the number of attention heads
        :param d_head: is the size of an attention head
        :param d_cond: is the size of the conditional embeddings
        """
        super().__init__()
        # Self-attention layer and pre-norm layer
        self.attn1 = CrossAttention(d_model, d_model, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)
        # Cross attention layer and pre-norm layer
        # Note: d_cond for keys/values is the size of the condition's feature dim
        self.attn2 = CrossAttention(d_model, d_cond, n_heads, d_head)
        self.norm2 = nn.LayerNorm(d_model)
        # Feed-forward network and pre-norm layer
        self.ff = FeedForward(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, sequence_length, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size, seq_cond, d_cond]`
        """
        # Self attention
        x = self.attn1(self.norm1(x)) + x
        # Cross-attention with conditioning
        x = self.attn2(self.norm2(x), cond=cond) + x
        # Feed-forward network
        x = self.ff(self.norm3(x)) + x
        return x


class CrossAttention(nn.Module):
    """
    ### Cross Attention Layer

    This falls-back to self-attention when conditional embeddings are not specified.
    """

    use_flash_attention: bool = False

    def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int, is_inplace: bool = True):
        """
        :param d_model: is the input embedding size (channels)
        :param n_heads: is the number of attention heads
        :param d_head: is the size of an attention head
        :param d_cond: is the size of the conditional embeddings (keys/values)
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        """
        super().__init__()

        self.is_inplace = is_inplace
        self.n_heads = n_heads
        self.d_head = d_head

        # Attention scaling factor
        self.scale = d_head ** -0.5

        # Query, key and value mappings
        d_attn = d_head * n_heads
        self.to_q = nn.Linear(d_model, d_attn, bias=False)
        self.to_k = nn.Linear(d_cond, d_attn, bias=False)
        self.to_v = nn.Linear(d_cond, d_attn, bias=False)

        # Final linear layer
        self.to_out = nn.Sequential(nn.Linear(d_attn, d_model))

        # Flash attention is not adapted for this specific case
        self.flash = None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        :param x: are the input embeddings of shape `[batch_size, seq_len, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size, seq_cond, d_cond]`
        """

        # If `cond` is `None` we perform self attention
        has_cond = cond is not None
        if not has_cond:
            cond = x

        # Get query, key and value vectors
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        # Fallback to normal attention
        return self.normal_attention(q, k, v)

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Normal Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq_x, d_attn]`
        :param k: are the key vectors before splitting heads, of shape `[batch_size, seq_cond, d_attn]`
        :param v: are the value vectors before splitting heads, of shape `[batch_size, seq_cond, d_attn]`
        """

        # Split them to heads of shape `[batch_size, seq_len, n_heads, d_head]`
        q = q.view(*q.shape[:2], self.n_heads, -1) # [B, seq_x, H, D_head]
        k = k.view(*k.shape[:2], self.n_heads, -1) # [B, seq_cond, H, D_head]
        v = v.view(*v.shape[:2], self.n_heads, -1) # [B, seq_cond, H, D_head]

        # Calculate attention: Q @ K^T / sqrt(d_head)
        # This gives shape `[batch_size, n_heads, seq_x, seq_cond]`
        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale

        # Compute softmax
        # `bhij` means `[batch, heads, seq_x, seq_cond]`
        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        # Compute attention output: attn @ V
        # This gives shape `[batch_size, n_heads, seq_x, d_head]`
        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq_x, n_heads * d_head]`
        out = out.reshape(*out.shape[:2], -1)
        # Map to `[batch_size, seq_x, d_model]` with a linear layer
        return self.to_out(out)


class FeedForward(nn.Module):
    """
    ### Feed-Forward Network
    """

    def __init__(self, d_model: int, d_mult: int = 4):
        """
        :param d_model: is the input embedding size
        :param d_mult: is multiplicative factor for the hidden layer size
        """
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(d_model, d_model * d_mult),
            nn.Dropout(0.),
            nn.Linear(d_model * d_mult, d_model)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class GeGLU(nn.Module):
    """
    ### GeGLU Activation

    $$\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$$
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # Combined linear projections $xW + b$ and $xV + c$
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor):
        # Get $xW + b$ and $xV + c$
        x, gate = self.proj(x).chunk(2, dim=-1)
        # $\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$
        return x * F.gelu(gate)
