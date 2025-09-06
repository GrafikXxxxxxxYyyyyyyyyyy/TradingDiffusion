import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional



class GeGLU(nn.Module):
    """
    GeGLU Activation
    GeGLU(x) = (xW+b) * GELU(xV+c)
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        
        # Combined linear projections xW+b and xV+c
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor):
        # Get xW+b and xV+c
        x, gate = self.proj(x).chunk(2, dim=-1)

        return x * F.gelu(gate)
    


class FeedForward(nn.Module):
    """
    d_model is the input embedding size
    d_mult is multiplicative factor for the hidden layer size
    """
    def __init__(self, d_model: int, d_mult: int = 4):
        super().__init__()

        self.net = nn.Sequential(
            GeGLU(d_model, d_model * d_mult),
            nn.Dropout(0.),
            nn.Linear(d_model * d_mult, d_model)
        )
    
    def forward(self, x: torch.Tensor):
        return self.net(x)
    


class CrossAttention(nn.Module):
    pass



class BasicTransformerBlock(nn.Module):
    pass



class SpatialTransformer(nn.Module):
    def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int):
        super().__init__()

        # Initial group normalization
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

        # Initial 1×1 convolution
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, n_heads, channels // n_heads, d_cond=d_cond) for _ in range(n_layers)]
        )

        # Final 1×1 convolution
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        # Get shape [batch_size, channels, height, width]
        b, c, h, w = x.shape

        # For residual connection
        x_in = x

        # Normalize
        x = self.norm(x)

        # Initial 1×1 convolution
        x = self.proj_in(x)

        # Transpose and reshape from [batch_size, channels, height, width] to [batch_size, height * width, channels]
        x = x.permute(0, 2, 3, 1).view(b, h * w, c)

        # Apply the transformer layers
        for block in self.transformer_blocks:
            x = block(x, cond)

        # Reshape and transpose from [batch_size, height * width, channels] to [batch_size, channels, height, width]
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)

        # Final 1×1 convolution
        x = self.proj_out(x)

        # Add residual
        return x + x_in