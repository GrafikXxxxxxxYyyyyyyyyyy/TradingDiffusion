import torch
import torch.nn as nn


# Какие-то ещё необходимые модули для диффузионной модели
# например TimestepEmbedding


class MultiheadAttention(nn.Module):
    pass    



class DiffusionBlock(nn.Module):
    def __init__(self):
        super().__init__()

    
    def forward(self, noisy_target, timestep, condition):
        pass



class CustomDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()

    
    def forward(self, noisy_target: torch.Tensor, timestep: Union[torch.Tensor, int], condition: torch.Tensor):
        pass