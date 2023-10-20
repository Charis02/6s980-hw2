import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves: int):
        super().__init__()
        self.num_octaves = num_octaves
        
    def forward(
        self,
        samples: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch embedded_dim"]:
        """Separately encode each channel using a positional encoding. The lowest
        frequency should be 2 * torch.pi, and each frequency thereafter should be
        double the previous frequency. For each frequency, you should encode the input
        signal using both sine and cosine.
        """

        no_batch = False
        if len(samples.shape) == 1:
            no_batch = True
            samples = samples.unsqueeze(0)
        
        frequencies = 2 * torch.pi * (2 ** torch.arange(self.num_octaves, dtype=torch.float32))
        freq_samples = torch.einsum("i,jk... -> jki",frequencies,samples)
        
        sin_encode = torch.sin(freq_samples).flatten(start_dim=-2)
        cos_encode = torch.cos(freq_samples).flatten(start_dim=-2)
        
        batch_size = samples.shape[0]
        embedding_size = 2* self.num_octaves * samples.shape[-1]
        result = torch.zeros(batch_size, embedding_size)
        result[:,::2] = sin_encode
        result[:,1::2] = cos_encode

        if no_batch:
            result = result.squeeze()

        return result


    def d_out(self, dimensionality: int):
        return 2 * self.num_octaves * dimensionality