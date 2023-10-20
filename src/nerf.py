from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

from .field.field import Field
import torch


class NeRF(nn.Module):
    cfg: DictConfig
    field: Field

    def __init__(self, cfg: DictConfig, field: Field) -> None:
        super().__init__()
        self.cfg = cfg
        self.field = field
        self.num_samples = cfg.num_samples
        self.relu = torch.nn.ReLU()

    def forward(
        self,
        origins: Float[Tensor, "batch 3"],
        directions: Float[Tensor, "batch 3"],
        near: float,
        far: float,
    ) -> Float[Tensor, "batch 3"]:
        """Render the rays using volumetric rendering. Use the following steps:

        1. Generate sample locations along the rays using self.generate_samples().
        2. Evaluate the neural field at the sample locations. The neural field's output
           has four channels: three for RGB color and one for volumetric density. Don't
           forget to map these channels to valid output ranges.
        3. Compute the alpha values for the evaluated volumetric densities using
           self.compute_alpha_values().
        4. Composite these alpha values together with the evaluated colors from.
        """

        samples,boundaries = self.generate_samples(origins,directions,near,far,self.num_samples)
        samples_stacked = samples.view(-1,3)    # -1 infers the remaining dimension
        field_values = torch.exp(self.field(samples_stacked))
        sigmas = field_values[:,-1:].view(-1,self.num_samples)
        colors = field_values[:,:-1].unsqueeze(2).view(-1,self.num_samples,3)
        alphas = self.compute_alpha_values(sigmas,boundaries)
        
        radiances = self.alpha_composite(alphas,colors)
        return radiances

    def generate_samples(
        self,
        origins: Float[Tensor, "batch 3"],
        directions: Float[Tensor, "batch 3"],
        near: float,
        far: float,
        num_samples: int,
    ) -> tuple[
        Float[Tensor, "batch sample 3"],  # xyz sample locations
        Float[Tensor, "batch sample+1"],  # sample boundaries
    ]:
        """For each ray, equally divide the space between the specified near and far
        planes into num_samples segments. Return the segment boundaries (including the
        endpoints at the near and far planes). Also return sample locations, which fall
        at the midpoints of the segments.
        """
        
        sample_boundaries = torch.linspace(near,far,num_samples+1)
        sample_locations = sample_boundaries.clone()
        sample_locations = (sample_locations[1:] + sample_locations[:-1])*0.5
        
        batch_size = origins.shape[0]
        sample_locations = sample_locations.unsqueeze(0).expand(batch_size,-1)
        origins = origins.unsqueeze(2).expand(-1,-1,num_samples)
 
        normalized_directions = torch.nn.functional.normalize(directions, dim=-1, p=2)
        scaled_directions = torch.einsum("...j , ...i -> ...ij",sample_locations,normalized_directions)
        
        sample_locations_3d = origins + scaled_directions
        sample_locations_3d = sample_locations_3d.swapaxes(1,2)
        sample_boundaries = sample_boundaries.unsqueeze(0).expand(batch_size,-1)
        
        return sample_locations_3d, sample_boundaries
    
    def compute_alpha_values(
        self,
        sigma: Float[Tensor, "batch sample"],
        boundaries: Float[Tensor, "batch sample+1"],
    ) -> Float[Tensor, "batch sample"]:
        """Compute alpha values from volumetric densities (values of sigma) and segment
        boundaries.
        """

        deltas = boundaries[...,1:] - boundaries[...,:-1]
        alphas = 1 - torch.exp(-sigma*deltas)

        return alphas

    def alpha_composite(
        self,
        alphas: Float[Tensor, "batch sample"],
        colors: Float[Tensor, "batch sample 3"],
    ) -> Float[Tensor, "batch 3"]:
        """Alpha-composite the supplied alpha values and colors. You may assume that the
        background is black.
        """

        to_mult = 1 - alphas
        T = torch.cumprod(to_mult, dim=-1)
        w = T * alphas
        c = torch.einsum("ij,ij...->i...", w, colors)
        return c
