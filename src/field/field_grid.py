from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor
import torch

from .field import Field


class FieldGrid(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a grid for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/grid.yaml):

        - side_length: the side length in each dimension

        Your architecture only needs to support 2D and 3D grids.
        """
        super().__init__(cfg, d_coordinate, d_out)
        assert d_coordinate in (2, 3)
        side_length = cfg.side_length
        self.side_length = side_length
        self.d_out = d_out
        
        if d_coordinate == 2:
            self.grid = torch.nn.Parameter(torch.randn((1, self.d_out, side_length, side_length)))  # Create a 2D grid
        else:  # d_coordinate = 3
            self.grid = torch.nn.Parameter(torch.randn((1, self.d_out, side_length, side_length, side_length)))  # Create a 3D grid

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Use torch.nn.functional.grid_sample to bilinearly sample from the image grid.
        Remember that your implementation must support either 2D and 3D queries,
        depending on what d_coordinate was during initialization.
        """

        coordinates = coordinates*2 - 1 # from [0,1] to [-1,1]

        if self.d_coordinate == 2:
            reshaped_coords = coordinates.view(coordinates.size(0),1,1,2)
            grid_to_sample = self.grid.expand(coordinates.size(0),self.d_out,self.side_length,self.side_length)
            result =  torch.nn.functional.grid_sample(grid_to_sample,reshaped_coords,align_corners=True) # look at documentation, takes weird shape
            result = result.squeeze()
            #print("HERE! shape is",result.shape)
        else:  # d_coordinate = 3
            coordinates = coordinates.view(coordinates.size(0), 1, 1, 1, 3)
            grid_to_sample = self.grid.expand(coordinates.size(0), self.d_out, self.side_length,self.side_length,self.side_length)
            result = torch.nn.functional.grid_sample(self.grid, coordinates [..., :2])
            result = result.squeeze()

        return result