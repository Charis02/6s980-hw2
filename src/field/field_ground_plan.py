from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor
import torch

from .field import Field
from .field_grid import FieldGrid
from .field_mlp import FieldMLP
from src.components.positional_encoding import PositionalEncoding


class FieldGroundPlan(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a neural ground plan. You should reuse the following components:

        - FieldGrid from  src/field/field_grid.py
        - FieldMLP from src/field/field_mlp.py
        - PositionalEncoding from src/components/positional_encoding.py

        Your ground plan only has to handle the 3D case.
        """
        super().__init__(cfg, d_coordinate, d_out)
        assert d_coordinate == 3
        
        self.grid = FieldGrid(cfg.grid,2,cfg.d_grid_feature)
        self.encoder = PositionalEncoding(cfg.positional_encoding_octaves)
        self.mlp = FieldMLP(cfg.mlp,cfg.d_grid_feature + self.encoder.d_out(1),d_out)

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the ground plan at the specified coordinates. You should:

        - Sample the grid using the X and Y coordinates.
        - Positionally encode the Z coordinates.
        - Concatenate the grid's outputs with the corresponding encoded Z values, then
          feed the result through the MLP.
        """
        grid_output = self.grid(coordinates[:, :2])
        z_encoded = self.encoder(coordinates[:, 2].unsqueeze(1))
        
        concatenated_input = torch.cat((grid_output, z_encoded), dim=1)

        output = self.mlp(concatenated_input)

        return output
