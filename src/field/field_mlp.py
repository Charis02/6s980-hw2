from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor
import torch

from .field import Field

from src.components.positional_encoding import PositionalEncoding


class FieldMLP(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up an MLP for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/mlp.yaml):

        - positional_encoding_octaves: The number of octaves in the positional encoding.
          If this parameter is None, do not positionally encode the input.
        - num_hidden_layers: The number of hidden linear layers.
        - d_hidden: The dimensionality of the hidden layers.

        Don't forget to add ReLU between your linear layers!
        """

        super().__init__(cfg, d_coordinate, d_out)

        self.encoder = None
        first_layer_in = d_coordinate

        if cfg.positional_encoding_octaves is not None:
            self.encoder = PositionalEncoding(cfg.positional_encoding_octaves)
            first_layer_in = int(self.encoder.d_out(d_coordinate))
            
        # add hidden layers
        self.d_hidden = cfg.d_hidden
        layers = []
        layers = [torch.nn.Linear(first_layer_in,cfg.d_hidden)]
        for _ in range(max(0,cfg.num_hidden_layers)):           
            layers.append(torch.nn.Linear(cfg.d_hidden,cfg.d_hidden))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(cfg.d_hidden,d_out))
        
        self.model = torch.nn.Sequential(*layers)

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the MLP at the specified coordinates."""

        input_coords = coordinates

        if self.encoder is not None:
            input_coords = self.encoder.forward(input_coords)

        result = self.model(input_coords)

        return result
