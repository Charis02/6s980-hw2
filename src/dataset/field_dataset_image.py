from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor
import torch

from .field_dataset import FieldDataset
from PIL import Image
import numpy as np

class FieldDatasetImage(FieldDataset):
    def __init__(self, cfg: DictConfig) -> None:
        """Load the image in cfg.path into memory here."""

        super().__init__(cfg)

        # Read the image
        image = Image.open(cfg.path)

        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Convert the NumPy array to a PyTorch tensor
        self.image_tensor = torch.from_numpy(image_array)

        # Normalize pixel values to the range [0, 1]
        self.image_tensor = self.image_tensor / 255.0

        # Ensure the tensor has the correct data type
        self.image_tensor = self.image_tensor.float()

    def query(
        self,
        coordinates: Float[Tensor, "batch d_coordinate"],
    ) -> Float[Tensor, "batch d_out"]:
        """Sample the image at the specified coordinates and return the corresponding
        colors. Remember that the coordinates will be in the range [0, 1].

        You may find the grid_sample function from torch.nn.functional helpful here.
        Pay special attention to grid_sample's expected input range for the grid
        parameter.
        """

        result = torch.tensor([])
        image_size = self.grid_size

        for coordinate in coordinates:
            # Multiply each coordinate by the respective dimension of image size
            x, y = coordinate
            x = int(x * image_size[0])
            y = int(y * image_size[1])

            # Append the value in the found coordinate in self.image_tensor to result
            pixel_value = self.image_tensor[y, x]  # Assuming y corresponds to rows and x to columns
            result = torch.cat((result, pixel_value.unsqueeze(0)))

        return result

    @property
    def d_coordinate(self) -> int:
        return 2

    @property
    def d_out(self) -> int:
        return 3

    @property
    def grid_size(self) -> tuple[int, ...]:
        """Return a grid size that corresponds to the image's shape."""

        return ((self.image_tensor.shape[0]),self.image_tensor.shape[1])
