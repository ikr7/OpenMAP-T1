import numpy as np
import torch
from scipy import ndimage

from utils.functions import normalize


def strip(voxel: np.typing.NDArray[np.float32], model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """
    Applies a given model to a 3D voxel array and returns the processed output.

    Args:
        voxel (numpy.ndarray): A 3D numpy array of shape (256, 256, 256) representing the input voxel data.
        model (torch.nn.Module): A PyTorch model to be used for processing the voxel data.
        device (torch.device): The device (CPU or GPU) on which the model and data should be loaded.

    Returns:
        torch.Tensor: A 3D tensor of shape (256, 256, 256) containing the processed output.
    """
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation for inference
    with torch.inference_mode():
        # Initialize an empty tensor to store the output
        output = torch.zeros(256, 256, 256, device=device)

        # Iterate over each slice in the voxel data
        for i, v in enumerate(voxel):
            # Reshape the slice to match the model's input dimensions
            image = v.reshape(1, 1, 256, 256)

            # Convert the numpy array to a PyTorch tensor and move it to the specified device
            image = torch.tensor(image, device=device)

            # Apply the model to the input image and apply the sigmoid activation function
            x_out = torch.sigmoid(model(image)).detach()

            # Store the output in the corresponding slice of the output tensor
            output[i] = x_out

        # Reshape the output tensor to the original voxel dimensions and return it
        return output.reshape(256, 256, 256)


def stripping(
    orig_image: np.typing.NDArray[np.float32], ssnet: torch.nn.Module
) -> tuple[np.typing.NDArray[np.float32], tuple[int, int, int]]:

    device = next(ssnet.parameters()).device

    sagittal_voxel = normalize(orig_image)
    coronal_voxel = sagittal_voxel.transpose(1, 2, 0)
    axial_voxel = sagittal_voxel.transpose(2, 1, 0)

    sagittal_prob = strip(sagittal_voxel, ssnet, device)
    coronal_prob = strip(coronal_voxel, ssnet, device).permute(2, 0, 1)
    axial_prob = strip(axial_voxel, ssnet, device).permute(2, 1, 0)

    ensembled_mask = (((sagittal_prob + coronal_prob + axial_prob) / 3) > 0.5).cpu().numpy()

    stripped_image = orig_image * ensembled_mask

    # mask centroid
    cx, cy, cz = map(int, ndimage.center_of_mass(ensembled_mask))
    shift = (128 - cx, 120 - cy, 128 - cz)

    shifted_image = np.roll(stripped_image, (shift), (0, 1, 2))

    cropped_shifted_image = shifted_image[32:-32, 16:-16, 32:-32]

    return cropped_shifted_image, shift
