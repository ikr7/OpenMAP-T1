import numpy as np
import torch
from scipy.ndimage import binary_closing

from utils.functions import normalize


def crop(voxel: np.typing.NDArray[np.float32], model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """
    Crops the given voxel data using the provided model and device.

    Args:
        voxel (numpy.ndarray): The input voxel data to be cropped, expected to be of shape (N, 256, 256).
        model (torch.nn.Module): The PyTorch model used for cropping.
        device (torch.device): The device (CPU or GPU) on which the computation will be performed.

    Returns:
        torch.Tensor: The cropped output tensor of shape (256, 256, 256).
    """
    model.eval()
    with torch.inference_mode():
        output = torch.zeros(256, 256, 256, device=device)
        for i, v in enumerate(voxel):
            image = v.reshape(1, 1, 256, 256)
            image = torch.tensor(image, device=device)
            x_out = torch.sigmoid(model(image)).detach()
            output[i] = x_out
        return output.reshape(256, 256, 256)


def closing(voxel: np.typing.NDArray[np.float32]) -> np.typing.NDArray[np.float32]:
    """
    Perform a binary closing operation on a 3D voxel array.

    This function applies a binary closing operation using a 3x3x3 structuring element
    and performs the operation for a specified number of iterations.

    Parameters:
    voxel (numpy.ndarray): A 3D numpy array representing the voxel data to be processed.

    Returns:
    numpy.ndarray: The voxel data after the binary closing operation.
    """
    selem = np.ones((3, 3, 3), dtype="bool")
    voxel = binary_closing(voxel, structure=selem, iterations=3)
    return voxel


def cropping(orig_image: np.typing.NDArray[np.float32], cnet: torch.nn.Module) -> np.typing.NDArray[np.float32]:

    device = next(cnet.parameters()).device

    sagittal_voxel = normalize(orig_image)
    coronal_voxel = sagittal_voxel.transpose(1, 2, 0)

    sagittal_prob = crop(sagittal_voxel, cnet, device)
    coronal_prob = crop(coronal_voxel, cnet, device).permute(2, 0, 1)

    ensembled_mask_tensor = ((sagittal_prob + coronal_prob) * 0.5) > 0.5
    ensembled_mask = closing(ensembled_mask_tensor.cpu().numpy())

    cropped_image = orig_image * ensembled_mask

    return cropped_image
