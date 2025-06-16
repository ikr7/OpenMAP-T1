from typing import Literal

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from utils.functions import normalize


def separate(
    voxel: np.typing.NDArray[np.float32], model: torch.nn.Module, device: torch.device, mode: Literal["c", "a"]
) -> torch.Tensor:
    """
    Separates the voxel data based on the specified mode and processes it using the given model.

    Args:
        voxel (list or numpy.ndarray): The input voxel data to be processed.
        model (torch.nn.Module): The neural network model used for processing the voxel data.
        device (torch.device): The device (CPU or GPU) on which the model and data are loaded.
        mode (str): The mode of separation, either 'c' for coronal or 'a' for axial.

    Returns:
        torch.Tensor: The processed output tensor with shape (stack[0], 3, stack[1], stack[2]).
    """
    if mode == "c":
        # Set the stack dimensions for coronal mode
        stack = (224, 192, 192)
    elif mode == "a":
        # Set the stack dimensions for axial mode
        stack = (192, 224, 192)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation for inference
    with torch.inference_mode():
        # Initialize an output tensor with the specified stack dimensions
        output = torch.zeros(stack[0], 3, stack[1], stack[2], device=device)

        # Iterate over each slice in the voxel data
        for i, v in enumerate(voxel):
            # Reshape the slice and convert it to a tensor
            image = torch.tensor(v.reshape(1, 1, stack[1], stack[2]), device=device)
            # Perform a forward pass through the model and apply softmax
            x_out = torch.softmax(model(image), 1).detach()
            # Store the output in the corresponding slice of the output tensor
            output[i] = x_out

        # Return the processed output tensor
        return output


def hemisphere(
    stripped_image: np.typing.NDArray[np.float32], hnet_coronal: torch.nn.Module, hnet_axial: torch.nn.Module
) -> np.typing.NDArray[np.uint8]:

    device = next(hnet_coronal.parameters()).device

    sagittal_image = normalize(stripped_image)
    coronal_image = sagittal_image.transpose(1, 2, 0)
    axial_image = sagittal_image.transpose(2, 1, 0)

    ensembled_prob = (
        separate(coronal_image, hnet_coronal, device, "c").permute(1, 3, 0, 2)
        + separate(axial_image, hnet_axial, device, "a").permute(1, 3, 2, 0)
    ) / 2

    hemisphere_map = torch.argmax(ensembled_prob, 0).cpu().numpy().astype(np.uint8)

    # todo: tidy up this
    dilated_mask_1 = binary_dilation(hemisphere_map == 1, iterations=5).astype(np.uint8)
    dilated_mask_1[hemisphere_map == 2] = 2

    dilated_mask_2 = binary_dilation(dilated_mask_1 == 2, iterations=5).astype(np.uint8) * 2
    dilated_mask_2[dilated_mask_1 == 1] = 1

    return dilated_mask_2


# def hemisphere(voxel, hnet_c, hnet_a, device):
#     """
#     Processes a voxel image to separate and dilate hemispheres using neural networks.

#     Args:
#         voxel (torch.Tensor): The input voxel image tensor.
#         hnet_c (torch.nn.Module): The neural network model for coronal separation.
#         hnet_a (torch.nn.Module): The neural network model for transverse separation.
#         device (torch.device): The device to run the neural networks on (e.g., 'cpu' or 'cuda').

#     Returns:
#         numpy.ndarray: The processed and dilated mask of the hemispheres.
#     """
#     # Normalize the voxel data
#     voxel = normalize(voxel)

#     # Transpose the voxel data for coronal and transverse views
#     coronal = voxel.transpose(1, 2, 0)
#     transverse = voxel.transpose(2, 1, 0)

#     # Separate the coronal and transverse views using the respective models
#     out_c = separate(coronal, hnet_c, device, "c").permute(1, 3, 0, 2)
#     out_a = separate(transverse, hnet_a, device, "a").permute(1, 3, 2, 0)

#     # Combine the outputs from both views
#     out_e = out_c + out_a

#     # Get the final output by taking the argmax along the first dimension
#     out_e = torch.argmax(out_e, 0).cpu().numpy()

#     # Clear the CUDA cache
#     torch.cuda.empty_cache()

#     # Perform binary dilation on the mask for class 1
#     dilated_mask_1 = binary_dilation(out_e == 1, iterations=5).astype("int16")
#     dilated_mask_1[out_e == 2] = 2

#     # Perform binary dilation on the mask for class 2
#     dilated_mask_2 = binary_dilation(dilated_mask_1 == 2, iterations=5).astype("int16") * 2
#     dilated_mask_2[dilated_mask_1 == 1] = 1

#     # Return the final dilated mask
#     return dilated_mask_2
