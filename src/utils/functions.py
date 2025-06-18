import os
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel import processing


def normalize(voxel: np.typing.NDArray[np.float32]) -> np.typing.NDArray[np.float32]:
    nonzero = voxel[voxel > 0]
    voxel = np.clip(voxel, 0, np.mean(nonzero) + np.std(nonzero) * 2)
    voxel = np.divide(voxel - np.min(voxel), (np.max(voxel) - np.min(voxel)))
    voxel = (voxel * 2) - 1
    return voxel.astype(np.float32)


def save_voxel_with_reference_image(voxel: np.typing.NDArray, reference_image: nib.nifti1.Nifti1Image, dest_path: Path):

    out_shape = (
        reference_image.header["dim"][1],
        reference_image.header["dim"][2],
        reference_image.header["dim"][3],
    )

    voxel_size = (
        reference_image.header["pixdim"][1],
        reference_image.header["pixdim"][2],
        reference_image.header["pixdim"][3],
    )

    result_nii = processing.conform(
        nib.nifti1.Nifti1Image(voxel.astype(reference_image.get_data_dtype()), reference_image.affine),
        out_shape=out_shape,
        voxel_size=voxel_size,
        order=0,
    )

    nib.loadsave.save(result_nii, dest_path)
