import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nibabel import processing

def sitk_to_nib_nifti1(sitk_img: sitk.Image) -> nib.nifti1.Nifti1Image:
    # zyx -> xyz
    arr = sitk.GetArrayFromImage(sitk_img).transpose((2, 1, 0))

    # affine
    spacing   = np.array(sitk_img.GetSpacing())
    origin    = np.array(sitk_img.GetOrigin())
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)
    affine_sitk = np.eye(4, dtype=np.float32)
    affine_sitk[:3, :3] = direction * spacing[:, None]
    affine_sitk[:3,  3] = origin

    # LPS -> RAS
    L2R = np.diag([-1, -1, 1, 1])
    affine_ras = L2R @ affine_sitk

    return nib.nifti1.Nifti1Image(arr, affine_ras)

def nib_nifti1_to_sitk(nifti_image: nib.nifti1.Nifti1Image) -> sitk.Image:
    
    # xyz -> zyx
    data = nifti_image.get_fdata().astype(np.float32).transpose((2, 1, 0))

    sitk_img = sitk.GetImageFromArray(data)

    # RAS -> LPS
    R2L    = np.diag([-1, -1, 1, 1])
    affine_lps = R2L @ nifti_image.affine

    # affine
    dir_mat    = affine_lps[:3, :3]
    spacing    = np.linalg.norm(dir_mat, axis=0)
    direction  = (dir_mat / spacing).flatten().tolist()
    origin     = affine_lps[:3,  3].tolist()

    sitk_img.SetSpacing(tuple(spacing.tolist()))
    sitk_img.SetDirection(tuple(direction))
    sitk_img.SetOrigin(tuple(origin))

    return sitk_img

def n4_bias_field_correction(image: nib.nifti1.Nifti1Image) -> nib.nifti1.Nifti1Image:
    
    shrink_factor = 4

    original_image = nib_nifti1_to_sitk(image)
    input_image = sitk.Shrink(original_image, [shrink_factor] * original_image.GetDimension())
    
    head_mask = sitk.LiThreshold(sitk.RescaleIntensity(original_image, 0, 255), 0, 1)
    mask_image = sitk.Shrink(head_mask, [shrink_factor] * original_image.GetDimension())
    
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    bias_corrector.Execute(input_image, mask_image)
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(original_image)
    
    corrected_image_full_resolution = original_image / sitk.Exp(log_bias_field)
    
    return sitk_to_nib_nifti1(corrected_image_full_resolution)


def conform_dimension(image: nib.nifti1.Nifti1Image) -> nib.nifti1.Nifti1Image:
    return processing.conform(image, out_shape=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0), order=1)


def preprocess(image: nib.nifti1.Nifti1Image) -> np.typing.NDArray[np.float32]:
    n4_corrected_image = n4_bias_field_correction(image)
    conformed_image = conform_dimension(n4_corrected_image)
    return conformed_image.get_fdata().astype(np.float32)
