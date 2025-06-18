import nibabel as nib
import numpy as np
import SimpleITK as sitk

from src.utils.preprocessing import (
    sitk_to_nib_nifti1,
    nib_nifti1_to_sitk,
    n4_bias_field_correction,
    conform_dimension,
)

orig_image_path = "tests/chris_t1.nii"
orig_image = nib.funcs.squeeze_image(nib.funcs.as_closest_canonical(nib.loadsave.load(orig_image_path)))
input_image = nib.nifti1.Nifti1Image(orig_image.get_fdata().astype(np.float32), affine=orig_image.affine)

numerical_tolerance = 1e-10


def test_sitk_to_nib_nifti1():
    sitk_image = sitk.ReadImage(orig_image_path)
    converted_nifti1_image = sitk_to_nib_nifti1(sitk_image)
    assert np.allclose(
        input_image.get_fdata(dtype=np.float32),
        converted_nifti1_image.get_fdata(dtype=np.float32),
        atol=numerical_tolerance,
    )
    if input_image.affine is not None and converted_nifti1_image.affine is not None:
        assert np.allclose(
            input_image.affine,
            converted_nifti1_image.affine,
            atol=numerical_tolerance,
        )


def test_nib_nifti1_to_sitk():
    sitk_image = sitk.ReadImage(orig_image_path)
    converted_sitk_image = nib_nifti1_to_sitk(input_image)
    assert np.allclose(
        sitk.GetArrayFromImage(sitk_image),
        sitk.GetArrayFromImage(converted_sitk_image),
        atol=numerical_tolerance,
    )
    assert np.allclose(
        sitk_image.GetSpacing(),
        converted_sitk_image.GetSpacing(),
        atol=numerical_tolerance,
    )
    assert np.allclose(
        sitk_image.GetOrigin(),
        converted_sitk_image.GetOrigin(),
        atol=numerical_tolerance,
    )
    assert np.allclose(
        sitk_image.GetDirection(),
        converted_sitk_image.GetDirection(),
        atol=numerical_tolerance,
    )


def test_n4_bias_field_correction():
    corrected_image = n4_bias_field_correction(input_image)

    # correction must not change image shape and affine
    assert input_image.shape == corrected_image.shape
    if input_image.affine is not None and corrected_image.affine is not None:
        assert np.allclose(
            input_image.affine,
            corrected_image.affine,
            atol=numerical_tolerance,
        )

    # correction must not alter an image with constant brightness (=zero bias)
    constant_image = nib.nifti1.Nifti1Image(np.ones((128, 128, 128), dtype=np.float32), affine=np.eye(4))
    corrected_constant_image = n4_bias_field_correction(constant_image)
    assert np.allclose(
        constant_image.get_fdata(),
        corrected_constant_image.get_fdata(),
        atol=numerical_tolerance,
    )

    # todo: add more tests with artifical bias field


def test_conform_dimension():
    conformed_image = conform_dimension(input_image)
    assert conformed_image.shape == (256, 256, 256)
    assert conformed_image.header.get_zooms()[:3] == (1.0, 1.0, 1.0)
