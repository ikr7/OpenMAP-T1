import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from utils.cropping import cropping
from utils.functions import save_voxel_with_reference_image
from utils.load_model import check_model_dir, load_cnet, load_pnet, load_ssnet
from utils.parcellation import parcellation
from utils.preprocessing import preprocess
from utils.stripping import stripping


@dataclass
class ParcellationArgs:
    input_dir: Path
    output_dir: Path
    model_dir: Path
    stop_after: Literal["cropping", "stripping", "parcellation"]
    save_intermediate_images: bool


def create_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument(
        "--input-dir",
        help="input directory containing .nii/.nii.gz files",
        dest="input_dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        help="output directory for parcellation results and intermidiate files",
        dest="output_dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--model-dir",
        help="model directory containing CNet/, HNet/, PNet/, and SSNet/",
        dest="model_dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--stop-after",
        help="perform face cropping only",
        dest="stop_after",
        choices=["cropping", "stripping", "parcellation"],
        default="parcellation",
    )

    parser.add_argument(
        "--save-intermediate-images",
        help="save intermediate images",
        dest="save_intermediate_images",
        action="store_true",
        default=False,
    )

    return parser


if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args(namespace=ParcellationArgs)

    input_dir = Path.cwd() / args.input_dir
    output_dir_root = Path.cwd() / args.output_dir
    model_dir = Path.cwd() / args.model_dir

    check_model_dir(model_dir)

    device = torch.device("cuda")

    cnet = load_cnet(model_dir).to(device)
    ssnet = load_ssnet(model_dir).to(device)

    pnet_coronal, pnet_sagittal, pnet_axial = load_pnet(model_dir)

    pnet_coronal = pnet_coronal.to(device)
    pnet_sagittal = pnet_sagittal.to(device)
    pnet_axial = pnet_axial.to(device)

    input_file_paths = sorted(
        [
            *input_dir.glob("**/*.nii"),
            *input_dir.glob("**/*.nii.gz"),
        ]
    )

    input_file_paths_pbar = tqdm(input_file_paths)

    for input_file_path in input_file_paths_pbar:

        input_file_paths_pbar.set_description_str(f"{input_file_path.name}")

        output_dir = output_dir_root / Path(*input_file_path.parts[len(input_dir.parts) :]).with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)

        parcellation_progress = tqdm(total=7, leave=False, bar_format="[{n}/{total}]: {desc}")

        # load image
        parcellation_progress.set_description_str("load image")

        orig_image = nib.funcs.squeeze_image(nib.funcs.as_closest_canonical(nib.loadsave.load(input_file_path)))
        input_image = nib.nifti1.Nifti1Image(orig_image.get_fdata().astype(np.float32), affine=orig_image.affine)

        parcellation_progress.update()

        # preprocess
        parcellation_progress.set_description_str("preprocess")
        preprocessed = preprocess(input_image)
        parcellation_progress.update()

        # face crop
        parcellation_progress.set_description_str("face crop")

        cropped = cropping(preprocessed, cnet)

        if args.save_intermediate_images:
            save_voxel_with_reference_image(cropped, orig_image, output_dir / f"{input_file_path.stem}_cropped.nii")

        parcellation_progress.update()

        if args.stop_after == "cropping":
            continue

        # skull-strip
        parcellation_progress.set_description_str("skull-strip")

        stripped, shift = stripping(cropped, ssnet)

        if args.save_intermediate_images:
            save_voxel_with_reference_image(stripped, orig_image, output_dir / f"{input_file_path.stem}_stripped.nii")

        parcellation_progress.update()

        if args.stop_after == "stripping":
            continue

        # parcellate
        parcellation_progress.set_description_str("parcellate")
        parcellation_map = parcellation(stripped, pnet_coronal, pnet_sagittal, pnet_axial)
        parcellation_progress.update()

        # hemisphere-separate
        parcellation_progress.set_description_str("hemisphere-separate")
        pass
        parcellation_progress.update()

        # postprocess
        parcellation_progress.set_description_str("postprocess")
        pass
        parcellation_progress.update()

        # save image
        parcellation_progress.set_description_str("save image")
        pass
        parcellation_progress.update()
