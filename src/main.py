import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.axes
import nibabel as nib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from utils.cropping import cropping
from utils.load_model import (
    load_cnet,
    load_pnet,
    load_ssnet,
)
from utils.parcellation import parcellation
from utils.preprocessing import preprocess
from utils.stripping import stripping


@dataclass
class ParcellationArgs:
    input_dir: Path
    output_dir: Path
    model_dir: Path
    only_face_cropping: bool


def create_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument(
        "--input-dir",
        help="input directory containing .nii/.nii.gz files",
        dest="input_dir",
        required=True,
        type=Path,
    )

    parser.add_argument("--output-dir", help="output directory", dest="output_dir", required=True, type=Path)

    parser.add_argument(
        "--model-dir",
        help="model directory containing CNet/, HNet/, PNet/, and SSNet/",
        dest="model_dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--only-face-cropping",
        help="perform face cropping only",
        dest="only_face_cropping",
        action="store_true",
        default=False,
    )

    return parser


def check_model_dir(model_dir: Path) -> None:
    cnet_path = model_dir / "CNet/CNet.pth"
    if not (cnet_path.exists() and cnet_path.is_file()):
        raise Exception(f"{model_dir} does not contain ./CNet/CNet.pth")


def show_image(voxel: np.typing.NDArray[np.float32], ax: matplotlib.axes.Axes, title: str = "") -> None:
    nonzero = voxel[voxel > 0]
    voxel = np.clip(voxel, 0, 2 * np.std(nonzero) + np.mean(nonzero))
    ax.imshow(voxel[voxel.shape[0] // 2], cmap="gray")
    ax.set_title(title)


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
        parcellation_progress.update()

        # skull-strip
        parcellation_progress.set_description_str("skull-strip")
        stripped, shift = stripping(cropped, ssnet)
        parcellation_progress.update()

        # parcellate
        parcellation_progress.set_description_str("parcellate")
        parcellation_map = parcellation(stripped, pnet_coronal, pnet_sagittal, pnet_axial)
        parcellation_progress.update()

        # fig = plt.figure()
        # show_image(preprocessed, fig.add_subplot(2, 2, 1), "preprocessed")
        # show_image(cropped, fig.add_subplot(2, 2, 2), "cropped")
        # show_image(stripped, fig.add_subplot(2, 2, 3), "stripped")
        # show_image(parcellation_map, fig.add_subplot(2, 2, 4), "parcellation_map")
        # fig.show()
        # breakpoint()

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
