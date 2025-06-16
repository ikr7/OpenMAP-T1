from pathlib import Path

import torch

from utils.network import UNet



def check_model_dir(model_dir: Path) -> None:

    weight_paths = [
        "CNet/CNet.pth",
        "SSNet/SSNet.pth",
        "PNet/coronal.pth",
        "PNet/sagittal.pth",
        "PNet/axial.pth",
        "HNet/coronal.pth",
        "HNet/axial.pth",
    ]

    for weight_path in weight_paths:
        resolved_weight_path = model_dir / weight_path
        if not (resolved_weight_path.exists() and resolved_weight_path.is_file()):
            raise Exception(f"{model_dir} does not contain {weight_path}")


def load_cnet(model_dir: Path) -> torch.nn.Module:
    cnet = UNet(1, 1)
    cnet.load_state_dict(torch.load(model_dir / "CNet" / "CNet.pth", weights_only=True))
    return cnet


def load_ssnet(model_dir: Path) -> torch.nn.Module:
    ssnet = UNet(1, 1)
    ssnet.load_state_dict(torch.load(model_dir / "SSNet" / "SSNet.pth", weights_only=True))
    return ssnet


def load_pnet(model_dir: Path) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    pnet_coronal = UNet(3, 142)
    pnet_coronal.load_state_dict(torch.load(model_dir / "PNet" / "coronal.pth", weights_only=True))
    pnet_sagittal = UNet(3, 142)
    pnet_sagittal.load_state_dict(torch.load(model_dir / "PNet" / "sagittal.pth", weights_only=True))
    pnet_axial = UNet(3, 142)
    pnet_axial.load_state_dict(torch.load(model_dir / "PNet" / "axial.pth", weights_only=True))
    return pnet_coronal, pnet_sagittal, pnet_axial


def load_hnet(model_dir: Path) -> tuple[torch.nn.Module, torch.nn.Module]:
    hnet_coronal = UNet(1, 3)
    hnet_coronal.load_state_dict(torch.load(model_dir / "HNet" / "coronal.pth", weights_only=True))
    hnet_axial = UNet(1, 3)
    hnet_axial.load_state_dict(torch.load(model_dir / "HNet" / "axial.pth", weights_only=True))
    return hnet_coronal, hnet_axial
