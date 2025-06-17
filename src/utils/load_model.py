from pathlib import Path
from contextlib import contextmanager

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


class ModelManager:

    def __init__(self, model_dir: Path, device: torch.device, optimize_memory: bool = False):
        self.model_dir = model_dir
        self.device = device
        self.optimize_memory = optimize_memory

        # Pre-load all models if not optimizing memory
        if not optimize_memory:
            self._preload_models()

    def _preload_models(self):
        self.cnet = load_cnet(self.model_dir).to(self.device)
        self.ssnet = load_ssnet(self.model_dir).to(self.device)

        pnet_models = load_pnet(self.model_dir)
        self.pnet_coronal = pnet_models[0].to(self.device)
        self.pnet_sagittal = pnet_models[1].to(self.device)
        self.pnet_axial = pnet_models[2].to(self.device)

        hnet_models = load_hnet(self.model_dir)
        self.hnet_coronal = hnet_models[0].to(self.device)
        self.hnet_axial = hnet_models[1].to(self.device)

    @contextmanager
    def load_cnet(self):
        if self.optimize_memory:
            model = load_cnet(self.model_dir).to(self.device)
            try:
                yield model
            finally:
                del model
                self.cleanup_gpu_memory()
        else:
            yield self.cnet

    @contextmanager
    def load_ssnet(self):
        if self.optimize_memory:
            model = load_ssnet(self.model_dir).to(self.device)
            try:
                yield model
            finally:
                del model
                self.cleanup_gpu_memory()
        else:
            yield self.ssnet

    @contextmanager
    def load_pnet(self):
        if self.optimize_memory:
            models = load_pnet(self.model_dir)
            pnet_coronal = models[0].to(self.device)
            pnet_sagittal = models[1].to(self.device)
            pnet_axial = models[2].to(self.device)
            try:
                yield pnet_coronal, pnet_sagittal, pnet_axial
            finally:
                del pnet_coronal, pnet_sagittal, pnet_axial
                self.cleanup_gpu_memory()
        else:
            yield self.pnet_coronal, self.pnet_sagittal, self.pnet_axial

    @contextmanager
    def load_hnet(self):
        if self.optimize_memory:
            models = load_hnet(self.model_dir)
            hnet_coronal = models[0].to(self.device)
            hnet_axial = models[1].to(self.device)
            try:
                yield hnet_coronal, hnet_axial
            finally:
                del hnet_coronal, hnet_axial
                self.cleanup_gpu_memory()
        else:
            yield self.hnet_coronal, self.hnet_axial

    @staticmethod
    def cleanup_gpu_memory() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
