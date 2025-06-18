import torch
from src.utils.network import ConvBlock, EncodeBlock, DecodeBlock, UNet
from src.utils.cropping import crop
from src.utils.hemisphere import separate
from src.utils.parcellation import parcellate


def test_convblock_forward_shape():
    torch.manual_seed(0)
    block = ConvBlock(3, 5)
    x = torch.randn(1, 3, 16, 16)
    out = block(x)
    assert out.shape == (1, 5, 16, 16)


def test_encodeblock_forward_shape():
    torch.manual_seed(0)
    block = EncodeBlock(3, 6)
    x = torch.randn(1, 3, 16, 16)
    h, skip = block(x)
    assert h.shape == (1, 6, 8, 8)
    assert skip.shape == (1, 6, 16, 16)


def test_decodeblock_forward_shape():
    torch.manual_seed(0)
    block = DecodeBlock(8, 4)
    x = torch.randn(1, 8, 8, 8)
    skip = torch.randn(1, 4, 16, 16)
    out = block(x, skip)
    assert out.shape == (1, 4, 16, 16)


def test_unet_forward_shape():
    torch.manual_seed(0)
    model = UNet(1, 2)
    model.eval()
    x = torch.randn(1, 1, 16, 16)
    out = model(x)
    assert out.shape == (1, 2, 16, 16)


def test_crop_shape():
    torch.manual_seed(0)
    dummy_model = torch.nn.Conv2d(1, 1, kernel_size=1)
    voxel = torch.zeros(256, 256, 256)
    out = crop(voxel.numpy(), dummy_model, device=torch.device("cpu"))
    assert out.shape == (256, 256, 256)


def test_separate_shape():
    torch.manual_seed(0)
    dummy_model = torch.nn.Conv2d(1, 3, kernel_size=1)
    voxel = torch.zeros(224, 192, 192)
    out = separate(voxel.numpy(), dummy_model, device=torch.device("cpu"), mode="c")
    assert out.shape == (224, 3, 192, 192)


def test_parcellate_shape():
    torch.manual_seed(0)
    dummy_model = torch.nn.Conv2d(3, 142, kernel_size=1)
    voxel = torch.zeros(192, 224, 192)
    out = parcellate(voxel.numpy(), dummy_model, device=torch.device("cpu"), mode="s")
    assert out.shape == (192, 142, 224, 192)
