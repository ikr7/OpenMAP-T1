import torch

m = torch.nn.Conv3d(3, 5, 3).to("cuda")


k = next(m.parameters()).device

breakpoint()