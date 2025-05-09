# utils/warp_function.py
import torch
import torch.nn.functional as F

def warp(distorted: torch.Tensor, uv_flow: torch.Tensor) -> torch.Tensor:
    """
    differentiable backward warp:
    - distorted:  B×3×H×W
    - uv_flow:    B×2×H×W in normalized [-1,1] coords
    returns:
    - reconstructed RGB: B×3×H×W
    """
    # grid_sample expects flow in (B,H,W,2) and coords in [-1,1]
    # permute: B×2×H×W → B×H×W×2
    grid = uv_flow.permute(0,2,3,1)

    # invert mapping: want to *pull* from distorted at UV coords
    # grid already maps output→input in normalized coords
    recon = F.grid_sample(
        distorted,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    return recon
