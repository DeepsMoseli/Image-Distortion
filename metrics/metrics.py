import torch

def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio for images in [0,1].
    """
    mse = torch.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1 / mse)
