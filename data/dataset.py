import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

class UVMapDataset(Dataset):
    """
    Paired distorted→clean RGB + UV maps.
    """
    def __init__(self, distorted_dir, uv_dir, clean_dir, transform, transform_type):
        self.distorted_dir  = distorted_dir
        self.uv_dir         = uv_dir
        self.clean_dir      = clean_dir
        self.transform      = transform
        self.transform_type = transform_type

        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        self.clean_paths = sorted(
            p for ext in exts
            for p in glob.glob(os.path.join(clean_dir, f"*{ext}"))
        )
        self.basenames = [
            os.path.splitext(os.path.basename(p))[0]
            for p in self.clean_paths
        ]

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        base       = self.basenames[idx]
        clean_path = self.clean_paths[idx]

        clean_img = Image.open(clean_path).convert('RGB')
        distorted_path = os.path.join(
            self.distorted_dir,
            f"{base}_{self.transform_type}.png"
        )
        dist_img = Image.open(distorted_path).convert('RGB')

        # UV map
        uv_path = os.path.join(self.uv_dir, f"{base}_uv_map.npy")
        uv = np.load(uv_path).astype(np.float32)     # H x W x 2
        uv = torch.from_numpy(uv).permute(2,0,1)     # 2 x H x W
        uv = uv.to(torch.float32)

        # Transforms
        if self.transform:
            clean_tensor = self.transform(clean_img)
            dist_tensor  = self.transform(dist_img)
        else:
            to_t = transforms.ToTensor()
            clean_tensor = to_t(clean_img)
            dist_tensor  = to_t(dist_img)

        # Resize UV to (H,W) of dist_tensor
        _, H, W = dist_tensor.shape
        uv = F.interpolate(
            uv.unsqueeze(0), size=(H, W),
            mode='bilinear', align_corners=False
        ).squeeze(0)

        # Normalize UV to [-1,1]
        uv[0] = 2 * (uv[0] / (W - 1) - 0.5)  # x-channel
        uv[1] = 2 * (uv[1] / (H - 1) - 0.5)  # y-channel

        return dist_tensor, uv, clean_tensor
