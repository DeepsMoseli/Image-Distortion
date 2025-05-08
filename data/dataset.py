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
    Dataset of paired (distorted→clean) images and UV maps.
    Assumes:
      - clean_dir contains files like 'basename.png', 'basename.jpg', etc.
      - distorted_dir contains ONLY PNGs named 'basename_<transform_type>.png'
      - uv_dir contains 'basename_uv_map.npy'
    """
    def __init__(self, distorted_dir, uv_dir, clean_dir, transform, transform_type):
        self.distorted_dir  = distorted_dir
        self.uv_dir         = uv_dir
        self.clean_dir      = clean_dir
        self.transform      = transform
        self.transform_type = transform_type

        # 1) index all your clean originals (any of these exts)
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        self.clean_paths = sorted(
            p for ext in exts
            for p in glob.glob(os.path.join(clean_dir, f"*{ext}"))
        )
        # strip off extension for lookup
        self.basenames = [os.path.splitext(os.path.basename(p))[0] for p in self.clean_paths]

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        base       = self.basenames[idx]
        clean_path = self.clean_paths[idx]

        # -- clean image --
        clean_img = Image.open(clean_path).convert('RGB')

        # -- distorted image, always PNG --
        distorted_path = os.path.join(
            self.distorted_dir,
            f"{base}_{self.transform_type}.png"
        )
        dist_img = Image.open(distorted_path).convert('RGB')

        # -- uv map at original resolution --
        uv_path = os.path.join(self.uv_dir, f"{base}_uv_map.npy")
        uv   = np.load(uv_path).astype(np.float32)       # H_orig x W_orig x 2
        uv   = torch.from_numpy(uv).permute(2,0,1)  # 2 x H_orig x W_orig 
        if torch.cuda.is_available():
            uv = uv.half()    

        # -- apply transforms to images --
        if self.transform:
            clean_tensor = self.transform(clean_img)
            dist_tensor  = self.transform(dist_img)
        else:
            to_t         = transforms.ToTensor()
            clean_tensor = to_t(clean_img)
            dist_tensor  = to_t(dist_img)

        # -- resize UV to match the image tensor size (C,H,W) --
        _, H, W = dist_tensor.shape
        uv = F.interpolate(uv.unsqueeze(0), size=(H, W),
                           mode='bilinear', align_corners=False).squeeze(0)

        # normalize uv to [-1,1] range
        _, H, W = uv.shape
        uv[0] = uv[0] / (W/2)   # u-coordinate
        uv[1] = uv[1] / (H/2)   # v-coordinate

        return dist_tensor, uv, clean_tensor
