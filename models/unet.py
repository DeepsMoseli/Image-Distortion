# -------- models/unet.py --------
import torch
import torch.nn as nn
from . import register_model

@register_model('unet')
class UNet(nn.Module):
    """
    Standard U-Net with customizable channel widths.
    Outputs 5 channels: first 2 are UV flow, last 3 are reconstructed RGB.
    """
    def __init__(self, in_ch=3, out_ch=5, features=[64,128,256,512]):
        super().__init__()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        # Encoder
        ch = in_ch
        for f in features:
            self.downs.append(self._double_conv(ch, f))
            ch = f
        # Bottleneck
        self.bottleneck = self._double_conv(ch, ch*2)
        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            self.ups.append(self._double_conv(f*2, f))
        # Final conv
        self.final = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def _double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[-(i//2 + 1)]
            if x.shape[2:] != s.shape[2:]:
                x = nn.functional.interpolate(x, size=s.shape[2:])
            x = torch.cat([s, x], dim=1)
            x = self.ups[i+1](x)
        return self.final(x)