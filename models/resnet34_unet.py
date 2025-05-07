import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from . import register_model

class DoubleConv(nn.Module):
    """Two 3×3 convs each followed by BN + ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

@register_model('resnet34_unet')
class ResNetUNet(nn.Module):
    """
    U-Net with a pretrained ResNet-34 encoder.
    Input:  (B, 3,  H,   W)
    Output: (B, 5,  H,   W)   # 2 UV channels + 3 RGB recon channels
    """
    def __init__(self, in_ch=3, out_ch=5, pretrained=True):
        super().__init__()
        # --- Encoder (ResNet34) ---
        backbone = models.resnet34(pretrained=pretrained)
        self.enc1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # → 64, H/2×W/2
        self.pool1 = backbone.maxpool                                        # → 64, H/4×W/4
        self.enc2 = backbone.layer1                                          # → 64, H/4×W/4
        self.enc3 = backbone.layer2                                          # → 128,H/8×W/8
        self.enc4 = backbone.layer3                                          # → 256,H/16×W/16
        self.enc5 = backbone.layer4                                          # → 512,H/32×W/32

        # --- Decoder ---
        # up5: 512 → 256, then concat enc4 (256+256 → dec5)
        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec5 = DoubleConv(256+256, 256)

        # up4: 256 → 128, then concat enc3 (128+128 → dec4)
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128+128, 128)

        # up3: 128 → 64, then concat enc2 (64+64 → dec3)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(64+64, 64)

        # up2: 64 → 64, then concat enc1 (64+64 → dec2)
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64+64, 64)

        # final conv: collapse 64 → out_ch channels
        self.final = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)              # 64 × H/2  × W/2
        e2 = self.enc2(self.pool1(e1)) # 64 × H/4  × W/4
        e3 = self.enc3(e2)             # 128× H/8  × W/8
        e4 = self.enc4(e3)             # 256× H/16 × W/16
        e5 = self.enc5(e4)             # 512× H/32 × W/32

        # Decoder + skip connections
        d5 = self.up5(e5)              # 256× H/16 × W/16
        d5 = self.dec5(torch.cat([d5, e4], dim=1))

        d4 = self.up4(d5)              # 128× H/8  × W/8
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)              # 64 × H/4  × W/4
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)              # 64 × H/2  × W/2
        d2 = self.dec2(torch.cat([d2, e1], dim=1))  # 64

        # Final 1×1 conv & upsample to input size
        out = self.final(d2)           # (B, out_ch, H/2, W/2)
        out = F.interpolate(out, size=x.shape[2:],  # back to H×W
                             mode='bilinear', align_corners=False)
        return out
