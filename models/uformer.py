import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register_model

class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) ×2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class TransformerBlock(nn.Module):
    """
    ViT‐style block: LayerNorm → MHA → residual → MLP → residual
    Operates on (B, C, H, W) tokens flattened to (B, N, C).
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1,2)                  # (B, H*W, C)
        # self-attn
        x2, _ = self.attn(self.norm1(x_seq),
                          self.norm1(x_seq),
                          self.norm1(x_seq))
        x_seq = x_seq + x2
        # MLP
        x2 = self.mlp(self.norm2(x_seq))
        x_seq = x_seq + x2
        # back to (B, C, H, W)
        return x_seq.transpose(1,2).view(B, C, H, W)

@register_model('uformer')
class UFormer(nn.Module):
    """
    UFormer: U-Net with Transformer blocks at each scale.
    Input:  (B, 3, H, W)
    Output: (B, 5, H, W)  # 2 UV + 3 RGB
    """
    def __init__(self, in_ch=3, out_ch=5,
                 features=[64,128,256,512],
                 num_heads=8, mlp_ratio=4.0):
        super().__init__()
        # Encoder
        self.down_convs = nn.ModuleList()
        self.trans_enc  = nn.ModuleList()
        prev_ch = in_ch
        for feat in features:
            self.down_convs.append(DoubleConv(prev_ch, feat))
            self.trans_enc.append(TransformerBlock(feat, num_heads, mlp_ratio))
            prev_ch = feat
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1])

        # Decoder
        self.up_convs = nn.ModuleList()
        self.trans_dec = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for feat in reversed(features[:-1]):
            self.up_convs.append(nn.ConvTranspose2d(prev_ch, feat, 2, 2))
            self.dec_convs.append(DoubleConv(feat*2, feat))
            self.trans_dec.append(TransformerBlock(feat, num_heads, mlp_ratio))
            prev_ch = feat

        # Final
        self.final = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        enc_feats = []
        # Encoder path
        for conv, trans in zip(self.down_convs, self.trans_enc):
            x = conv(x)
            x = trans(x)
            enc_feats.append(x)
            x = self.pool(x)
        # Bottleneck
        x = self.bottleneck(x)
        # Decoder path
        for up, dec, trans, enc in zip(
            self.up_convs, self.dec_convs, self.trans_dec, reversed(enc_feats[:-1])
        ):
            x = up(x)
            if x.shape[-2:] != enc.shape[-2:]:
                x = F.interpolate(x, size=enc.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([enc, x], dim=1)
            x = dec(x)
            x = trans(x)
        # Final conv + upsample to exact input size
        out = self.final(x)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out
