# models/nafnet.py
import torch
import torch.nn as nn
from . import register_model
from NAFNet.models.nafn import NAFNet as _BaseNAFNet

@register_model('nafnet')
class NAFNetModel(_BaseNAFNet):
    """
    Official NAFNet wrapped to output 5 channels:
      - 2 for UV flow
      - 3 for RGB reconstruction
    """
    def __init__(self, in_ch=3, out_ch=5,
                 width=48, enc_blks=[1,1,1,28,1], middle_blk_num=1):
        super().__init__(
            img_channel=in_ch,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blks,
            dec_blk_nums=enc_blks[::-1],
        )
        # override final conv to produce `out_ch` instead of 3
        self.final = nn.Conv2d(width, out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out = super().forward(x)    # (B, width, H, W)
        return self.final(out)      # (B, out_ch, H, W)
