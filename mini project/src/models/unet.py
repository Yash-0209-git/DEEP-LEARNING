"""
unet.py — U-Net segmentation model for Road Crack Detection.

Architecture (default BASE_CHANNELS=32):
    Encoder:
        E1: 2x[Conv3x3(3->32)  + BN + ReLU]        -> [32,  H,    W   ] (skip S1)
        MaxPool -> [32,  H/2,  W/2 ]
        E2: 2x[Conv3x3(32->64) + BN + ReLU]        -> [64,  H/2,  W/2 ] (skip S2)
        MaxPool -> [64,  H/4,  W/4 ]
        E3: 2x[Conv3x3(64->128)+ BN + ReLU]        -> [128, H/4,  W/4 ] (skip S3)
        MaxPool -> [128, H/8,  W/8 ]
        E4: 2x[Conv3x3(128->256)+BN + ReLU]        -> [256, H/8,  W/8 ] (skip S4)
        MaxPool -> [256, H/16, W/16]
    Bottleneck:
        2x[Conv3x3(256->512)+BN+ReLU]              -> [512, H/16, W/16]
    Decoder:
        D4: TransposedConv(512->256) + Cat(S4) -> [512, H/8, W/8]
            2x[Conv3x3(512->256)+BN+ReLU]         -> [256, H/8,  W/8 ]
        D3: TransposedConv(256->128) + Cat(S3) -> [256, H/4, W/4]
            2x[Conv3x3(256->128)+BN+ReLU]         -> [128, H/4,  W/4 ]
        D2: TransposedConv(128->64)  + Cat(S2) -> [128, H/2, W/2]
            2x[Conv3x3(128->64) +BN+ReLU]         -> [64,  H/2,  W/2 ]
        D1: TransposedConv(64->32)   + Cat(S1) -> [64,  H,   W  ]
            2x[Conv3x3(64->32)  +BN+ReLU]         -> [32,  H,    W   ]
    Head:
        Conv1x1(32->1)                             -> [1,   H,    W   ]

Output: raw logits. Apply sigmoid during inference.
"""

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    """Two consecutive Conv3x3 -> BN -> ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """MaxPool2x2 followed by a DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    """
    Transposed convolution for 2x upsampling + concatenation of skip feature
    map + DoubleConv.

    The transposed conv halves the channel count.
    After concatenation with the skip map (same channel count), the total
    doubles again, which is then reduced back by DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # in_ch comes from the deeper feature map; halve before cat.
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # After cat: (in_ch // 2) + (in_ch // 2) = in_ch channels.
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against rounding when input size is not a power of 2.
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for binary segmentation.

    Args:
        base_channels: Output channels of the first encoder block.
                       Channels double at each encoder level:
                       [c, 2c, 4c, 8c], bottleneck = 16c.
    """

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = _DoubleConv(3, c)          # [c, H, W]      skip S1
        self.enc2 = _Down(c, c * 2)            # [2c, H/2, W/2] skip S2
        self.enc3 = _Down(c * 2, c * 4)        # [4c, H/4, W/4] skip S3
        self.enc4 = _Down(c * 4, c * 8)        # [8c, H/8, W/8] skip S4

        # Bottleneck
        self.bottleneck = _Down(c * 8, c * 16)  # [16c, H/16, W/16]

        # Decoder
        self.dec4 = _Up(c * 16, c * 8)   # cat with S4 -> [8c,  H/8, W/8]
        self.dec3 = _Up(c * 8,  c * 4)   # cat with S3 -> [4c,  H/4, W/4]
        self.dec2 = _Up(c * 4,  c * 2)   # cat with S2 -> [2c,  H/2, W/2]
        self.dec1 = _Up(c * 2,  c)       # cat with S1 -> [c,   H,   W  ]

        # Output head
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path — collect skip maps
        s1 = self.enc1(x)           # [c,   H,    W   ]
        s2 = self.enc2(s1)          # [2c,  H/2,  W/2 ]
        s3 = self.enc3(s2)          # [4c,  H/4,  W/4 ]
        s4 = self.enc4(s3)          # [8c,  H/8,  W/8 ]

        # Bottleneck
        x = self.bottleneck(s4)     # [16c, H/16, W/16]

        # Decoder path — merge with skip maps
        x = self.dec4(x, s4)        # [8c,  H/8,  W/8 ]
        x = self.dec3(x, s3)        # [4c,  H/4,  W/4 ]
        x = self.dec2(x, s2)        # [2c,  H/2,  W/2 ]
        x = self.dec1(x, s1)        # [c,   H,    W   ]

        return self.head(x)         # [1,   H,    W   ] raw logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from src.config import IMAGE_SIZE, UNET_BASE_CHANNELS
    model = UNet(base_channels=UNET_BASE_CHANNELS)
    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = model(dummy)
    print(f"UNet output shape    : {out.shape}")
    print(f"Trainable parameters : {count_parameters(model):,}")
