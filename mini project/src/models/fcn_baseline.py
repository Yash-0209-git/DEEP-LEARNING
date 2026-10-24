"""
fcn_baseline.py — Small Fully Convolutional Encoder-Decoder (no skip connections).

Architecture summary (default BASE_CHANNELS=32, IMAGE_SIZE=256):
    Encoder:
        Block 1: Conv3x3(3  -> 32)  + BN + ReLU + MaxPool2x2  -> [32, 128, 128]
        Block 2: Conv3x3(32 -> 64)  + BN + ReLU + MaxPool2x2  -> [64, 64, 64]
        Block 3: Conv3x3(64 -> 128) + BN + ReLU + MaxPool2x2  -> [128, 32, 32]
    Bottleneck:
        Conv3x3(128 -> 256) + BN + ReLU                       -> [256, 32, 32]
    Decoder:
        Block 1: Bilinear 2x + Conv3x3(256 -> 128) + BN + ReLU -> [128, 64, 64]
        Block 2: Bilinear 2x + Conv3x3(128 -> 64)  + BN + ReLU -> [64, 128, 128]
        Block 3: Bilinear 2x + Conv3x3(64  -> 32)  + BN + ReLU -> [32, 256, 256]
    Head:
        Conv1x1(32 -> 1)                                       -> [1, 256, 256]

Output: raw logits. Apply sigmoid for probabilities during inference.

Intentional design choice: no skip connections.  This is the deliberate weakness
that makes the FCN a meaningful comparison baseline against U-Net.
"""

import torch
import torch.nn as nn

from src.config import FCN_BASE_CHANNELS


class _EncoderBlock(nn.Module):
    """Conv3x3 -> BN -> ReLU -> MaxPool2x2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DecoderBlock(nn.Module):
    """Bilinear 2x upsample -> Conv3x3 -> BN -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FCNBaseline(nn.Module):
    """
    Small encoder-decoder without skip connections.

    Args:
        base_channels: Number of output channels in the first encoder block.
                       Subsequent blocks double this (32 -> 64 -> 128 -> 256).
    """

    def __init__(self, base_channels: int = FCN_BASE_CHANNELS):
        super().__init__()
        c = base_channels  # shorthand

        # Encoder: progressively halve spatial resolution.
        self.enc1 = _EncoderBlock(3, c)          # -> [c, H/2, W/2]
        self.enc2 = _EncoderBlock(c, c * 2)      # -> [2c, H/4, W/4]
        self.enc3 = _EncoderBlock(c * 2, c * 4)  # -> [4c, H/8, W/8]

        # Bottleneck: keep spatial size, expand channels.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c * 4, c * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
        )

        # Decoder: progressively halve channels and restore spatial resolution.
        self.dec1 = _DecoderBlock(c * 8, c * 4)  # -> [4c, H/4, W/4]
        self.dec2 = _DecoderBlock(c * 4, c * 2)  # -> [2c, H/2, W/2]
        self.dec3 = _DecoderBlock(c * 2, c)       # -> [c,  H,   W  ]

        # 1x1 conv to produce a single-channel logit map.
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.bottleneck(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        return self.head(x)   # [B, 1, H, W] raw logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from src.config import IMAGE_SIZE
    model = FCNBaseline()
    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = model(dummy)
    print(f"FCNBaseline output shape : {out.shape}")
    print(f"Trainable parameters     : {count_parameters(model):,}")
