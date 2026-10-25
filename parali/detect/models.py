"""Burn-scar segmentation models: a small U-Net trained from scratch, and the Prithvi-EO-2.0
foundation model (tiny, 5.8M encoder parameters, Apache-2.0) with a light convolutional decoder.

Both take 6 HLS bands (Blue, Green, Red, NIR narrow, SWIR1, SWIR2) and output 2-class logits.
"""

from __future__ import annotations

import torch
from torch import nn

# Prithvi-EO-2.0 pre-training statistics (reflectance x 10000), used to normalise every input.
MEAN = torch.tensor([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]).view(1, 6, 1, 1)
STD = torch.tensor([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]).view(1, 6, 1, 1)


def normalise(reflectance: torch.Tensor) -> torch.Tensor:
    """reflectance in 0..1 -> standardised input."""
    return (reflectance * 10000.0 - MEAN.to(reflectance)) / STD.to(reflectance)


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Classic U-Net, 4 levels, ~1.9M parameters at base=24."""

    def __init__(self, in_ch: int = 6, n_classes: int = 2, base: int = 24):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList(
            [_block(in_ch, c[0]), _block(c[0], c[1]), _block(c[1], c[2]), _block(c[2], c[3])]
        )
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(3)])
        self.dec = nn.ModuleList([_block(c[i] * 2, c[i]) for i in range(3)])
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < 3:
                skips.append(x)
                x = self.pool(x)
        for i in reversed(range(3)):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], dim=1))
        return self.head(x)


class PrithviSeg(nn.Module):
    """Prithvi-EO-2.0 encoder + a progressive-upsampling decoder (features from 4 depths)."""

    def __init__(
        self, weights: str | None, n_classes: int = 2, img_size: int = 224, freeze_encoder: bool = False
    ):
        super().__init__()
        from parali.detect.prithvi_mae import PrithviViT

        self.encoder = PrithviViT(
            img_size=img_size,
            patch_size=(1, 16, 16),
            num_frames=1,
            in_chans=6,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4,
            coords_encoding=[],
            coords_scale_learn=True,
        )
        if weights:
            state = torch.load(weights, map_location="cpu", weights_only=True)
            enc = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
            enc.pop("pos_embed", None)  # recomputed for this image size and single frame
            self.load_report = self.encoder.load_state_dict(enc, strict=False)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.taps = [2, 5, 8, 11]
        d = 192
        self.fuse = nn.Conv2d(d * 4, 256, 1)
        self.decoder = nn.Sequential(
            *[
                layer
                for cin, cout in [(256, 128), (128, 64), (64, 32), (32, 32)]
                for layer in (
                    nn.ConvTranspose2d(cin, cout, 2, stride=2),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(cout, cout, 3, padding=1),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                )
            ]
        )
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder.forward_features(x)
        maps = self.encoder.prepare_features_for_image_model([feats[i] for i in self.taps])
        y = self.fuse(torch.cat(maps, dim=1))  # (B, 256, H/16, W/16)
        return self.head(self.decoder(y))  # 4 x upsample by 2 = H, W
