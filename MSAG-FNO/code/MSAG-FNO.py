

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from FNO import SpectralConv, TensorBatch, _positive_integer, fit_model

__all__ = ["MSAGOperatorBlock", "MSAGFNO", "train_msag_fno"]


class MSAGOperatorBlock(nn.Module):


    def __init__(
        self, width: int, modes: int, ms: bool = True, ag: bool = True,
    ) -> None:
        super().__init__()
        self.ms = ms
        self.ag = ag
        self.fine = SpectralConv(width, modes)
        self.local = nn.Conv2d(width, width, 1)
        if ms:
            self.coarse = SpectralConv(width, max(2, modes // 2))
        if ag:
            bottleneck = max(2, width // 4)
            self.channel = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(width, bottleneck, 1),
                nn.GELU(),
                nn.Conv2d(bottleneck, width, 1),
                nn.Sigmoid(),
            )
            self.spatial = nn.Sequential(
                nn.Conv2d(2, 1, 5, padding=2), nn.Sigmoid(),
            )

    def forward(self, x: Tensor) -> Tensor:

        if x.ndim != 4:
            raise ValueError("Expected a four-dimensional NCHW tensor.")
        if self.ms and (x.shape[-2] < 4 or x.shape[-1] < 2):
            raise ValueError("The coarse branch requires H >= 4 and W >= 2.")
        spectral = self.fine(x)
        if self.ms:
            coarse = self.coarse(F.avg_pool2d(x, 2))
            coarse = F.interpolate(
                coarse, size=x.shape[-2:], mode="bilinear", align_corners=False,
            )
            spectral = (spectral + coarse) / 2
        if self.ag:
            channel_gate = self.channel(x)
            spatial_descriptor = torch.cat(
                [x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1,
            )
            gate = channel_gate * self.spatial(spatial_descriptor)
            spectral = spectral * gate
        return F.gelu(spectral + self.local(x))


class MSAGFNO(nn.Module):



    def __init__(
        self, inputs: int = 17, width: int = 12, modes: int = 5,
        outputs: int = 26, *, kind: str = "MSAG-FNO",
    ) -> None:
        super().__init__()
        if kind not in ("MSAG-FNO", "MS-FNO", "AG-FNO"):
            raise ValueError("kind must be MSAG-FNO, MS-FNO or AG-FNO.")
        for value, name in ((inputs, "inputs"), (width, "width"),
                            (modes, "modes"), (outputs, "outputs")):
            _positive_integer(value, name)
        self.kind = kind
        self.lift = nn.Conv2d(inputs, width, 1)
        self.blocks = nn.ModuleList([
            MSAGOperatorBlock(
                width, modes,
                ms=kind in ("MS-FNO", "MSAG-FNO"),
                ag=kind in ("AG-FNO", "MSAG-FNO"),
            )
            for _ in range(4)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv2d(width * 2, outputs, 1),
        )
        nn.init.constant_(self.head[-1].bias, -2.0)

    def forward(self, x: Tensor, features: bool = False):

        if x.ndim != 4 or x.shape[1] != self.lift.in_channels:
            raise ValueError("Expected input shape (N, inputs, H, W).")
        x = self.lift(x)
        for layer in self.blocks:
            x = layer(x)
        prediction = F.softplus(self.head(x))
        return (prediction, x.mean((2, 3))) if features else prediction


def train_msag_fno(
    model: MSAGFNO, train_batch: TensorBatch, validation_batch: TensorBatch,
    *, epochs: int = 60, learning_rate: float = 0.003,
    sequence_channels: int = 25, seed: int = 2026,
) -> MSAGFNO:

    if not isinstance(model, MSAGFNO):
        raise TypeError("train_msag_fno requires this module's MSAGFNO model.")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return fit_model(
        model, train_batch, validation_batch, epochs=epochs,
        learning_rate=learning_rate, sequence_channels=sequence_channels,
        seed=seed,
    )
