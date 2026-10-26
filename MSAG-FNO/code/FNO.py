

from __future__ import annotations

import copy
import math
import random
from typing import TypeVar

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

__all__ = [
    "SpectralConv", "FNOOperatorBlock", "FNO", "TensorBatch", "seed_all",
    "masked_depth_loss", "fit_model", "train_fno",
]

TensorBatch = tuple[Tensor, Tensor, Tensor]
ModelT = TypeVar("ModelT", bound=nn.Module)


def _positive_integer(value: int, name: str) -> None:

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def seed_all(seed: int = 2026) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SpectralConv(nn.Module):

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        _positive_integer(width, "width")
        _positive_integer(modes, "modes")
        self.modes = modes
        self.weights = nn.Parameter(
            torch.randn(2, width, width, modes, modes, dtype=torch.cfloat) / width
        )

    def forward(self, x: Tensor) -> Tensor:

        if x.ndim != 4 or x.shape[1] != self.weights.shape[1]:
            raise ValueError("Expected NCHW input with the configured width.")
        batch, channels, height, width = x.shape
        if height < 2 or width < 1:
            raise ValueError("Spectral convolution requires H >= 2 and W >= 1.")
        transformed = torch.fft.rfft2(x, norm="ortho")
        output = torch.zeros(
            batch, channels, height, width // 2 + 1,
            device=x.device, dtype=transformed.dtype,
        )
        modes = min(self.modes, height // 2, width // 2 + 1)
        for corner, rows in enumerate((slice(0, modes), slice(-modes, None))):
            output[:, :, rows, :modes] = torch.einsum(
                "bixy,ioxy->boxy",
                transformed[:, :, rows, :modes],
                self.weights[corner, :, :, :modes, :modes],
            )
        return torch.fft.irfft2(output, s=(height, width), norm="ortho")


class FNOOperatorBlock(nn.Module):


    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.ms = False
        self.ag = False
        self.fine = SpectralConv(width, modes)
        self.local = nn.Conv2d(width, width, 1)

    def forward(self, x: Tensor) -> Tensor:

        return F.gelu(self.fine(x) + self.local(x))


class FNO(nn.Module):


    def __init__(
        self, inputs: int = 17, width: int = 12, modes: int = 5,
        outputs: int = 26,
    ) -> None:
        super().__init__()
        for value, name in ((inputs, "inputs"), (width, "width"),
                            (modes, "modes"), (outputs, "outputs")):
            _positive_integer(value, name)
        self.kind = "FNO"
        self.lift = nn.Conv2d(inputs, width, 1)
        self.blocks = nn.ModuleList([
            FNOOperatorBlock(width, modes) for _ in range(4)
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


def masked_depth_loss(
    prediction: Tensor, target: Tensor, mask: Tensor,
    *, sequence_channels: int = 25,
) -> Tensor:

    if prediction.ndim != 4 or prediction.shape != target.shape:
        raise ValueError("Prediction and target must have the same NCHW shape.")
    if (isinstance(sequence_channels, bool)
            or not isinstance(sequence_channels, int)
            or not 0 < sequence_channels < prediction.shape[1]):
        raise ValueError("sequence_channels must split channels into two groups.")
    if mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean and have exactly the target shape.")
    if target.device != prediction.device or mask.device != prediction.device:
        raise ValueError("Prediction, target and mask must be on the same device.")
    error = (prediction - target).square()
    return 0.5 * (
        (error[:, :sequence_channels] * mask[:, :sequence_channels]).sum()
        / mask[:, :sequence_channels].sum().clamp_min(1)
        + (error[:, sequence_channels:] * mask[:, sequence_channels:]).sum()
        / mask[:, sequence_channels:].sum().clamp_min(1)
    )


def _validate_batch(model: nn.Module, batch: TensorBatch) -> None:

    if not isinstance(batch, tuple) or len(batch) != 3:
        raise TypeError("A batch must be a tuple (inputs, target, boolean_mask).")
    x, target, mask = batch
    if not all(isinstance(value, Tensor) for value in batch):
        raise TypeError("Every batch element must be a torch.Tensor.")
    if x.ndim != 4 or target.ndim != 4:
        raise ValueError("Inputs and targets must use NCHW layout.")
    if x.shape[0] < 1 or x.shape[0] != target.shape[0]:
        raise ValueError("Inputs and targets require the same nonempty batch.")
    if x.shape[-2:] != target.shape[-2:]:
        raise ValueError("Inputs and targets must share their spatial shape.")
    if mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean and have exactly the target shape.")
    parameter = next(p for p in model.parameters() if not p.is_complex())
    if x.dtype != parameter.dtype or target.dtype != parameter.dtype:
        raise TypeError("Inputs and targets must match the model's real dtype.")
    if any(value.device != parameter.device for value in batch):
        raise ValueError("All tensors must already be on the model's device.")
    if not torch.isfinite(x).all() or not torch.isfinite(target).all():
        raise ValueError("Inputs and targets must be finite, including masked cells.")


def fit_model(
    model: ModelT, train_batch: TensorBatch, validation_batch: TensorBatch,
    *, epochs: int, learning_rate: float = 0.003,
    sequence_channels: int = 25, seed: int = 2026,
) -> ModelT:

    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
        raise ValueError("epochs must be a nonnegative integer.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    _validate_batch(model, train_batch)
    _validate_batch(model, validation_batch)
    seed_all(seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    frozen_before = {
        name: p.detach().clone()
        for name, p in model.named_parameters() if not p.requires_grad
    }
    best_loss = math.inf
    best_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate) if parameters else None
    x, target, mask = train_batch
    valid_x, valid_target, valid_mask = validation_batch
    for _ in range(epochs if optimizer is not None else 0):
        model.train()
        optimizer.zero_grad()
        loss = masked_depth_loss(
            model(x), target, mask, sequence_channels=sequence_channels,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss is not finite.")
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(masked_depth_loss(
                model(valid_x), valid_target, valid_mask,
                sequence_channels=sequence_channels,
            ))
        if not math.isfinite(valid_loss):
            raise FloatingPointError("Validation loss is not finite.")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state, strict=True)
    model.eval()
    for name, parameter in model.named_parameters():
        if name in frozen_before and not torch.equal(parameter, frozen_before[name]):
            raise RuntimeError(f"Frozen parameter changed: {name}")
    return model


def train_fno(
    model: FNO, train_batch: TensorBatch, validation_batch: TensorBatch,
    *, epochs: int = 60, learning_rate: float = 0.003,
    sequence_channels: int = 25, seed: int = 2026,
) -> FNO:
 
    if not isinstance(model, FNO):
        raise TypeError("train_fno requires this module's FNO model.")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return fit_model(
        model, train_batch, validation_batch, epochs=epochs,
        learning_rate=learning_rate, sequence_channels=sequence_channels,
        seed=seed,
    )
