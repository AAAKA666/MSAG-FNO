

from __future__ import annotations

import copy
import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from FNO import TensorBatch, fit_model

__all__ = [
    "STRATEGIES", "MSAGFNO", "set_finetune", "prepare_finetuning",
    "load_pretrained_model", "fine_tune",
]

STRATEGIES = ("all", "none", "first1", "first2", "first3")


def _load_msag_module():

    module_name = "msag_fno_model"
    module_path = Path(__file__).resolve().with_name("MSAG-FNO.py")
    if module_name in sys.modules:
        existing = sys.modules[module_name]
        if Path(existing.__file__).resolve() != module_path:
            raise ImportError("msag_fno_model already refers to a different file.")
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import the sibling MSAG-FNO.py module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


MSAGFNO = _load_msag_module().MSAGFNO


def _check_model(model: nn.Module) -> None:

    if getattr(model, "kind", None) != "MSAG-FNO":
        raise TypeError("A full MSAG-FNO model is required for these strategies.")
    if not hasattr(model, "blocks") or len(model.blocks) != 4:
        raise TypeError("Expected four sequential operator blocks.")
    if not all(getattr(block, "ms", False) and getattr(block, "ag", False)
               for block in model.blocks):
        raise TypeError("All blocks must enable both multi-scale and gating.")


def set_finetune(model: nn.Module, strategy: str) -> list[str]:

    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}.")
    _check_model(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if strategy == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif strategy != "none":
        count = int(strategy.replace("first", ""))
        for block in model.blocks[:count]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
    return [name for name, p in model.named_parameters() if p.requires_grad]


def prepare_finetuning(pretrained_model: nn.Module, strategy: str) -> nn.Module:

    _check_model(pretrained_model)
    model = copy.deepcopy(pretrained_model)
    set_finetune(model, strategy)
    return model


def load_pretrained_model(
    checkpoint: Mapping[str, Any], *, inputs: int = 17, width: int = 12,
    modes: int = 5, outputs: int = 26, strategy: str = "none",
    device: str | torch.device = "cpu",
) -> MSAGFNO:

    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}.")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be an in-memory mapping.")
    if "state_dict" in checkpoint:
        for key, expected in (("kind", "MSAG-FNO"), ("width", width), ("modes", modes)):
            if key in checkpoint and checkpoint[key] != expected:
                raise ValueError(f"Checkpoint {key} does not match the configuration.")
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, Mapping) or not state:
        raise TypeError("Expected a nonempty state_dict mapping.")
    if not all(isinstance(key, str) and isinstance(value, Tensor)
               for key, value in state.items()):
        raise TypeError("state_dict must contain string keys and tensor values.")
    model = MSAGFNO(inputs=inputs, width=width, modes=modes, outputs=outputs)
    model.load_state_dict(state, strict=True)
    model.to(device)
    set_finetune(model, strategy)
    model.eval()
    return model


def fine_tune(
    pretrained_model: nn.Module, train_batch: TensorBatch,
    validation_batch: TensorBatch, *, strategy: str = "all",
    epochs: int = 40, learning_rate: float = 0.003,
    sequence_channels: int = 25, seed: int = 2026,
) -> nn.Module:

    model = prepare_finetuning(pretrained_model, strategy)
    return fit_model(
        model, train_batch, validation_batch, epochs=epochs,
        learning_rate=learning_rate, sequence_channels=sequence_channels,
        seed=seed,
    )
