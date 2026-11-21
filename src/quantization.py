"""Dynamic and static INT8 post-training quantization."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.ao.quantization import DeQuantStub, QuantStub

from src.dataset import ProductTokenizer
from src.evaluate import evaluate_model, save_evaluation

MODEL_DIR = Path("models")
DYNAMIC_MODEL_DIR = MODEL_DIR / "dynamic_int8"
STATIC_MODEL_DIR = MODEL_DIR / "static_int8"
DYNAMIC_MODEL_PATH = DYNAMIC_MODEL_DIR / "model.pth"
STATIC_MODEL_PATH = STATIC_MODEL_DIR / "model.pth"


class StaticQuantizedLinear(nn.Module):
    """Quantize/dequantize around a Linear so it can be statically quantized in-place."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.quantize = QuantStub()
        self.linear = linear
        self.dequantize = DeQuantStub()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.linear(self.quantize(inputs)))


def _wrap_linear_modules(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, StaticQuantizedLinear(child))
        else:
            _wrap_linear_modules(child)


def quantize_dynamic(
    model: nn.Module, model_path: str | Path = DYNAMIC_MODEL_PATH,
) -> nn.Module:
    """Deep-copy and apply PyTorch dynamic INT8 quantization to supported Linear layers."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    quantized = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).cpu().eval(), {nn.Linear}, dtype=torch.qint8, inplace=False,
    ).eval()
    converted = [
        module for module in quantized.modules()
        if isinstance(module, torch.ao.nn.quantized.dynamic.Linear)
    ]
    if not converted:
        raise RuntimeError("Dynamic quantization did not convert any Linear modules.")
    if any(module.weight().dtype != torch.qint8 for module in converted):
        raise RuntimeError("Dynamic quantized Linear modules do not contain qint8 weights.")
    torch.save(quantized, model_path)
    return quantized


def quantize_static(
    model: nn.Module,
    calibration_examples: Iterable[tuple[list[int], int]],
    model_path: str | Path = STATIC_MODEL_PATH,
    calibration_batches: int = 32,
    batch_size: int = 64,
    backend: str | None = None,
) -> nn.Module:
    """Calibrate and statically convert Linear layers with real INT8 CPU operators."""
    if not torch.backends.quantized.supported_engines:
        raise RuntimeError("This PyTorch build has no supported quantized CPU engine.")
    if calibration_batches < 1 or batch_size < 1:
        raise ValueError("calibration_batches and batch_size must be positive.")
    engine = backend or (
        "onednn" if "onednn" in torch.backends.quantized.supported_engines else "fbgemm"
    )
    if engine not in torch.backends.quantized.supported_engines:
        raise ValueError(f"Unsupported quantized engine '{engine}'.")
    torch.backends.quantized.engine = engine
    prepared_source = copy.deepcopy(model).cpu().eval()
    _wrap_linear_modules(prepared_source)
    qconfig = torch.ao.quantization.get_default_qconfig(engine)
    for module in prepared_source.modules():
        if isinstance(module, StaticQuantizedLinear):
            module.qconfig = qconfig
    prepared = torch.ao.quantization.prepare(prepared_source, inplace=False)

    examples = list(calibration_examples)
    if not examples:
        raise ValueError("Static quantization requires representative calibration examples.")
    from src.train import NextItemDataset, collate_examples
    from torch.utils.data import DataLoader

    loader = DataLoader(
        NextItemDataset(examples[:calibration_batches * batch_size]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_examples,
    )
    prepared.eval()
    with torch.inference_mode():
        for batch_index, (tokens, lengths, _) in enumerate(loader):
            prepared(tokens, lengths)
            if batch_index + 1 >= calibration_batches:
                break
    quantized = torch.ao.quantization.convert(prepared, inplace=False).eval()
    converted = [
        module for module in quantized.modules()
        if isinstance(module, torch.ao.nn.quantized.Linear)
    ]
    if not converted:
        raise RuntimeError("Static quantization did not produce quantized Linear operators.")
    if any(module.weight().dtype != torch.qint8 for module in converted):
        raise RuntimeError("Static quantized Linear modules do not contain qint8 weights.")
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized, model_path)
    return quantized


def load_quantized_model(path: str | Path) -> nn.Module:
    model = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected a PyTorch model checkpoint at {path}.")
    return model.eval()


def evaluate_quantized_models(
    baseline: nn.Module,
    test_transactions: list[list[str]],
    tokenizer: ProductTokenizer,
    calibration_examples: list[tuple[list[int], int]],
) -> dict[str, dict[str, float | int | None]]:
    results: dict[str, dict[str, float | int | None]] = {}
    baseline_path = Path("models") / "baseline_model.pth"
    if not baseline_path.exists():
        torch.save(baseline, baseline_path)
    for name, model, path in (
        ("baseline", baseline, baseline_path),
        ("dynamic_int8", quantize_dynamic(baseline), DYNAMIC_MODEL_PATH),
        (
            "static_int8",
            quantize_static(baseline, calibration_examples),
            STATIC_MODEL_PATH,
        ),
    ):
        metrics = evaluate_model(model.cpu().eval(), test_transactions, tokenizer, path)
        save_evaluation(name, metrics)
        results[name] = metrics
        print(
            f"{name}: size={metrics['model_size_mb']:.3f} MB, "
            f"top1={metrics['top1_accuracy']:.3f}, top3={metrics['top3_accuracy']:.3f}, "
            f"latency={metrics['average_inference_time_ms']:.3f} ms"
        )
    return results
