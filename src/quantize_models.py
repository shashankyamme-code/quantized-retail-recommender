"""Create and verify separate dynamic and static INT8 baseline artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch

from src.dataset import (
    TOKENIZER_PATH,
    TRANSACTIONS_PATH,
    ProductTokenizer,
    build_next_item_examples,
    load_transaction_baskets,
    split_transactions,
)
from src.evaluate import model_file_size_mb, model_parameter_count
from src.quantization import (
    DYNAMIC_MODEL_PATH,
    STATIC_MODEL_PATH,
    load_quantized_model,
    quantize_dynamic,
    quantize_static,
)
from src.recommend import recommend_next
from src.train import BASELINE_MODEL_PATH, load_model

BASELINE_METRICS_PATH = Path("results") / "baseline_metrics.json"
QUANTIZATION_REPORT_PATH = Path("results") / "quantization_artifacts.json"
SPLIT_SEED = 42


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_examples(examples: list[tuple[list[int], int]]) -> str:
    encoded = json.dumps(examples, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _quantized_linear_report(model: torch.nn.Module, kind: str) -> dict[str, Any]:
    if kind == "dynamic":
        quantized_type = torch.ao.nn.quantized.dynamic.Linear
    elif kind == "static":
        quantized_type = torch.ao.nn.quantized.Linear
    else:
        raise ValueError(f"Unknown quantization kind: {kind}")
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, quantized_type):
            weight = module.weight()
            layers.append({
                "name": name,
                "operator": type(module).__name__,
                "weight_dtype": str(weight.dtype),
                "weight_qscheme": str(weight.qscheme()),
                "weight_shape": list(weight.shape),
            })
    if not layers:
        raise RuntimeError(f"No actual {kind} quantized Linear operators were found.")
    if any(layer["weight_dtype"] != "torch.qint8" for layer in layers):
        raise RuntimeError(f"At least one {kind} Linear weight is not INT8.")
    return {"quantized_linear_count": len(layers), "layers": layers}


def _verify_inference(
    model: torch.nn.Module,
    tokenizer: ProductTokenizer,
    sample_baskets: list[list[str]],
) -> dict[str, Any]:
    checks = []
    for basket in sample_baskets:
        tokens = [tokenizer.bos_id, *tokenizer.encode(basket)]
        input_tensor = torch.tensor([tokens[-model.max_length:]], dtype=torch.long, device="cpu")
        lengths = torch.tensor([input_tensor.shape[1]], dtype=torch.long, device="cpu")
        with torch.inference_mode():
            logits = model(input_tensor, lengths)
        if logits.shape != (1, tokenizer.vocab_size):
            raise RuntimeError(
                f"Expected output shape (1, {tokenizer.vocab_size}), got {tuple(logits.shape)}."
            )
        recommendations = recommend_next(
            basket, model=model, tokenizer=tokenizer, top_k=min(3, len(tokenizer.products)),
        )
        if not recommendations or any(
            item["product"] not in tokenizer.product_to_id for item in recommendations
        ):
            raise RuntimeError(f"Model returned invalid product recommendations for {basket}.")
        checks.append({
            "basket": basket,
            "output_shape": list(logits.shape),
            "recommendations": recommendations,
        })
    return {"device": "cpu", "sample_basket_count": len(checks), "checks": checks}


def quantize_saved_models(
    baseline_path: str | Path = BASELINE_MODEL_PATH,
    tokenizer_path: str | Path = TOKENIZER_PATH,
    dataset_path: str | Path = TRANSACTIONS_PATH,
    baseline_metrics_path: str | Path = BASELINE_METRICS_PATH,
    dynamic_model_path: str | Path = DYNAMIC_MODEL_PATH,
    static_model_path: str | Path = STATIC_MODEL_PATH,
    calibration_batches: int = 32,
    calibration_batch_size: int = 64,
    seed: int = SPLIT_SEED,
    report_path: str | Path = QUANTIZATION_REPORT_PATH,
) -> dict[str, Any]:
    """Quantize copies of the saved baseline; fail if it differs from the evaluated baseline."""
    baseline_path = Path(baseline_path)
    tokenizer_path = Path(tokenizer_path)
    dataset_path = Path(dataset_path)
    baseline_metrics_path = Path(baseline_metrics_path)
    dynamic_model_path = Path(dynamic_model_path)
    static_model_path = Path(static_model_path)

    baseline_hash_before = _sha256_file(baseline_path)
    if baseline_metrics_path.exists():
        baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
        expected_hash = baseline_metrics.get("checkpoint_sha256")
        if expected_hash and expected_hash != baseline_hash_before:
            raise RuntimeError(
                "Baseline checkpoint no longer matches results/baseline_metrics.json; "
                "rerun baseline evaluation before quantizing."
            )

    tokenizer = ProductTokenizer.load(tokenizer_path)
    train_transactions, _, test_transactions = split_transactions(
        load_transaction_baskets(dataset_path), seed=seed,
    )
    baseline = load_model(baseline_path, map_location="cpu")
    if baseline.vocab_size != tokenizer.vocab_size:
        raise ValueError("Baseline checkpoint and product vocabulary sizes do not match.")

    # Calibration uses only prefixes from training transactions, never the held-out test set.
    training_examples = build_next_item_examples(
        train_transactions,
        tokenizer,
        max_length=baseline.max_length,
        shuffle_baskets=True,
        seed=seed,
    )
    calibration_count = min(calibration_batches * calibration_batch_size, len(training_examples))
    calibration_examples = training_examples[:calibration_count]
    if not calibration_examples:
        raise ValueError("No training examples are available for static calibration.")
    test_examples = build_next_item_examples(
        test_transactions, tokenizer, max_length=baseline.max_length,
    )
    if not test_examples:
        raise ValueError("No held-out test examples are available for model verification.")

    dynamic_model = quantize_dynamic(baseline, dynamic_model_path)
    static_model = quantize_static(
        baseline,
        calibration_examples,
        static_model_path,
        calibration_batches=calibration_batches,
        batch_size=calibration_batch_size,
    )

    loaded_models = {
        "baseline": load_model(baseline_path, map_location="cpu"),
        "dynamic_int8": load_quantized_model(dynamic_model_path),
        "static_int8": load_quantized_model(static_model_path),
    }
    sample_baskets = [
        ["Milk", "Bread", "Butter"],
        ["Pasta", "Pasta Sauce"],
        ["Coffee", "Sugar"],
    ]
    sample_baskets = [
        [product for product in basket if product in tokenizer.product_to_id]
        for basket in sample_baskets
    ]
    sample_baskets = [basket for basket in sample_baskets if basket]
    if not sample_baskets:
        sample_baskets = [tokenizer.products[:min(3, len(tokenizer.products))]]
    report: dict[str, Any] = {
        "baseline_checkpoint_sha256_before": baseline_hash_before,
        "baseline_checkpoint_sha256_after": _sha256_file(baseline_path),
        "baseline_checkpoint_unchanged": baseline_hash_before == _sha256_file(baseline_path),
        "vocabulary_path": str(tokenizer_path),
        "vocabulary_size": tokenizer.vocab_size,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "transaction_split_sizes": {
            "train": len(train_transactions),
            "test": len(test_transactions),
        },
        "test_examples_count": len(test_examples),
        "test_examples_sha256": _sha256_examples(test_examples),
        "calibration": {
            "source": "training transaction prefixes only",
            "training_transactions": len(train_transactions),
            "available_training_examples": len(training_examples),
            "calibration_examples_used": calibration_count,
            "calibration_batch_size": calibration_batch_size,
            "calibration_batches": (calibration_count + calibration_batch_size - 1) // calibration_batch_size,
            "calibration_examples_sha256": _sha256_examples(calibration_examples),
            "seed": seed,
            "sequence_max_length": baseline.max_length,
            "transaction_split_seed": seed,
            "basket_order_shuffled": True,
            "calibration_order": "first calibration_examples_used generated training prefixes",
            "activation_observer": "default PyTorch eager qconfig observer",
        },
        "runtime": {
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "cpu_quantized_engines": torch.backends.quantized.supported_engines,
            "selected_quantized_engine": torch.backends.quantized.engine,
            "inference_device": "cpu",
        },
        "models": {},
    }
    descriptions = {
        "baseline": "Unquantized FP32 baseline checkpoint",
        "dynamic_int8": (
            "PyTorch eager dynamic qint8 quantization of Linear modules; activations "
            "are dynamically quantized at runtime."
        ),
        "static_int8": (
            "PyTorch eager static post-training quantization of supported Linear modules "
            "using observed activation ranges from representative training prefixes."
        ),
    }
    paths = {
        "baseline": baseline_path,
        "dynamic_int8": dynamic_model_path,
        "static_int8": static_model_path,
    }
    for name, model in loaded_models.items():
        model_info: dict[str, Any] = {
            "checkpoint_path": str(paths[name]),
            "checkpoint_size_bytes": paths[name].stat().st_size,
            "checkpoint_size_mb": model_file_size_mb(paths[name]),
            "parameter_count": model_parameter_count(model),
            "method": descriptions[name],
            "inference_verification": _verify_inference(model, tokenizer, sample_baskets),
        }
        if name == "dynamic_int8":
            model_info.update(_quantized_linear_report(model, "dynamic"))
        elif name == "static_int8":
            model_info.update(_quantized_linear_report(model, "static"))
            model_info["unsupported_operations_kept_float"] = [
                "Embedding tables",
                "LayerNorm",
                "GELU and Dropout",
                "Attention score and value matrix multiplications",
                "Softmax, causal/padding masks, residual additions, and tensor reshaping",
            ]
        else:
            model_info["quantized_linear_count"] = 0
        config_name = "baseline_quantization_config.json" if name == "baseline" else "quantization_config.json"
        config_path = paths[name].with_name(config_name)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({
            "model": name,
            "method": descriptions[name],
            "baseline_checkpoint_sha256": baseline_hash_before,
            "vocabulary_path": str(tokenizer_path),
            "vocabulary_size": tokenizer.vocab_size,
            "torch_version": torch.__version__,
            "runtime_device": "cpu",
            "quantized_linear_dtype": (
                "torch.qint8" if name in {"dynamic_int8", "static_int8"} else None
            ),
            "static_quantization_engine": (
                torch.backends.quantized.engine if name == "static_int8" else None
            ),
            "static_qconfig": (
                f"torch.ao.quantization.get_default_qconfig({torch.backends.quantized.engine!r})"
                if name == "static_int8" else None
            ),
            "calibration": report["calibration"] if name == "static_int8" else None,
            "quantized_layers": model_info.get("layers", []),
            "unsupported_operations_kept_float": model_info.get(
                "unsupported_operations_kept_float", [],
            ),
        }, indent=2), encoding="utf-8")
        model_info["configuration_path"] = str(config_path)
        report["models"][name] = model_info

    if not report["baseline_checkpoint_unchanged"]:
        raise RuntimeError("The baseline checkpoint changed during quantization.")
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        name: {
            "checkpoint_size_mb": info["checkpoint_size_mb"],
            "parameter_count": info["parameter_count"],
            "quantized_linear_count": info["quantized_linear_count"],
            "checkpoint_path": info["checkpoint_path"],
            "inference_verified": info["inference_verification"]["sample_basket_count"],
        }
        for name, info in report["models"].items()
    }, indent=2))
    print(f"Static calibration examples used: {calibration_count}")
    print(f"Saved quantization report: {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-batches", type=int, default=32)
    parser.add_argument("--calibration-batch-size", type=int, default=64)
    args = parser.parse_args()
    quantize_saved_models(
        calibration_batches=args.calibration_batches,
        calibration_batch_size=args.calibration_batch_size,
    )
