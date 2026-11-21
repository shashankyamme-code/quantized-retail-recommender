"""Held-out accuracy, model-size, latency, comparison, and chart utilities."""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.dataset import ProductTokenizer, build_next_item_examples
from src.train import NextItemDataset, collate_examples

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "performance_plots"


def model_memory_size_bytes(model: torch.nn.Module) -> int:
    """Count unique parameter and buffer storages currently resident in the model."""
    seen_storages: set[tuple[str, int]] = set()
    total_bytes = 0
    for tensor in (*model.parameters(), *model.buffers()):
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage.data_ptr())
        if identity not in seen_storages:
            total_bytes += storage.nbytes()
            seen_storages.add(identity)
    return total_bytes


def examples_sha256(examples: list[tuple[list[int], int]]) -> str:
    serialized = json.dumps(examples, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def benchmark_cpu_inference(
    model: torch.nn.Module,
    examples: list[tuple[list[int], int]],
    warmup_iterations: int = 20,
    repetitions: int = 5,
    sample_count: int = 100,
) -> dict[str, float | int | list[float] | str]:
    """Benchmark repeatable single-example CPU inference on a fixed test-prefix sample."""
    if not examples:
        raise ValueError("CPU inference benchmarking requires at least one test example.")
    if warmup_iterations < 0 or repetitions < 1 or sample_count < 1:
        raise ValueError("warmup_iterations must be non-negative; repetitions and sample_count must be positive.")
    fixed_examples = examples[:min(sample_count, len(examples))]
    inputs = [
        (
            torch.tensor([context], dtype=torch.long, device="cpu"),
            torch.tensor([len(context)], dtype=torch.long, device="cpu"),
        )
        for context, _ in fixed_examples
    ]
    cpu_model = model.cpu().eval()
    with torch.inference_mode():
        for index in range(warmup_iterations):
            tokens, lengths = inputs[index % len(inputs)]
            cpu_model(tokens, lengths)

        repetition_latencies: list[float] = []
        sample_latencies_ms: list[float] = []
        for _ in range(repetitions):
            started = time.perf_counter()
            for tokens, lengths in inputs:
                sample_started = time.perf_counter()
                cpu_model(tokens, lengths)
                sample_latencies_ms.append((time.perf_counter() - sample_started) * 1000)
            elapsed_ms = (time.perf_counter() - started) * 1000
            repetition_latencies.append(elapsed_ms / len(inputs))

    return {
        "device": "cpu",
        "warmup_iterations": warmup_iterations,
        "repetitions": repetitions,
        "samples_per_repetition": len(inputs),
        "average_latency_ms": sum(repetition_latencies) / len(repetition_latencies),
        "median_latency_ms": float(pd.Series(sample_latencies_ms).median()),
        "p95_latency_ms": float(pd.Series(sample_latencies_ms).quantile(0.95)),
        "repetition_average_latency_ms": repetition_latencies,
        "input_sha256": examples_sha256(fixed_examples),
        "benchmark_method": "single-example CPU forward pass; same ordered test prefixes for each repetition",
    }


def model_file_size_mb(model_path: str | Path) -> float:
    return Path(model_path).stat().st_size / 1_000_000


def model_parameter_count(model: torch.nn.Module) -> int:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    quantized_linear_types = (
        torch.ao.nn.quantized.Linear,
        torch.ao.nn.quantized.dynamic.Linear,
    )
    for module in model.modules():
        if isinstance(module, quantized_linear_types):
            parameter_count += module.weight().numel()
            bias = module.bias()
            if bias is not None:
                parameter_count += bias.numel()
    return parameter_count


def evaluate_model(
    model: torch.nn.Module,
    transactions: list[list[str]],
    tokenizer: ProductTokenizer,
    model_path: str | Path | None = None,
    batch_size: int = 256,
    latency_samples: int = 100,
    warmup_iterations: int = 20,
    benchmark_repetitions: int = 5,
) -> dict[str, Any]:
    examples = build_next_item_examples(transactions, tokenizer)
    if not examples:
        raise ValueError("The held-out transactions produced no next-item examples.")
    if latency_samples < 1:
        raise ValueError("latency_samples must be at least one.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    loader = DataLoader(
        NextItemDataset(examples), batch_size=batch_size, shuffle=False, collate_fn=collate_examples,
    )
    model.cpu().eval()
    loss_sum = 0.0
    total = top1 = top3 = 0
    with torch.inference_mode():
        for tokens, lengths, targets in loader:
            logits = model(tokens, lengths)
            loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
            ranking_logits = logits.clone()
            for token_id in tokenizer.special_token_ids.values():
                if token_id < ranking_logits.shape[-1]:
                    ranking_logits[:, token_id] = torch.finfo(ranking_logits.dtype).min
            ranking_logits = ranking_logits[:, :len(tokenizer.products) + 1]
            predictions = ranking_logits.topk(min(3, len(tokenizer.products)), dim=-1).indices
            top1 += (predictions[:, :1] == targets[:, None]).any(dim=1).sum().item()
            top3 += (predictions == targets[:, None]).any(dim=1).sum().item()
            total += len(targets)

    benchmark = benchmark_cpu_inference(
        model,
        examples,
        warmup_iterations=warmup_iterations,
        repetitions=benchmark_repetitions,
        sample_count=latency_samples,
    )
    return {
        "test_loss": loss_sum / total,
        "top1_accuracy": top1 / total,
        "top3_accuracy": top3 / total,
        "model_size_mb": model_file_size_mb(model_path) if model_path else None,
        "parameter_count": model_parameter_count(model),
        "average_inference_time_ms": benchmark["average_latency_ms"],
        "inference_benchmark": benchmark,
        "test_examples": total,
    }


def save_evaluation(name: str, metrics: dict[str, float | int | None]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}_evaluation.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def build_comparison(results: dict[str, dict[str, float | int | None]]) -> pd.DataFrame:
    labels = {
        "baseline": "Baseline",
        "dynamic_int8": "Dynamic INT8",
        "static_int8": "Static INT8",
    }
    if not all(name in results for name in labels):
        raise ValueError("Comparison requires baseline, dynamic_int8, and static_int8 results.")
    baseline_size = float(results["baseline"]["model_size_mb"])
    baseline_accuracy = float(results["baseline"]["top1_accuracy"])
    baseline_latency = float(results["baseline"]["average_inference_time_ms"])
    rows = []
    for key, label in labels.items():
        metrics = results[key]
        size = float(metrics["model_size_mb"])
        accuracy = float(metrics["top1_accuracy"])
        rows.append({
            "model": label,
            "model_size_mb": size,
            "top1_accuracy": accuracy,
            "top3_accuracy": float(metrics["top3_accuracy"]),
            "test_loss": float(metrics["test_loss"]),
            "average_inference_time_ms": float(metrics["average_inference_time_ms"]),
            "size_reduction_percent": (baseline_size - size) / baseline_size * 100 if baseline_size else 0,
            "top1_accuracy_difference": accuracy - baseline_accuracy,
            "absolute_top1_accuracy_difference": abs(accuracy - baseline_accuracy),
            "relative_top1_accuracy_degradation_percent": (
                (baseline_accuracy - accuracy) / baseline_accuracy * 100 if baseline_accuracy else 0
            ),
            "inference_time_difference_ms": (
                float(metrics["average_inference_time_ms"]) - baseline_latency
            ),
            "inference_time_change_percent": (
                (float(metrics["average_inference_time_ms"]) - baseline_latency)
                / baseline_latency * 100 if baseline_latency else 0
            ),
            "parameter_count": int(metrics["parameter_count"]),
        })
    frame = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    _create_comparison_plots(frame)
    return frame


def _create_comparison_plots(frame: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    charts = (
        ("model_size_mb", "Model size (MB)", "Model size comparison", "model_size_comparison.png"),
        ("top1_accuracy", "Accuracy", "Top-1 accuracy comparison", "accuracy_comparison.png"),
        (
            "average_inference_time_ms", "Average inference time (ms)",
            "Inference latency comparison", "inference_time_comparison.png",
        ),
    )
    for column, ylabel, title, filename in charts:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(frame["model"], frame[column], color=["#264653", "#2a9d8f", "#e9c46a"])
        axis.set(title=title, ylabel=ylabel)
        axis.tick_params(axis="x", rotation=12)
        figure.tight_layout()
        figure.savefig(PLOTS_DIR / filename, dpi=150)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 5))
    for _, row in frame.iterrows():
        axis.scatter(row["model_size_mb"], row["top1_accuracy"], s=100, label=row["model"])
    axis.set(title="Compression versus accuracy", xlabel="Model size (MB)", ylabel="Top-1 accuracy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "compression_accuracy_tradeoff.png", dpi=150)
    plt.close(figure)
