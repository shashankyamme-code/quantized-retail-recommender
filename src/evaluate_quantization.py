"""Compare saved baseline and INT8 checkpoints on the identical held-out test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.dataset import (
    TOKENIZER_PATH,
    TRANSACTIONS_PATH,
    ProductTokenizer,
    build_next_item_examples,
    load_transaction_baskets,
    split_transactions,
)
from src.evaluate import evaluate_model, examples_sha256, model_parameter_count
from src.quantization import DYNAMIC_MODEL_PATH, STATIC_MODEL_PATH, load_quantized_model
from src.recommend import recommend_next
from src.train import BASELINE_MODEL_PATH, load_model

BASELINE_METRICS_PATH = Path("results") / "baseline_metrics.json"
QUANTIZATION_REPORT_PATH = Path("results") / "quantization_artifacts.json"
RESULTS_DIR = Path("results")
SPLIT_SEED = 42
MODEL_PATHS = {
    "baseline": BASELINE_MODEL_PATH,
    "dynamic_int8": DYNAMIC_MODEL_PATH,
    "static_int8": STATIC_MODEL_PATH,
}
MODEL_LABELS = {
    "baseline": "Baseline",
    "dynamic_int8": "Dynamic INT8",
    "static_int8": "Static INT8",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_change(value: float, reference: float) -> float | None:
    return (value - reference) / reference * 100 if reference else None


def _quantized_linear_count(model: torch.nn.Module, model_name: str) -> int:
    if model_name == "dynamic_int8":
        operator_type = torch.ao.nn.quantized.dynamic.Linear
    elif model_name == "static_int8":
        operator_type = torch.ao.nn.quantized.Linear
    else:
        return 0
    layers = [module for module in model.modules() if isinstance(module, operator_type)]
    if not layers or any(layer.weight().dtype != torch.qint8 for layer in layers):
        raise RuntimeError(f"{model_name} checkpoint has no valid qint8 Linear operators.")
    return len(layers)


def _verify_recommendations(
    model: torch.nn.Module,
    tokenizer: ProductTokenizer,
    examples: list[list[str]],
) -> list[dict[str, Any]]:
    checks = []
    for basket in examples:
        recommendations = recommend_next(
            basket, model=model, tokenizer=tokenizer, top_k=min(3, len(tokenizer.products)),
        )
        if not recommendations or any(
            item["product"] not in tokenizer.product_to_id for item in recommendations
        ):
            raise RuntimeError(f"Invalid recommendations for basket {basket}.")
        checks.append({"basket": basket, "top_3": recommendations})
    return checks


def _comparison_rows(
    model_metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline = model_metrics["baseline"]
    baseline_size = float(baseline["model_size_mb"])
    baseline_top1 = float(baseline["top1_accuracy"])
    baseline_top3 = float(baseline["top3_accuracy"])
    baseline_latency = float(baseline["average_inference_time_ms"])
    rows = []
    for name in MODEL_LABELS:
        metrics = model_metrics[name]
        size = float(metrics["model_size_mb"])
        top1 = float(metrics["top1_accuracy"])
        top3 = float(metrics["top3_accuracy"])
        latency = float(metrics["average_inference_time_ms"])
        rows.append({
            "model_key": name,
            "model": MODEL_LABELS[name],
            "checkpoint_size_bytes": int(metrics["checkpoint_size_bytes"]),
            "model_size_mb": size,
            "parameter_count": int(metrics["parameter_count"]),
            "quantized_linear_count": int(metrics["quantized_linear_count"]),
            "top1_accuracy": top1,
            "top1_accuracy_percent": top1 * 100,
            "top3_accuracy": top3,
            "top3_accuracy_percent": top3 * 100,
            "test_loss": float(metrics["test_loss"]),
            "average_cpu_inference_latency_ms": latency,
            "median_cpu_inference_latency_ms": float(
                metrics["inference_benchmark"]["median_latency_ms"]
            ),
            "p95_cpu_inference_latency_ms": float(
                metrics["inference_benchmark"]["p95_latency_ms"]
            ),
            "size_reduction_percent": (
                (baseline_size - size) / baseline_size * 100 if baseline_size else None
            ),
            "top1_accuracy_change": top1 - baseline_top1,
            "top1_accuracy_change_percentage_points": (top1 - baseline_top1) * 100,
            "top1_relative_accuracy_change_percent": _relative_change(top1, baseline_top1),
            "top3_accuracy_change": top3 - baseline_top3,
            "top3_accuracy_change_percentage_points": (top3 - baseline_top3) * 100,
            "top3_relative_accuracy_change_percent": _relative_change(top3, baseline_top3),
            "latency_change_ms": latency - baseline_latency,
            "latency_improvement_percent": (
                (baseline_latency - latency) / baseline_latency * 100
                if baseline_latency else None
            ),
            "inference_speedup_ratio": baseline_latency / latency if latency else None,
            "test_examples": int(metrics["test_examples"]),
        })
    return rows


def _create_comparison_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [row["model"] for row in rows]
    colors = ["#264653", "#2a9d8f", "#e9c46a"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(labels, [row["model_size_mb"] for row in rows], color=colors)
    axes[0, 0].set(title="Serialized checkpoint size", ylabel="Size (MB)")

    positions = range(len(rows))
    width = 0.36
    axes[0, 1].bar(
        [position - width / 2 for position in positions],
        [row["top1_accuracy_percent"] for row in rows],
        width,
        label="Top-1",
        color="#2a9d8f",
    )
    axes[0, 1].bar(
        [position + width / 2 for position in positions],
        [row["top3_accuracy_percent"] for row in rows],
        width,
        label="Top-3",
        color="#e9c46a",
    )
    axes[0, 1].set(
        title="Held-out recommendation accuracy",
        ylabel="Accuracy (%)",
        xticks=list(positions),
        xticklabels=labels,
    )
    axes[0, 1].legend()

    axes[1, 0].bar(
        labels,
        [row["average_cpu_inference_latency_ms"] for row in rows],
        color=colors,
    )
    axes[1, 0].set(title="CPU inference latency", ylabel="Average latency (ms/example)")

    for row, color in zip(rows, colors):
        axes[1, 1].scatter(
            row["model_size_mb"], row["top1_accuracy_percent"],
            s=110, color=color, label=row["model"],
        )
    axes[1, 1].set(
        title="Compression and accuracy trade-off",
        xlabel="Checkpoint size (MB)",
        ylabel="Top-1 accuracy (%)",
    )
    axes[1, 1].legend()

    for axis in (axes[0, 0], axes[1, 0]):
        axis.tick_params(axis="x", rotation=10)
    figure.suptitle("Baseline and INT8 quantization comparison", fontsize=15)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _make_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    quantized = [row for row in rows if row["model_key"] != "baseline"]
    smallest = min(quantized, key=lambda row: row["model_size_mb"])
    best_accuracy = max(quantized, key=lambda row: row["top1_accuracy"])
    best_accuracy_overall = max(rows, key=lambda row: row["top1_accuracy"])
    best_top3_overall = max(rows, key=lambda row: row["top3_accuracy"])
    baseline = next(row for row in rows if row["model_key"] == "baseline")
    fastest = min(rows, key=lambda row: row["average_cpu_inference_latency_ms"])

    # Accuracy-first among quantized models; ties prefer smaller file then lower latency.
    best_tradeoff = max(
        quantized,
        key=lambda row: (
            row["top1_accuracy"],
            -row["model_size_mb"],
            -row["average_cpu_inference_latency_ms"],
        ),
    )

    def latency_sentence(row: dict[str, Any]) -> str:
        improvement = row["latency_improvement_percent"]
        if improvement is None:
            return f"{row['model']} relative latency change is undefined because baseline latency is zero."
        if improvement > 0:
            return f"{row['model']} was faster by {improvement:.2f}% versus baseline."
        if improvement < 0:
            return f"{row['model']} was slower by {abs(improvement):.2f}% versus baseline."
        return f"{row['model']} matched baseline latency."

    compression_summary = ", ".join(
        f"{row['model']}: {row['size_reduction_percent']:.2f}%"
        for row in quantized
    )
    accuracy_summary = "; ".join(
        f"{row['model']}: {row['top1_accuracy_change_percentage_points']:+.2f} pp "
        f"(relative {row['top1_relative_accuracy_change_percent']:+.2f}%)"
        if row["top1_relative_accuracy_change_percent"] is not None
        else f"{row['model']}: {row['top1_accuracy_change_percentage_points']:+.2f} pp "
        "(relative change undefined; baseline accuracy is zero)"
        for row in quantized
    )
    latency_summary = "; ".join(latency_sentence(row) for row in quantized)
    return {
        "greatest_size_reduction": {
            "model": smallest["model"],
            "size_reduction_percent": smallest["size_reduction_percent"],
            "all_quantized_size_reductions": compression_summary,
        },
        "best_quantized_top1_accuracy": {
            "model": best_accuracy["model"],
            "top1_accuracy": best_accuracy["top1_accuracy"],
            "top3_accuracy": best_accuracy["top3_accuracy"],
        },
        "best_recommendation_accuracy_overall": {
            "top1_model": best_accuracy_overall["model"],
            "top1_accuracy": best_accuracy_overall["top1_accuracy"],
            "top3_model": best_top3_overall["model"],
            "top3_accuracy": best_top3_overall["top3_accuracy"],
        },
        "latency": {
            "fastest_model_overall": fastest["model"],
            "fastest_latency_ms": fastest["average_cpu_inference_latency_ms"],
            "quantized_vs_baseline": latency_summary,
        },
        "accuracy_cost": accuracy_summary,
        "best_tradeoff": {
            "model": best_tradeoff["model"],
            "selection_rule": (
                "Among quantized variants, prioritize measured Top-1 accuracy; break ties "
                "with smaller checkpoint size, then lower CPU latency."
            ),
            "top1_accuracy_change_percentage_points": (
                best_tradeoff["top1_accuracy_change_percentage_points"]
            ),
            "size_reduction_percent": best_tradeoff["size_reduction_percent"],
            "latency_improvement_percent": best_tradeoff["latency_improvement_percent"],
            "baseline_fastest_note": (
                f"{fastest['model']} had the lowest measured latency; baseline latency was "
                f"{baseline['average_cpu_inference_latency_ms']:.4f} ms/example."
            ),
        },
    }


def run_quantization_comparison(
    baseline_path: str | Path = BASELINE_MODEL_PATH,
    dynamic_path: str | Path = DYNAMIC_MODEL_PATH,
    static_path: str | Path = STATIC_MODEL_PATH,
    tokenizer_path: str | Path = TOKENIZER_PATH,
    dataset_path: str | Path = TRANSACTIONS_PATH,
    baseline_metrics_path: str | Path = BASELINE_METRICS_PATH,
    quantization_report_path: str | Path = QUANTIZATION_REPORT_PATH,
    results_dir: str | Path = RESULTS_DIR,
    warmup_iterations: int = 20,
    repetitions: int = 5,
    benchmark_samples: int = 100,
    cpu_threads: int | None = None,
) -> dict[str, Any]:
    """Reload and reevaluate all saved models under identical held-out CPU conditions."""
    paths = {
        "baseline": Path(baseline_path),
        "dynamic_int8": Path(dynamic_path),
        "static_int8": Path(static_path),
    }
    results_dir = Path(results_dir)
    for path in (*paths.values(), Path(tokenizer_path), Path(dataset_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation artifact does not exist: {path}")
    baseline_reference = json.loads(Path(baseline_metrics_path).read_text(encoding="utf-8"))
    quantization_reference = json.loads(
        Path(quantization_report_path).read_text(encoding="utf-8")
    )

    baseline_hash = _sha256_file(paths["baseline"])
    if baseline_hash != baseline_reference.get("checkpoint_sha256"):
        raise RuntimeError("Baseline checkpoint hash differs from its saved baseline metrics.")
    dataset_hash = _sha256_file(Path(dataset_path))
    if dataset_hash != baseline_reference.get("dataset_sha256"):
        raise RuntimeError("Dataset hash differs from the dataset used by baseline evaluation.")
    if dataset_hash != quantization_reference.get("dataset_sha256"):
        raise RuntimeError("Dataset hash differs from the quantization artifact report.")

    tokenizer = ProductTokenizer.load(tokenizer_path)
    transactions = load_transaction_baskets(dataset_path)
    _, _, test_transactions = split_transactions(transactions, seed=SPLIT_SEED)
    examples = build_next_item_examples(test_transactions, tokenizer)
    examples_hash = examples_sha256(examples)
    expected_hashes = {
        baseline_reference.get("test_examples_sha256"),
        quantization_reference.get("test_examples_sha256"),
    }
    if expected_hashes != {examples_hash}:
        raise RuntimeError(
            "Recreated held-out test examples do not match both saved evaluation fingerprints."
        )

    previous_threads = torch.get_num_threads()
    selected_threads = (
        cpu_threads if cpu_threads is not None else baseline_reference.get(
            "benchmark_conditions", {},
        ).get("torch_cpu_threads", previous_threads)
    )
    if selected_threads < 1:
        raise ValueError("cpu_threads must be a positive integer.")
    torch.set_num_threads(selected_threads)
    try:
        loaded_models: dict[str, torch.nn.Module] = {
            "baseline": load_model(paths["baseline"], map_location="cpu"),
            "dynamic_int8": load_quantized_model(paths["dynamic_int8"]),
            "static_int8": load_quantized_model(paths["static_int8"]),
        }
        expected_vocab_size = tokenizer.vocab_size
        model_metrics: dict[str, dict[str, Any]] = {}
        sample_baskets = [
            [name for name in basket if name in tokenizer.product_to_id]
            for basket in (
                ["Milk", "Bread", "Butter"],
                ["Pasta", "Pasta Sauce"],
                ["Coffee", "Sugar"],
            )
        ]
        sample_baskets = [basket for basket in sample_baskets if basket]
        if not sample_baskets:
            sample_baskets = [tokenizer.products[:min(3, len(tokenizer.products))]]

        for name, model in loaded_models.items():
            if model.vocab_size != expected_vocab_size:
                raise ValueError(f"{name} vocabulary size does not match the saved tokenizer.")
            metrics = evaluate_model(
                model=model,
                transactions=test_transactions,
                tokenizer=tokenizer,
                model_path=paths[name],
                latency_samples=benchmark_samples,
                warmup_iterations=warmup_iterations,
                benchmark_repetitions=repetitions,
            )
            benchmark = metrics["inference_benchmark"]
            if benchmark["input_sha256"] != evaluate_model_input_hash(examples, benchmark_samples):
                raise RuntimeError(f"{name} benchmark did not use the fixed test-prefix sample.")
            model_metrics[name] = {
                **metrics,
                "checkpoint_path": str(paths[name]),
                "checkpoint_size_bytes": paths[name].stat().st_size,
                "parameter_count": model_parameter_count(model),
                "quantized_linear_count": _quantized_linear_count(model, name),
                "sample_recommendation_verification": _verify_recommendations(
                    model, tokenizer, sample_baskets,
                ),
            }
    finally:
        torch.set_num_threads(previous_threads)

    rows = _comparison_rows(model_metrics)
    analysis = _make_analysis(rows)
    report = {
        "methodology": {
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_hash,
            "vocabulary_path": str(tokenizer_path),
            "test_transaction_count": len(test_transactions),
            "test_example_count": len(examples),
            "test_examples_sha256": examples_hash,
            "same_test_examples_for_all_models": True,
            "transaction_split": "80/10/10 by transaction, seed 42, before prefix creation",
            "evaluation": "held-out next-item cross-entropy, Top-1 and Top-3 accuracy",
            "latency": (
                "CPU single-example forward pass on the same first N ordered held-out "
                "prefixes, with shared warm-up, repetition count, and CPU thread count"
            ),
            "device": "cpu",
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "torch_cpu_threads": selected_threads,
            "warmup_iterations": warmup_iterations,
            "repetitions": repetitions,
            "benchmark_samples_per_repetition": min(benchmark_samples, len(examples)),
            "input_batch_size": 1,
        },
        "models": model_metrics,
        "comparison": rows,
        "analysis": analysis,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "quantization_comparison.csv"
    json_path = results_dir / "quantization_comparison.json"
    chart_path = results_dir / "quantization_comparison.png"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _create_comparison_chart(rows, chart_path)
    print(pd.DataFrame(rows)[[
        "model", "model_size_mb", "top1_accuracy_percent", "top3_accuracy_percent",
        "test_loss", "average_cpu_inference_latency_ms", "size_reduction_percent",
        "top1_accuracy_change_percentage_points", "latency_improvement_percent",
    ]].to_string(index=False))
    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison JSON: {json_path}")
    print(f"Saved comparison chart: {chart_path}")
    print(json.dumps(analysis, indent=2))
    return report


def evaluate_model_input_hash(
    examples: list[tuple[list[int], int]], sample_count: int,
) -> str:
    """Hash the exact fixed examples used by the shared latency benchmark."""
    return examples_sha256(examples[:min(sample_count, len(examples))])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--benchmark-samples", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=None)
    args = parser.parse_args()
    run_quantization_comparison(
        warmup_iterations=args.warmup_iterations,
        repetitions=args.repetitions,
        benchmark_samples=args.benchmark_samples,
        cpu_threads=args.cpu_threads,
    )
