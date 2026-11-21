"""Measure the unquantized baseline model on its held-out test transactions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

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
from src.evaluate import (
    RESULTS_DIR,
    benchmark_cpu_inference,
    evaluate_model,
    examples_sha256,
    model_file_size_mb,
    model_memory_size_bytes,
    model_parameter_count,
)
from src.train import BASELINE_MODEL_PATH, load_model

BASELINE_METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"
BASELINE_TABLE_PATH = RESULTS_DIR / "baseline_comparison.csv"
SPLIT_SEED = 42


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_sha256(path: Path) -> str:
    return _sha256_file(path)


def run_baseline_evaluation(
    model_path: str | Path = BASELINE_MODEL_PATH,
    tokenizer_path: str | Path = TOKENIZER_PATH,
    dataset_path: str | Path = TRANSACTIONS_PATH,
    results_path: str | Path = BASELINE_METRICS_PATH,
    table_path: str | Path = BASELINE_TABLE_PATH,
    warmup_iterations: int = 20,
    repetitions: int = 5,
    benchmark_samples: int = 100,
) -> dict[str, Any]:
    """Evaluate the saved baseline without modifying or retraining its weights."""
    model_path = Path(model_path)
    tokenizer_path = Path(tokenizer_path)
    dataset_path = Path(dataset_path)
    results_path = Path(results_path)
    table_path = Path(table_path)

    tokenizer = ProductTokenizer.load(tokenizer_path)
    baskets = load_transaction_baskets(dataset_path)
    train_transactions, validation_transactions, test_transactions = split_transactions(
        baskets, seed=SPLIT_SEED,
    )
    model = load_model(model_path, map_location="cpu")
    saved_test_examples = build_next_item_examples(test_transactions, tokenizer, model.max_length)
    metrics = evaluate_model(
        model=model,
        transactions=test_transactions,
        tokenizer=tokenizer,
        model_path=model_path,
        latency_samples=benchmark_samples,
        warmup_iterations=warmup_iterations,
        benchmark_repetitions=repetitions,
    )
    checkpoint_bytes = model_path.stat().st_size
    in_memory_bytes = model_memory_size_bytes(model)

    results: dict[str, Any] = {
        "model": "unquantized_baseline",
        "checkpoint_path": str(model_path),
        "checkpoint_sha256": _sha256_file(model_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _dataset_sha256(dataset_path),
        "vocabulary_path": str(tokenizer_path),
        "vocabulary_size": tokenizer.vocab_size,
        "parameter_count": model_parameter_count(model),
        "checkpoint_size_bytes": checkpoint_bytes,
        "checkpoint_size_mb": model_file_size_mb(model_path),
        "checkpoint_size_mib": checkpoint_bytes / (1024 * 1024),
        "in_memory_parameter_and_buffer_size_bytes": in_memory_bytes,
        "in_memory_parameter_and_buffer_size_mb": in_memory_bytes / 1_000_000,
        "in_memory_parameter_and_buffer_size_mib": in_memory_bytes / (1024 * 1024),
        "transaction_split_sizes": {
            "train": len(train_transactions),
            "validation": len(validation_transactions),
            "test": len(test_transactions),
        },
        "test_examples": metrics["test_examples"],
        "test_examples_sha256": examples_sha256(saved_test_examples),
        "test_loss": metrics["test_loss"],
        "top1_accuracy": metrics["top1_accuracy"],
        "top3_accuracy": metrics["top3_accuracy"],
        "cpu_inference": metrics["inference_benchmark"],
        "benchmark_conditions": {
            "device": "cpu",
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "torch_cpu_threads": torch.get_num_threads(),
            "warmup_iterations": warmup_iterations,
            "repetitions": repetitions,
            "fixed_test_prefixes_per_repetition": min(benchmark_samples, len(saved_test_examples)),
            "input_batch_size": 1,
        },
        "methodology": (
            "Whole transactions are deterministically split 80/10/10 with seed 42 before "
            "prefix examples are constructed. Top-k accuracy and cross-entropy are computed "
            "over the held-out test examples. CPU latency uses fixed, ordered test prefixes, "
            "single-example forward calls, warm-up iterations, and repeated full passes."
        ),
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    # Keep the older evaluation filename in sync for existing dashboard/report consumers.
    if results_path.name == BASELINE_METRICS_PATH.name:
        (results_path.parent / "baseline_evaluation.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8",
        )

    row = {
        "model": results["model"],
        "checkpoint_size_mb": results["checkpoint_size_mb"],
        "in_memory_parameter_and_buffer_size_mb": results["in_memory_parameter_and_buffer_size_mb"],
        "parameter_count": results["parameter_count"],
        "top1_accuracy": results["top1_accuracy"],
        "top3_accuracy": results["top3_accuracy"],
        "test_loss": results["test_loss"],
        "average_cpu_inference_latency_ms": results["cpu_inference"]["average_latency_ms"],
        "median_cpu_inference_latency_ms": results["cpu_inference"]["median_latency_ms"],
        "p95_cpu_inference_latency_ms": results["cpu_inference"]["p95_latency_ms"],
        "test_examples": results["test_examples"],
        "test_examples_sha256": results["test_examples_sha256"],
        "device": "cpu",
        "warmup_iterations": warmup_iterations,
        "repetitions": repetitions,
        "fixed_test_prefixes_per_repetition": results["benchmark_conditions"][
            "fixed_test_prefixes_per_repetition"
        ],
    }
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(table_path, index=False)
    print(json.dumps(row, indent=2))
    print(f"Saved baseline metrics to {results_path}")
    print(f"Saved baseline comparison row to {table_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--benchmark-samples", type=int, default=100)
    args = parser.parse_args()
    run_baseline_evaluation(
        warmup_iterations=args.warmup_iterations,
        repetitions=args.repetitions,
        benchmark_samples=args.benchmark_samples,
    )
