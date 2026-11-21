"""Run the reproducible data, training, quantization, and comparison pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset import (
    TOKENIZER_PATH,
    TRANSACTIONS_PATH,
    ProductTokenizer,
    analyze_transactions,
    build_next_item_examples,
    generate_transactions,
    load_transaction_baskets,
    save_transactions,
    split_transactions,
)
from src.evaluate import build_comparison
from src.quantization import evaluate_quantized_models
from src.train import train_model


def prepare_data(transaction_count: int = 10_000) -> ProductTokenizer:
    frame = generate_transactions(transaction_count)
    save_transactions(frame)
    analyze_transactions(frame)
    print(f"Prepared {len(frame):,} synthetic transactions at {TRANSACTIONS_PATH}")
    return ProductTokenizer.load(TOKENIZER_PATH)


def run_pipeline(
    transaction_count: int = 10_000,
    epochs: int = 4,
    force_data: bool = False,
) -> None:
    if force_data or not Path(TRANSACTIONS_PATH).exists():
        tokenizer = prepare_data(transaction_count)
    else:
        tokenizer = ProductTokenizer.load(TOKENIZER_PATH)
    baskets = load_transaction_baskets()
    train_transactions, validation_transactions, test_transactions = split_transactions(baskets)
    model, _ = train_model(
        tokenizer=tokenizer,
        train_transactions=train_transactions,
        validation_transactions=validation_transactions,
        test_transactions=test_transactions,
        epochs=epochs,
    )
    train_examples = build_next_item_examples(
        train_transactions, tokenizer, max_length=model.max_length, shuffle_baskets=True,
    )
    results = evaluate_quantized_models(model, test_transactions, tokenizer, train_examples)
    comparison = build_comparison(results)
    print("\n" + comparison.to_string(index=False))
    print("Comparison saved to results/comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--force-data", action="store_true")
    arguments = parser.parse_args()
    run_pipeline(arguments.transactions, arguments.epochs, arguments.force_data)
