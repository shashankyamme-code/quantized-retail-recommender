"""Held-out evaluation of Transformer predictions against training-set itemset rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    TOKENIZER_PATH,
    ProductTokenizer,
    build_next_item_examples,
    load_transaction_baskets,
    split_transactions,
)
from src.evaluate import evaluate_model
from src.recommend import (
    _probe_rule_index,
    mine_frequent_itemsets,
    recommend_next,
)
from src.train import BASELINE_MODEL_PATH, NextItemDataset, collate_examples, load_model

RESULTS_DIR = Path("results")


def _jsonable_rules(frame) -> list[dict[str, Any]]:
    rules = []
    for row in frame.to_dict(orient="records"):
        rules.append({
            "antecedent": list(row["antecedent"]),
            "consequent": list(row["consequent"]),
            "support": float(row["support"]),
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
            "transaction_count": int(row["transaction_count"]),
        })
    return rules


def evaluate_recommendations(
    train_transactions: list[list[str]],
    test_transactions: list[list[str]],
    model: torch.nn.Module,
    tokenizer: ProductTokenizer,
    min_support: float = 0.005,
    min_confidence: float = 0.2,
    max_itemset_size: int = 3,
    example_count: int = 5,
    results_dir: str | Path = RESULTS_DIR,
    model_path: str | Path | None = BASELINE_MODEL_PATH,
) -> dict[str, Any]:
    """Evaluate the model and rules on held-out transactions; mine rules on train only."""
    if not train_transactions or not test_transactions:
        raise ValueError("Both training and held-out test transactions are required.")
    if example_count < 1:
        raise ValueError("example_count must be positive.")

    # Association rules are fitted on training baskets; all reported prediction metrics use test baskets.
    frequent_itemsets, rules = mine_frequent_itemsets(
        train_transactions,
        min_support=min_support,
        min_confidence=min_confidence,
        max_itemset_size=max_itemset_size,
    )
    serialized_rules = rules.to_dict(orient="records")
    rule_index: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for rule in serialized_rules:
        rule_index.setdefault(tuple(rule["antecedent"]), []).append(rule)

    transformer_metrics = evaluate_model(model.cpu().eval(), test_transactions, tokenizer, model_path)
    test_examples = build_next_item_examples(test_transactions, tokenizer)
    comparable_examples = [
        (context, target) for context, target in test_examples
        if any(token_id in tokenizer.id_to_product for token_id in context)
    ]
    if not comparable_examples:
        raise ValueError("Held-out transactions need at least one product prefix for comparison.")

    loader = DataLoader(
        NextItemDataset(comparable_examples),
        batch_size=256,
        shuffle=False,
        collate_fn=collate_examples,
    )
    transformer_top1 = transformer_top3 = rule_top1 = rule_top3 = 0
    rule_covered = 0
    model.eval()
    with torch.inference_mode():
        for tokens, lengths, targets in loader:
            logits = model(tokens, lengths)
            for token_id in tokenizer.special_token_ids.values():
                if token_id < logits.shape[-1]:
                    logits[:, token_id] = torch.finfo(logits.dtype).min
            product_logits = logits[:, :len(tokenizer.products) + 1]
            model_candidates = product_logits.topk(min(5, len(tokenizer.products)), dim=-1).indices
            transformer_top1 += (model_candidates[:, 0] == targets).sum().item()
            transformer_top3 += (model_candidates[:, :3] == targets[:, None]).any(dim=1).sum().item()

            for context, target in zip(tokens.tolist(), targets.tolist()):
                prefix = tokenizer.decode(context)
                association_predictions = _probe_rule_index(prefix, rule_index, top_k=5)
                if association_predictions:
                    rule_covered += 1
                recommended_ids = [
                    tokenizer.product_to_id[str(result["product"])]
                    for result in association_predictions
                ]
                rule_top1 += target in recommended_ids[:1]
                rule_top3 += target in recommended_ids[:3]

    comparison_count = len(comparable_examples)
    examples: list[dict[str, Any]] = []
    for basket in test_transactions[:example_count]:
        if len(basket) < 2:
            continue
        prefix = basket[:1]
        actual_next = basket[1]
        examples.append({
            "basket_prefix": prefix,
            "actual_next_product": actual_next,
            "transformer_predictions": recommend_next(
                prefix, model=model, tokenizer=tokenizer, top_k=5,
            ),
            "association_rule_recommendations": _probe_rule_index(
                prefix, rule_index, top_k=5,
            ),
        })

    results = {
        "evaluation_method": "Transformer and association rules evaluated on held-out test transactions",
        "association_rules_fit_on": "training_transactions_only",
        "test_transactions": len(test_transactions),
        "transformer": transformer_metrics,
        "comparison": {
            "test_examples_with_nonempty_product_prefix": comparison_count,
            "transformer_top1_accuracy": transformer_top1 / comparison_count,
            "transformer_top3_accuracy": transformer_top3 / comparison_count,
            "association_rule_top1_accuracy": rule_top1 / comparison_count,
            "association_rule_top3_accuracy": rule_top3 / comparison_count,
            "association_rule_coverage": rule_covered / comparison_count,
            "association_rule_recommendation_source": "association_rule",
            "transformer_recommendation_source": "transformer",
        },
        "itemset_parameters": {
            "min_support": min_support,
            "min_confidence": min_confidence,
            "max_itemset_size": max_itemset_size,
        },
        "examples": examples,
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "recommendation_evaluation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )
    itemsets_to_save = frequent_itemsets.copy()
    itemsets_to_save["itemset"] = itemsets_to_save["itemset"].map(list).map(json.dumps)
    itemsets_to_save.to_csv(results_dir / "frequent_itemsets.csv", index=False)
    rules_to_save = rules.copy()
    for column in ("antecedent", "consequent"):
        rules_to_save[column] = rules_to_save[column].map(list).map(json.dumps)
    rules_to_save.to_csv(results_dir / "association_rules.csv", index=False)
    return results


def run_evaluation() -> dict[str, Any]:
    tokenizer = ProductTokenizer.load(TOKENIZER_PATH)
    train_transactions, _, test_transactions = split_transactions(load_transaction_baskets())
    model = load_model(BASELINE_MODEL_PATH, map_location="cpu")
    results = evaluate_recommendations(
        train_transactions=train_transactions,
        test_transactions=test_transactions,
        model=model,
        tokenizer=tokenizer,
    )
    print(json.dumps(results["comparison"], indent=2))
    print(f"Saved held-out recommendation results to {RESULTS_DIR / 'recommendation_evaluation.json'}")
    print(f"Saved training-set itemsets and rules to {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_evaluation()
