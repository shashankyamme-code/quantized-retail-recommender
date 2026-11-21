"""Transformer next-item prediction and frequent-itemset association rules."""

from __future__ import annotations

import itertools
from collections import Counter
from numbers import Integral
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.nn import functional as F

from src.dataset import (
    PRODUCT_CATEGORIES,
    TOKENIZER_PATH,
    TRANSACTIONS_PATH,
    ProductTokenizer,
    load_transaction_baskets,
)
from src.train import BASELINE_MODEL_PATH, load_model

Rule = dict[str, str | float | int | tuple[str, ...]]


def _normalize_basket(
    basket: Iterable[str | int],
    tokenizer: ProductTokenizer,
) -> tuple[list[str], list[int]]:
    if isinstance(basket, (str, bytes)):
        raise ValueError("Basket must be a sequence of product names or product IDs.")
    try:
        values = list(basket)
    except TypeError as error:
        raise ValueError("Basket must be a sequence of product names or product IDs.") from error
    if not values:
        raise ValueError("Please provide at least one product in the basket.")

    products: list[str] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"Invalid product ID: {value}")
        if isinstance(value, Integral):
            product_id = int(value)
            if product_id not in tokenizer.id_to_product:
                raise ValueError(f"Unknown product ID: {product_id}")
            products.append(tokenizer.id_to_product[product_id])
        elif isinstance(value, str):
            if value not in tokenizer.product_to_id:
                raise ValueError(f"Unknown product: {value}")
            products.append(value)
        else:
            raise ValueError(f"Invalid product value: {value!r}")

    unique_products = list(dict.fromkeys(products))
    return unique_products, tokenizer.encode(unique_products)


def recommend_next(
    basket: Iterable[str | int],
    model: torch.nn.Module | None = None,
    tokenizer: ProductTokenizer | None = None,
    top_k: int = 3,
    exclude_purchased: bool = True,
    model_path: str | Path = BASELINE_MODEL_PATH,
    tokenizer_path: str | Path = TOKENIZER_PATH,
) -> list[dict[str, str | float | int]]:
    """Return Transformer-ranked product predictions; integer inputs are product token IDs."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    tokenizer = tokenizer or ProductTokenizer.load(tokenizer_path)
    product_names, encoded = _normalize_basket(basket, tokenizer)
    model = model or load_model(model_path)
    model = model.cpu().eval()

    max_length = int(getattr(model, "max_length", 12))
    model_vocab_size = int(getattr(model, "vocab_size", tokenizer.vocab_size))
    # Older saved quantized models predate the BOS vocabulary entry.
    history = [tokenizer.bos_id, *encoded] if model_vocab_size == tokenizer.vocab_size else encoded
    context = history[-max_length:]
    tokens = torch.tensor([context], dtype=torch.long, device="cpu")
    lengths = torch.tensor([len(context)], dtype=torch.long, device="cpu")
    with torch.inference_mode():
        logits = model(tokens, lengths)[0].clone()
        if logits.numel() < len(tokenizer.products) + 1:
            raise ValueError("The model output vocabulary does not match the product vocabulary.")
        for token_id in tokenizer.special_token_ids.values():
            if token_id < logits.numel():
                logits[token_id] = torch.finfo(logits.dtype).min
        if exclude_purchased:
            for token_id in set(encoded):
                logits[token_id] = torch.finfo(logits.dtype).min
        product_logits = logits[:len(tokenizer.product_to_id) + 1]
        scores = F.softmax(product_logits, dim=-1)

    candidate_count = len(tokenizer.products)
    if exclude_purchased:
        candidate_count -= len(encoded)
    count = min(top_k, candidate_count)
    if count <= 0:
        return []
    probabilities, token_ids = torch.topk(scores, count)
    return [
        {
            "product": tokenizer.id_to_product[int(token_id)],
            "product_id": int(token_id),
            "score": float(probability),
            "category": PRODUCT_CATEGORIES.get(tokenizer.id_to_product[int(token_id)], "Other"),
            "source": "transformer",
        }
        for probability, token_id in zip(probabilities, token_ids)
    ]


def recommend_top_k(
    basket: Iterable[str | int],
    model: torch.nn.Module | None = None,
    tokenizer: ProductTokenizer | None = None,
    **options: object,
) -> dict[str, list[dict[str, str | float | int]]]:
    """Return separate Top-1, Top-3, and Top-5 prediction lists."""
    tokenizer = tokenizer or ProductTokenizer.load()
    model = model or load_model()
    products, _ = _normalize_basket(basket, tokenizer)
    return {
        f"top_{limit}": recommend_next(
            products, model=model, tokenizer=tokenizer, top_k=limit, **options,
        )
        for limit in (1, 3, 5)
    }


def mine_frequent_itemsets(
    transactions: list[list[str]] | None = None,
    min_support: float = 0.01,
    min_confidence: float = 0.1,
    max_itemset_size: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerate bounded frequent itemsets and derive support/confidence/lift rules.

    The product domain and default maximum set size are intentionally small, making
    transaction-wise combination counting transparent and practical without extra packages.
    """
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be in (0, 1].")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1].")
    if (
        isinstance(max_itemset_size, bool)
        or not isinstance(max_itemset_size, int)
        or max_itemset_size < 1
    ):
        raise ValueError("max_itemset_size must be a positive integer.")

    baskets = transactions if transactions is not None else load_transaction_baskets()
    if not baskets:
        raise ValueError("At least one transaction is required for itemset mining.")
    transaction_sets: list[frozenset[str]] = []
    for transaction in baskets:
        if isinstance(transaction, (str, bytes)):
            raise ValueError("Each transaction must be a collection of product names.")
        products = set(transaction)
        unknown = products.difference(PRODUCT_CATEGORIES)
        if unknown:
            raise ValueError(f"Unknown product(s) in transactions: {', '.join(sorted(unknown))}")
        transaction_sets.append(frozenset(products))

    itemset_counts: Counter[tuple[str, ...]] = Counter()
    limit = min(max_itemset_size, max(map(len, transaction_sets)))
    for transaction in transaction_sets:
        ordered_products = sorted(transaction)
        for size in range(1, min(limit, len(ordered_products)) + 1):
            itemset_counts.update(itertools.combinations(ordered_products, size))

    transaction_count = len(transaction_sets)
    minimum_count = min_support * transaction_count
    frequent_counts = {
        itemset: count for itemset, count in itemset_counts.items()
        if count >= minimum_count
    }
    support_by_set = {
        itemset: count / transaction_count for itemset, count in frequent_counts.items()
    }
    itemset_rows = [
        {
            "itemset": itemset,
            "itemset_size": len(itemset),
            "support": count / transaction_count,
            "transaction_count": count,
        }
        for itemset, count in sorted(
            frequent_counts.items(), key=lambda entry: (-entry[1], len(entry[0]), entry[0]),
        )
    ]

    rule_rows: list[dict[str, object]] = []
    for itemset, joint_count in frequent_counts.items():
        if len(itemset) < 2:
            continue
        joint_support = joint_count / transaction_count
        for antecedent_size in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, antecedent_size):
                consequent = tuple(product for product in itemset if product not in antecedent)
                antecedent_support = support_by_set.get(antecedent)
                consequent_support = support_by_set.get(consequent)
                if antecedent_support is None or consequent_support is None:
                    continue
                confidence = joint_support / antecedent_support
                if confidence < min_confidence:
                    continue
                rule_rows.append({
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "support": joint_support,
                    "confidence": confidence,
                    "lift": confidence / consequent_support,
                    "transaction_count": joint_count,
                })
    rule_rows.sort(
        key=lambda rule: (
            -float(rule["confidence"]),
            -float(rule["lift"]),
            -float(rule["support"]),
            rule["antecedent"],
            rule["consequent"],
        ),
    )
    return (
        pd.DataFrame(itemset_rows, columns=[
            "itemset", "itemset_size", "support", "transaction_count",
        ]),
        pd.DataFrame(rule_rows, columns=[
            "antecedent", "consequent", "support", "confidence", "lift", "transaction_count",
        ]),
    )


def probe_itemsets(
    basket: Iterable[str | int],
    transactions: list[list[str]] | None = None,
    top_k: int = 10,
    min_support: float = 0.01,
    min_confidence: float = 0.1,
    max_itemset_size: int = 3,
    rules: pd.DataFrame | None = None,
) -> list[Rule]:
    """Return association-rule recommendations, distinct from Transformer predictions."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    tokenizer = ProductTokenizer()
    products, _ = _normalize_basket(basket, tokenizer)
    if rules is None:
        _, rules = mine_frequent_itemsets(
            transactions=transactions,
            min_support=min_support,
            min_confidence=min_confidence,
            max_itemset_size=max_itemset_size,
        )
    rule_index: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for rule in rules.to_dict(orient="records"):
        rule_index.setdefault(tuple(rule["antecedent"]), []).append(rule)
    return _probe_rule_index(products, rule_index, top_k)


def _probe_rule_index(
    products: list[str],
    rule_index: dict[tuple[str, ...], list[dict[str, object]]],
    top_k: int,
) -> list[Rule]:
    selected = set(products)
    best_by_product: dict[str, Rule] = {}
    for antecedent_size in range(1, len(products) + 1):
        for antecedent in itertools.combinations(sorted(selected), antecedent_size):
            for rule in rule_index.get(antecedent, []):
                for product in rule["consequent"]:
                    if product in selected:
                        continue
                    recommendation: Rule = {
                        "product": product,
                        "category": PRODUCT_CATEGORIES[product],
                        "score": float(rule["confidence"]),
                        "support": float(rule["support"]),
                        "confidence": float(rule["confidence"]),
                        "lift": float(rule["lift"]),
                        "antecedent": antecedent,
                        "source": "association_rule",
                    }
                    current = best_by_product.get(product)
                    if current is None or (
                        recommendation["confidence"], recommendation["lift"], recommendation["support"]
                    ) > (current["confidence"], current["lift"], current["support"]):
                        best_by_product[product] = recommendation
    return sorted(
        best_by_product.values(),
        key=lambda item: (-float(item["confidence"]), -float(item["lift"]), -float(item["support"])),
    )[:top_k]
