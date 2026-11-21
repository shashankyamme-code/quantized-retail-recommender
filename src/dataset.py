"""Synthetic retail basket generation, tokenization, and analysis."""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SEED = 42
DATA_DIR = Path("data")
TRANSACTIONS_PATH = DATA_DIR / "retail_transactions.csv"
TOKENIZER_PATH = DATA_DIR / "product_vocabulary.json"

PRODUCT_CATEGORIES = {
    "Milk": "Dairy", "Cheese": "Dairy", "Yogurt": "Dairy", "Butter": "Dairy",
    "Cream": "Dairy", "Eggs": "Dairy", "Bread": "Bakery", "Bagels": "Bakery",
    "Croissants": "Bakery", "Muffins": "Bakery", "Coffee": "Beverages",
    "Tea": "Beverages", "Soft Drinks": "Beverages", "Orange Juice": "Beverages",
    "Bottled Water": "Beverages", "Cereal": "Breakfast", "Oatmeal": "Breakfast",
    "Sugar": "Baking", "Flour": "Baking", "Pasta": "Pantry",
    "Pasta Sauce": "Pantry", "Rice": "Pantry", "Canned Tomatoes": "Pantry",
    "Chips": "Snacks", "Chocolate": "Snacks", "Cookies": "Snacks",
    "Apples": "Fruits", "Bananas": "Fruits", "Oranges": "Fruits",
    "Strawberries": "Fruits", "Tomatoes": "Vegetables", "Lettuce": "Vegetables",
    "Carrots": "Vegetables", "Onions": "Vegetables", "Shampoo": "Personal care",
    "Toothpaste": "Personal care", "Soap": "Personal care",
    "Laundry Detergent": "Household", "Paper Towels": "Household",
    "Dish Soap": "Household",
}

ASSOCIATION_TEMPLATES = (
    ("Milk", "Cereal", "Bananas"),
    ("Bread", "Butter", "Eggs"),
    ("Chips", "Soft Drinks"),
    ("Pasta", "Pasta Sauce", "Cheese"),
    ("Coffee", "Sugar", "Cream"),
    ("Tea", "Honey", "Cookies"),
    ("Yogurt", "Strawberries", "Bananas"),
    ("Rice", "Canned Tomatoes", "Onions"),
    ("Flour", "Sugar", "Eggs", "Butter"),
    ("Shampoo", "Soap", "Toothpaste"),
    ("Laundry Detergent", "Paper Towels", "Dish Soap"),
    ("Bagels", "Cream Cheese", "Coffee"),
    ("Oatmeal", "Milk", "Apples"),
    ("Lettuce", "Tomatoes", "Carrots"),
    ("Chocolate", "Cookies", "Milk"),
    ("Orange Juice", "Croissants", "Yogurt"),
)

# Products used in associations but not otherwise listed above.
PRODUCT_CATEGORIES.update({
    "Honey": "Pantry", "Cream Cheese": "Dairy",
})


class ProductTokenizer:
    """Stable product vocabulary with PAD, BOS, EOS, and UNK special tokens."""

    def __init__(self, products: Iterable[str] | None = None):
        names = sorted(products if products is not None else PRODUCT_CATEGORIES)
        if not names or len(set(names)) != len(names):
            raise ValueError("The product vocabulary must contain unique product names.")
        # Product IDs start at one and remain stable; special token IDs follow them.
        self.product_to_id = {name: index + 1 for index, name in enumerate(names)}
        self.id_to_product = {index: name for name, index in self.product_to_id.items()}
        self.pad_id = 0
        self.bos_id = len(names) + 1
        self.eos_id = len(names) + 2
        self.unk_id = len(names) + 3
        self.special_token_ids = {
            "<pad>": self.pad_id,
            "<bos>": self.bos_id,
            "<eos>": self.eos_id,
            "<unk>": self.unk_id,
        }

    @property
    def vocab_size(self) -> int:
        return len(self.product_to_id) + len(self.special_token_ids)

    @property
    def products(self) -> list[str]:
        return list(self.product_to_id)

    def encode(self, products: Iterable[str], add_special_tokens: bool = False) -> list[int]:
        try:
            encoded = [self.product_to_id[name] for name in products]
        except KeyError as error:
            raise ValueError(f"Unknown product: {error.args[0]}") from error
        return [self.bos_id, *encoded, self.eos_id] if add_special_tokens else encoded

    def encode_sequence(self, products: Iterable[str]) -> list[int]:
        """Encode a complete basket with explicit beginning and ending markers."""
        return self.encode(products, add_special_tokens=True)

    def decode(self, token_ids: Iterable[int]) -> list[str]:
        return [self.id_to_product[token_id] for token_id in token_ids if token_id in self.id_to_product]

    def save(self, path: str | Path = TOKENIZER_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "products": self.product_to_id,
            "special_tokens": self.special_token_ids,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path = TOKENIZER_PATH) -> ProductTokenizer:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        # Accept the original flat product mapping so existing project data still loads.
        mapping = saved.get("products", saved)
        tokenizer = cls(mapping.keys())
        tokenizer.product_to_id = {str(name): int(token_id) for name, token_id in mapping.items()}
        tokenizer.id_to_product = {token_id: name for name, token_id in tokenizer.product_to_id.items()}
        if sorted(tokenizer.id_to_product) != list(range(1, len(mapping) + 1)):
            raise ValueError("Vocabulary token IDs must be contiguous and start at one.")
        tokenizer.bos_id = len(mapping) + 1
        tokenizer.eos_id = len(mapping) + 2
        tokenizer.unk_id = len(mapping) + 3
        tokenizer.special_token_ids = {
            "<pad>": tokenizer.pad_id,
            "<bos>": tokenizer.bos_id,
            "<eos>": tokenizer.eos_id,
            "<unk>": tokenizer.unk_id,
        }
        saved_special_tokens = saved.get("special_tokens")
        if saved_special_tokens is not None and saved_special_tokens != tokenizer.special_token_ids:
            raise ValueError("Saved special-token IDs do not match the product vocabulary.")
        return tokenizer


def generate_transactions(n_transactions: int = 10_000, seed: int = SEED) -> pd.DataFrame:
    """Generate realistic, varied baskets with intentionally repeated product signals."""
    if n_transactions < 1:
        raise ValueError("n_transactions must be a positive integer.")
    rng = random.Random(seed)
    products = list(PRODUCT_CATEGORIES)
    transactions: list[dict[str, object]] = []
    for transaction_id in range(1, n_transactions + 1):
        basket: set[str] = set()
        template_count = 1 if rng.random() < 0.78 else 2
        for template in rng.sample(ASSOCIATION_TEMPLATES, template_count):
            if rng.random() < 0.88:
                basket.update(template)
            else:
                basket.update(rng.sample(template, k=max(2, len(template) - 1)))

        target_size = rng.randint(2, 9)
        while len(basket) < target_size:
            basket.add(rng.choice(products))
        basket_list = sorted(basket)
        rng.shuffle(basket_list)
        transactions.append({
            "transaction_id": transaction_id,
            "products": json.dumps(basket_list, ensure_ascii=False),
            "item_count": len(basket_list),
        })
    return pd.DataFrame(transactions)


def clean_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize basket values and remove blank, unknown, and duplicate products."""
    if "products" not in frame.columns:
        raise ValueError("Input data must contain a 'products' column.")
    cleaned: list[dict[str, object]] = []
    for row_index, value in enumerate(frame["products"]):
        if isinstance(value, str):
            try:
                items = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid product list in row {row_index}.") from error
        else:
            items = value
        if not isinstance(items, (list, tuple)):
            raise ValueError(f"Products in row {row_index} must be a list.")
        unique_items = list(dict.fromkeys(
            item.strip() for item in items
            if isinstance(item, str) and item.strip() in PRODUCT_CATEGORIES
        ))
        if len(unique_items) >= 2:
            cleaned.append({
                "transaction_id": int(frame.iloc[row_index].get("transaction_id", row_index + 1)),
                "products": json.dumps(unique_items, ensure_ascii=False),
                "item_count": len(unique_items),
            })
    result = pd.DataFrame(cleaned, columns=["transaction_id", "products", "item_count"])
    if result.empty:
        raise ValueError("No valid transactions remain after cleaning.")
    return result.reset_index(drop=True)


def save_transactions(
    frame: pd.DataFrame,
    path: str | Path = TRANSACTIONS_PATH,
    tokenizer_path: str | Path = TOKENIZER_PATH,
) -> tuple[Path, Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_transactions(frame).to_csv(path, index=False)
    ProductTokenizer().save(tokenizer_path)
    return path, Path(tokenizer_path)


def load_transaction_baskets(path: str | Path = TRANSACTIONS_PATH) -> list[list[str]]:
    frame = pd.read_csv(path)
    return [json.loads(value) for value in clean_transactions(frame)["products"]]


def split_transactions(
    transactions: list[list[str]], seed: int = SEED,
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    if len(transactions) < 3:
        raise ValueError("At least three transactions are needed for train/validation/test splits.")
    indices = list(range(len(transactions)))
    random.Random(seed).shuffle(indices)
    train_end = int(len(indices) * 0.8)
    validation_end = int(len(indices) * 0.9)
    train = [transactions[index] for index in indices[:train_end]]
    validation = [transactions[index] for index in indices[train_end:validation_end]]
    test = [transactions[index] for index in indices[validation_end:]]
    return train, validation, test


def build_next_item_examples(
    transactions: list[list[str]],
    tokenizer: ProductTokenizer,
    max_length: int = 12,
    shuffle_baskets: bool = False,
    seed: int = SEED,
) -> list[tuple[list[int], int]]:
    """Create BOS-prefixed contexts and next-product targets within each transaction."""
    if max_length < 1:
        raise ValueError("max_length must be positive.")
    rng = random.Random(seed)
    examples: list[tuple[list[int], int]] = []
    for basket in transactions:
        items = list(dict.fromkeys(basket))
        if shuffle_baskets:
            rng.shuffle(items)
        encoded = tokenizer.encode(items)
        for next_index, target in enumerate(encoded):
            # BOS marks each independent basket; contexts never cross transaction boundaries.
            history = [tokenizer.bos_id, *encoded[:next_index]]
            examples.append((history[-max_length:], target))
    return examples


def analyze_transactions(
    frame: pd.DataFrame,
    plots_dir: str | Path = Path("results") / "performance_plots",
    top_n: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baskets = [json.loads(value) for value in clean_transactions(frame)["products"]]
    product_counts = Counter(product for basket in baskets for product in basket)
    pair_counts: Counter[tuple[str, str]] = Counter()
    for basket in baskets:
        pair_counts.update(itertools.combinations(sorted(set(basket)), 2))
    frequency = pd.DataFrame(
        [(name, PRODUCT_CATEGORIES[name], count) for name, count in product_counts.most_common()],
        columns=["product", "category", "transaction_count"],
    )
    combinations = pd.DataFrame(
        [(left, right, count) for (left, right), count in pair_counts.most_common()],
        columns=["product_1", "product_2", "transaction_count"],
    )
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    top_products = frequency.head(top_n).sort_values("transaction_count")
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top_products["product"], top_products["transaction_count"], color="#2a9d8f")
    axis.set(title="Most frequently purchased products", xlabel="Transactions", ylabel="Product")
    figure.tight_layout()
    figure.savefig(plots_dir / "product_frequency.png", dpi=150)
    plt.close(figure)

    top_pairs = combinations.head(top_n).copy()
    if not top_pairs.empty:
        top_pairs["pair"] = top_pairs["product_1"] + " + " + top_pairs["product_2"]
        top_pairs = top_pairs.sort_values("transaction_count")
    figure, axis = plt.subplots(figsize=(10, 6))
    if top_pairs.empty:
        axis.text(0.5, 0.5, "No product combinations", ha="center", va="center")
        axis.set_axis_off()
    else:
        axis.barh(top_pairs["pair"], top_pairs["transaction_count"], color="#e9c46a")
        axis.set(title="Most common product combinations", xlabel="Transactions", ylabel="Product pair")
    figure.tight_layout()
    figure.savefig(plots_dir / "common_combinations.png", dpi=150)
    plt.close(figure)
    return frequency, combinations
