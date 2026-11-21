"""Train the baseline product language model from scratch."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import (
    SEED,
    TOKENIZER_PATH,
    TRANSACTIONS_PATH,
    ProductTokenizer,
    build_next_item_examples,
    load_transaction_baskets,
    split_transactions,
)
from src.model import RetailLanguageModel, count_parameters

MODEL_DIR = Path("models")
BASELINE_MODEL_PATH = MODEL_DIR / "baseline_model.pth"
TRAINING_HISTORY_PATH = Path("results") / "training_history.json"
TRAINING_METRICS_PATH = Path("results") / "training_metrics.json"
TRAINING_LOSS_PLOT_PATH = Path("results") / "performance_plots" / "training_validation_loss.png"
MODEL_CONFIG = {
    "max_length": 12,
    "d_model": 48,
    "n_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
}


class NextItemDataset(Dataset):
    def __init__(self, examples: list[tuple[list[int], int]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.examples[index]


def collate_examples(batch: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(context) for context, _ in batch], dtype=torch.long)
    width = max(int(lengths.max()), 1)
    tokens = torch.zeros((len(batch), width), dtype=torch.long)
    targets = torch.tensor([target for _, target in batch], dtype=torch.long)
    for index, (context, _) in enumerate(batch):
        tokens[index, :len(context)] = torch.tensor(context, dtype=torch.long)
    return tokens, lengths, targets


def set_seed(seed: int = SEED) -> None:
    # Seed Python, NumPy, CPU, and CUDA RNGs so data order and model initialization repeat.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _epoch(
    model: RetailLanguageModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_items = 0
    for tokens, lengths, targets in loader:
        tokens, lengths, targets = tokens.to(device), lengths.to(device), targets.to(device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            logits = model(tokens, lengths)
            loss = criterion(logits, targets)
            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss += loss.item() * len(targets)
        total_items += len(targets)
    if total_items == 0:
        raise ValueError("Cannot train or evaluate on an empty set of examples.")
    return total_loss / total_items


def train_model(
    tokenizer: ProductTokenizer | None = None,
    train_transactions: list[list[str]] | None = None,
    validation_transactions: list[list[str]] | None = None,
    test_transactions: list[list[str]] | None = None,
    epochs: int = 4,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    seed: int = SEED,
    model_path: str | Path = BASELINE_MODEL_PATH,
    vocabulary_path: str | Path = TOKENIZER_PATH,
    metrics_path: str | Path = TRAINING_METRICS_PATH,
    plot_path: str | Path = TRAINING_LOSS_PLOT_PATH,
) -> tuple[RetailLanguageModel, dict[str, list[float]]]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    set_seed(seed)
    if tokenizer is None:
        tokenizer = ProductTokenizer.load(vocabulary_path)
    if train_transactions is None:
        train_transactions, default_validation, default_test = split_transactions(load_transaction_baskets())
        if validation_transactions is None:
            validation_transactions = default_validation
        if test_transactions is None:
            test_transactions = default_test
    if validation_transactions is None:
        raise ValueError("Provide validation transactions separate from the training split.")
    if not train_transactions or not validation_transactions:
        raise ValueError("Training and validation splits must both contain transactions.")

    # Split complete transactions before generating prefix examples to prevent leakage.
    train_examples = build_next_item_examples(
        train_transactions, tokenizer, MODEL_CONFIG["max_length"], shuffle_baskets=True, seed=seed,
    )
    validation_examples = build_next_item_examples(
        validation_transactions, tokenizer, MODEL_CONFIG["max_length"], shuffle_baskets=False, seed=seed,
    )
    if not train_examples or not validation_examples:
        raise ValueError("Training and validation splits must each produce next-item examples.")
    train_loader = DataLoader(
        NextItemDataset(train_examples), batch_size=batch_size, shuffle=True, collate_fn=collate_examples,
    )
    validation_loader = DataLoader(
        NextItemDataset(validation_examples), batch_size=batch_size, shuffle=False, collate_fn=collate_examples,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetailLanguageModel(tokenizer.vocab_size, **MODEL_CONFIG).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: dict[str, list[float]] = {"train_loss": [], "validation_loss": []}
    epoch_seconds: list[float] = []
    for epoch_index in range(epochs):
        started = time.perf_counter()
        train_loss = _epoch(model, train_loader, criterion, device, optimizer)
        validation_loss = _epoch(model, validation_loader, criterion, device)
        epoch_seconds.append(time.perf_counter() - started)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        print(
            f"Epoch {epoch_index + 1}/{epochs} "
            f"train_loss={train_loss:.4f} validation_loss={validation_loss:.4f} "
            f"duration={epoch_seconds[-1]:.1f}s"
        )

    model = model.to("cpu").eval()
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the full-module file for existing inference/quantization code and also save portable weights.
    torch.save(model, model_path)
    weights_path = model_path.with_name("baseline_model_state_dict.pth")
    torch.save(model.state_dict(), weights_path)
    tokenizer.save(vocabulary_path)

    config_path = model_path.with_name("baseline_model_config.json")
    config = {
        **MODEL_CONFIG,
        "vocab_size": tokenizer.vocab_size,
        "special_token_ids": tokenizer.special_token_ids,
        "parameter_count": count_parameters(model),
        "training_method": "trained from scratch",
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    metrics = {
        "training_method": "trained from scratch",
        "device": str(device),
        "parameter_count": count_parameters(model),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "transaction_split_sizes": {
            "train": len(train_transactions),
            "validation": len(validation_transactions),
            "test": len(test_transactions) if test_transactions is not None else None,
        },
        "next_item_example_counts": {
            "train": len(train_examples),
            "validation": len(validation_examples),
        },
        "epoch_seconds": epoch_seconds,
        "model_config_path": str(config_path),
        "model_weights_path": str(weights_path),
        **history,
    }
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Keep the previous dashboard history path synchronized for the normal training run.
    if metrics_path == TRAINING_METRICS_PATH:
        TRAINING_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        TRAINING_HISTORY_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    epochs_axis = range(1, epochs + 1)
    axis.plot(epochs_axis, history["train_loss"], marker="o", label="Training")
    axis.plot(epochs_axis, history["validation_loss"], marker="o", label="Validation")
    axis.set(title="Training and validation loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    print(f"Saved baseline model ({count_parameters(model):,} parameters) to {model_path}")
    print(f"Saved model weights to {weights_path}")
    print(f"Saved model configuration to {config_path}")
    print(f"Saved training metrics to {metrics_path}")
    print(f"Saved loss plot to {plot_path}")
    return model, history


def load_model(path: str | Path = BASELINE_MODEL_PATH, map_location: str = "cpu") -> RetailLanguageModel:
    model = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(model, RetailLanguageModel):
        raise TypeError(f"Expected a RetailLanguageModel checkpoint at {path}.")
    return model.eval()


def load_model_weights(
    weights_path: str | Path,
    config_path: str | Path,
    map_location: str = "cpu",
) -> RetailLanguageModel:
    """Rebuild the model from its JSON configuration and a weights-only checkpoint."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    architecture = {name: config[name] for name in MODEL_CONFIG}
    model = RetailLanguageModel(vocab_size=config["vocab_size"], **architecture)
    state_dict = torch.load(weights_path, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict)
    return model.eval()


if __name__ == "__main__":
    train_model()
