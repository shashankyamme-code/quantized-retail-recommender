"""Focused tests for data preparation, recommendation, metrics, and quantization."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.dataset import (
    ProductTokenizer,
    build_next_item_examples,
    clean_transactions,
    generate_transactions,
    split_transactions,
)
from src.evaluate import benchmark_cpu_inference, build_comparison, evaluate_model, model_memory_size_bytes
from src.evaluate_baseline import run_baseline_evaluation
from src.evaluate_quantization import _comparison_rows, _make_analysis
from src.evaluate_recommendations import evaluate_recommendations
from src.model import CausalSelfAttention, RetailLanguageModel
from src.quantization import load_quantized_model, quantize_dynamic, quantize_static
from src.quantize_models import quantize_saved_models
from src.recommend import mine_frequent_itemsets, probe_itemsets, recommend_next, recommend_top_k
from src.train import load_model, load_model_weights


def test_synthetic_dataset_generation_and_cleaning():
    generated = generate_transactions(40, seed=7)
    assert generated.equals(generate_transactions(40, seed=7))
    assert len(generated) == 40
    assert generated["item_count"].min() >= 2
    cleaned = clean_transactions(generated)
    assert len(cleaned) == 40
    duplicate_basket = clean_transactions(
        pd.DataFrame({"products": [json.dumps(["Milk", "Milk", "Bread"])]})
    )
    assert json.loads(duplicate_basket.iloc[0]["products"]) == ["Milk", "Bread"]


def test_tokenizer_and_deterministic_transaction_splits(tmp_path):
    tokenizer = ProductTokenizer(["Milk", "Bread", "Butter"])
    encoded = tokenizer.encode(["Milk", "Bread"])
    assert tokenizer.decode(encoded) == ["Milk", "Bread"]
    assert tokenizer.vocab_size == 7
    complete_sequence = tokenizer.encode_sequence(["Milk", "Bread"])
    assert complete_sequence == [tokenizer.bos_id, *encoded, tokenizer.eos_id]
    assert tokenizer.decode(complete_sequence) == ["Milk", "Bread"]
    vocabulary_path = tmp_path / "product_vocabulary.json"
    tokenizer.save(vocabulary_path)
    loaded = ProductTokenizer.load(vocabulary_path)
    assert loaded.product_to_id == tokenizer.product_to_id
    assert loaded.special_token_ids == tokenizer.special_token_ids
    with pytest.raises(ValueError, match="Unknown product"):
        tokenizer.encode(["Not a product"])
    baskets = [["Milk", "Bread", "Butter"]] * 10
    train, validation, test = split_transactions(baskets, seed=8)
    assert (len(train), len(validation), len(test)) == (8, 1, 1)
    examples = build_next_item_examples(train[:1], tokenizer, shuffle_baskets=False)
    assert len(examples) == 3
    assert examples[0] == ([tokenizer.bos_id], encoded[0])
    assert examples[1] == ([tokenizer.bos_id, encoded[0]], encoded[1])


def test_transaction_split_keeps_entire_baskets_in_one_partition():
    products = list(ProductTokenizer().products)[:20]
    transactions = [products[index:index + 2] for index in range(0, 20, 2)]
    train, validation, test = split_transactions(transactions, seed=10)
    product_sets = [set(product for basket in part for product in basket) for part in (train, validation, test)]
    assert product_sets[0].isdisjoint(product_sets[1])
    assert product_sets[0].isdisjoint(product_sets[2])
    assert product_sets[1].isdisjoint(product_sets[2])


def _small_model() -> RetailLanguageModel:
    torch.manual_seed(3)
    return RetailLanguageModel(vocab_size=10, max_length=4, d_model=16, n_heads=4, num_layers=1, dropout=0)


def _calibration_examples():
    return [([1], 2), ([2, 1], 3), ([3, 2, 1], 4), ([4, 3], 5)]


def test_model_recommendations_and_itemset_probing():
    tokenizer = ProductTokenizer(["Milk", "Bread", "Butter", "Cereal", "Coffee", "Sugar"])
    model = _small_model()
    results = recommend_next(["Milk"], model=model, tokenizer=tokenizer, top_k=3)
    assert len(results) == 3
    assert all(result["product"] != "Milk" for result in results)
    assert all(0 <= result["score"] <= 1 for result in results)
    basket_transactions = [["Milk", "Bread"], ["Milk", "Butter"], ["Milk", "Bread"]]
    probed = probe_itemsets(["Milk"], basket_transactions)
    assert probed[0]["product"] == "Bread"
    with pytest.raises(ValueError, match="at least one"):
        recommend_next([], model=model, tokenizer=tokenizer)


def test_recommendations_accept_product_ids_and_return_top_1_3_5():
    tokenizer = ProductTokenizer(["Milk", "Bread", "Butter", "Cereal", "Coffee", "Sugar"])
    model = _small_model()
    predictions = recommend_top_k(
        [tokenizer.product_to_id["Milk"]],
        model=model,
        tokenizer=tokenizer,
    )
    assert [len(predictions[f"top_{k}"]) for k in (1, 3, 5)] == [1, 3, 5]
    assert all(item["source"] == "transformer" for item in predictions["top_5"])
    assert all(item["product"] != "Milk" for item in predictions["top_5"])
    assert all(item["product_id"] in tokenizer.id_to_product for item in predictions["top_5"])
    assert sum(item["score"] for item in predictions["top_5"]) == pytest.approx(1.0)


def test_recommendation_rejects_empty_unknown_and_invalid_inputs():
    tokenizer = ProductTokenizer(["Milk", "Bread", "Butter", "Cereal", "Coffee", "Sugar"])
    model = _small_model()
    with pytest.raises(ValueError, match="at least one"):
        recommend_next([], model=model, tokenizer=tokenizer)
    with pytest.raises(ValueError, match="Unknown product ID"):
        recommend_next([999], model=model, tokenizer=tokenizer)
    with pytest.raises(ValueError, match="Unknown product"):
        recommend_next(["Unlisted item"], model=model, tokenizer=tokenizer)
    with pytest.raises(ValueError, match="Invalid product ID"):
        recommend_next([True], model=model, tokenizer=tokenizer)


def test_frequent_itemsets_compute_support_confidence_lift_and_rules():
    transactions = (
        [["Milk", "Bread"]] * 3
        + [["Milk"]]
        + [["Bread"]]
        + [["Milk", "Bread", "Butter"]]
    )
    itemsets, rules = mine_frequent_itemsets(
        transactions, min_support=0.1, min_confidence=0.1, max_itemset_size=2,
    )
    milk_bread = itemsets[itemsets["itemset"].apply(lambda items: set(items) == {"Milk", "Bread"})].iloc[0]
    assert milk_bread["transaction_count"] == 4
    assert milk_bread["support"] == pytest.approx(4 / 6)
    milk_rule = rules[
        rules["antecedent"].apply(lambda items: items == ("Milk",))
        & rules["consequent"].apply(lambda items: items == ("Bread",))
    ].iloc[0]
    assert milk_rule["confidence"] == pytest.approx(0.8)
    assert milk_rule["lift"] == pytest.approx(0.96)
    probed = probe_itemsets(["Milk"], rules=rules)
    assert probed[0]["product"] == "Bread"
    assert probed[0]["source"] == "association_rule"
    assert probed[0]["confidence"] == pytest.approx(0.8)
    assert probed[0]["lift"] == pytest.approx(0.96)


def test_held_out_recommendation_evaluation_saves_results_without_test_rule_leakage(tmp_path):
    tokenizer = ProductTokenizer(["Milk", "Bread", "Butter", "Cereal"])
    model = _small_model()
    training = [
        ["Milk", "Bread"],
        ["Milk", "Bread", "Butter"],
        ["Milk", "Bread"],
        ["Milk", "Cereal"],
        ["Bread", "Butter"],
        ["Milk", "Bread"],
    ]
    held_out = [["Milk", "Butter"], ["Bread", "Cereal"], ["Milk", "Cereal"]]
    checkpoint = tmp_path / "baseline.pth"
    torch.save(model, checkpoint)
    results = evaluate_recommendations(
        training,
        held_out,
        model,
        tokenizer,
        min_support=0.1,
        min_confidence=0.1,
        example_count=2,
        results_dir=tmp_path / "results",
        model_path=checkpoint,
    )
    comparison = results["comparison"]
    assert results["test_transactions"] == len(held_out)
    assert results["association_rules_fit_on"] == "training_transactions_only"
    assert 0 <= comparison["transformer_top1_accuracy"] <= 1
    assert 0 <= comparison["transformer_top3_accuracy"] <= 1
    assert 0 <= comparison["association_rule_top1_accuracy"] <= 1
    assert 0 <= comparison["association_rule_top3_accuracy"] <= 1
    assert len(results["examples"]) == 2
    assert all(
        example["transformer_predictions"][0]["source"] == "transformer"
        for example in results["examples"]
    )
    assert (tmp_path / "results" / "recommendation_evaluation.json").exists()
    assert (tmp_path / "results" / "frequent_itemsets.csv").exists()
    assert (tmp_path / "results" / "association_rules.csv").exists()


def test_dynamic_quantization_converts_linear_and_saves(tmp_path):
    model_path = tmp_path / "dynamic.pth"
    quantized = quantize_dynamic(_small_model(), model_path)
    converted = [
        layer for layer in quantized.modules()
        if isinstance(layer, torch.ao.nn.quantized.dynamic.Linear)
    ]
    assert converted
    assert all(layer.weight().dtype == torch.qint8 for layer in converted)
    assert model_path.stat().st_size > 0
    assert quantized(torch.tensor([[1, 2]]), torch.tensor([2])).shape == (1, 10)


def test_static_quantization_calibrates_and_converts_linear(tmp_path):
    model_path = tmp_path / "static.pth"
    quantized = quantize_static(_small_model(), _calibration_examples(), model_path, calibration_batches=2)
    converted = [
        layer for layer in quantized.modules()
        if isinstance(layer, torch.ao.nn.quantized.Linear)
    ]
    assert converted
    assert all(layer.weight().dtype == torch.qint8 for layer in converted)
    assert model_path.stat().st_size > 0
    assert quantized(torch.tensor([[1, 2]]), torch.tensor([2])).shape == (1, 10)
    loaded = load_quantized_model(model_path)
    assert loaded(torch.tensor([[1, 2]]), torch.tensor([2])).shape == (1, 10)


def test_quantization_runner_saves_loadable_models_and_preserves_baseline(tmp_path):
    tokenizer = ProductTokenizer(["Milk", "Bread", "Butter", "Cereal"])
    transactions = [
        ["Milk", "Cereal"], ["Bread", "Butter"], ["Milk", "Bread"], ["Cereal", "Butter"],
        ["Bread", "Cereal"], ["Milk", "Butter"], ["Milk", "Cereal"], ["Bread", "Butter"],
        ["Cereal", "Milk"], ["Butter", "Bread"], ["Milk", "Butter"], ["Cereal", "Bread"],
        ["Milk", "Cereal"], ["Bread", "Butter"], ["Milk", "Bread"], ["Butter", "Cereal"],
        ["Milk", "Butter"], ["Bread", "Cereal"], ["Milk", "Cereal"], ["Bread", "Butter"],
    ]
    dataset_path = tmp_path / "transactions.csv"
    pd.DataFrame({
        "transaction_id": range(1, len(transactions) + 1),
        "products": [json.dumps(basket) for basket in transactions],
        "item_count": [len(basket) for basket in transactions],
    }).to_csv(dataset_path, index=False)
    tokenizer_path = tmp_path / "vocabulary.json"
    tokenizer.save(tokenizer_path)
    baseline_path = tmp_path / "baseline.pth"
    baseline = RetailLanguageModel(
        vocab_size=tokenizer.vocab_size, max_length=12, d_model=16, n_heads=4,
        num_layers=1, dropout=0,
    )
    torch.save(baseline.eval(), baseline_path)
    baseline_bytes = baseline_path.read_bytes()
    report = quantize_saved_models(
        baseline_path=baseline_path,
        tokenizer_path=tokenizer_path,
        dataset_path=dataset_path,
        baseline_metrics_path=tmp_path / "missing-baseline-metrics.json",
        dynamic_model_path=tmp_path / "dynamic_int8" / "model.pth",
        static_model_path=tmp_path / "static_int8" / "model.pth",
        calibration_batches=1,
        calibration_batch_size=4,
        report_path=tmp_path / "results" / "quantization_artifacts.json",
    )
    assert baseline_path.read_bytes() == baseline_bytes
    assert report["baseline_checkpoint_unchanged"]
    assert report["calibration"]["calibration_examples_used"] == 4
    assert report["models"]["dynamic_int8"]["quantized_linear_count"] > 0
    assert report["models"]["static_int8"]["quantized_linear_count"] > 0
    assert report["models"]["dynamic_int8"]["checkpoint_size_bytes"] > 0
    assert report["models"]["static_int8"]["checkpoint_size_bytes"] > 0
    assert report["models"]["dynamic_int8"]["inference_verification"]["sample_basket_count"] >= 1
    assert report["models"]["static_int8"]["inference_verification"]["sample_basket_count"] >= 1
    assert Path(report["models"]["dynamic_int8"]["configuration_path"]).exists()
    assert Path(report["models"]["static_int8"]["configuration_path"]).exists()
    assert Path(tmp_path / "results" / "quantization_artifacts.json").exists()


def test_evaluation_returns_accuracy_loss_size_and_latency(tmp_path):
    tokenizer = ProductTokenizer(["Milk", "Bread", "Butter", "Cereal", "Coffee", "Sugar"])
    model = _small_model()
    checkpoint = tmp_path / "model.pth"
    torch.save(model, checkpoint)
    assert isinstance(load_model(checkpoint), RetailLanguageModel)
    results = evaluate_model(model, [["Milk", "Bread", "Butter"], ["Cereal", "Milk"]], tokenizer,
                             checkpoint, latency_samples=2)
    assert 0 <= results["top1_accuracy"] <= 1
    assert 0 <= results["top3_accuracy"] <= 1
    assert results["test_loss"] > 0
    assert results["model_size_mb"] > 0
    assert results["average_inference_time_ms"] > 0
    assert results["inference_benchmark"]["device"] == "cpu"
    assert results["inference_benchmark"]["warmup_iterations"] == 20
    assert results["inference_benchmark"]["repetitions"] == 5
    assert results["inference_benchmark"]["samples_per_repetition"] == 2


def test_cpu_benchmark_reuses_same_fixed_inputs_and_reports_repetitions():
    model = _small_model()
    examples = [([1], 2), ([2, 1], 3), ([3, 2, 1], 4)]
    result = benchmark_cpu_inference(
        model, examples, warmup_iterations=2, repetitions=3, sample_count=2,
    )
    assert result["device"] == "cpu"
    assert result["warmup_iterations"] == 2
    assert result["repetitions"] == 3
    assert result["samples_per_repetition"] == 2
    assert len(result["repetition_average_latency_ms"]) == 3
    assert all(value > 0 for value in result["repetition_average_latency_ms"])
    assert result["input_sha256"] == benchmark_cpu_inference(
        model, examples, warmup_iterations=0, repetitions=1, sample_count=2,
    )["input_sha256"]
    assert model_memory_size_bytes(model) > 0


def test_baseline_evaluation_saves_real_metrics_without_changing_checkpoint(tmp_path):
    tokenizer = ProductTokenizer(["Milk", "Bread", "Butter", "Cereal"])
    transactions = [
        ["Milk", "Cereal"], ["Bread", "Butter"], ["Milk", "Bread"], ["Cereal", "Butter"],
        ["Bread", "Cereal"], ["Milk", "Butter"], ["Milk", "Cereal"], ["Bread", "Butter"],
        ["Cereal", "Milk"], ["Butter", "Bread"], ["Milk", "Butter"], ["Cereal", "Bread"],
        ["Milk", "Cereal"], ["Bread", "Butter"], ["Milk", "Bread"], ["Butter", "Cereal"],
        ["Milk", "Butter"], ["Bread", "Cereal"], ["Milk", "Cereal"], ["Bread", "Butter"],
    ]
    data_path = tmp_path / "transactions.csv"
    pd.DataFrame({
        "transaction_id": range(1, len(transactions) + 1),
        "products": [json.dumps(basket) for basket in transactions],
        "item_count": [len(basket) for basket in transactions],
    }).to_csv(data_path, index=False)
    vocabulary_path = tmp_path / "vocabulary.json"
    tokenizer.save(vocabulary_path)
    model = RetailLanguageModel(
        vocab_size=tokenizer.vocab_size, max_length=12, d_model=16, n_heads=4,
        num_layers=1, dropout=0,
    )
    checkpoint_path = tmp_path / "baseline.pth"
    torch.save(model.eval(), checkpoint_path)
    checkpoint_bytes_before = checkpoint_path.read_bytes()
    metrics_path = tmp_path / "results" / "baseline_metrics.json"
    csv_path = tmp_path / "results" / "baseline_comparison.csv"
    results = run_baseline_evaluation(
        model_path=checkpoint_path,
        tokenizer_path=vocabulary_path,
        dataset_path=data_path,
        results_path=metrics_path,
        table_path=csv_path,
        warmup_iterations=1,
        repetitions=2,
        benchmark_samples=3,
    )
    saved_results = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert results["checkpoint_sha256"] == saved_results["checkpoint_sha256"]
    assert checkpoint_path.read_bytes() == checkpoint_bytes_before
    assert saved_results["transaction_split_sizes"] == {
        "train": 16, "validation": 2, "test": 2,
    }
    assert saved_results["cpu_inference"]["repetitions"] == 2
    assert saved_results["cpu_inference"]["samples_per_repetition"] == 3
    assert saved_results["checkpoint_size_bytes"] == checkpoint_path.stat().st_size
    assert saved_results["parameter_count"] == sum(parameter.numel() for parameter in model.parameters())
    assert pd.read_csv(csv_path).iloc[0]["device"] == "cpu"


def test_causal_attention_cannot_see_future_positions():
    attention = CausalSelfAttention(d_model=8, n_heads=2, dropout=0).eval()
    padding_mask = torch.zeros((1, 3), dtype=torch.bool)
    original = torch.tensor([[[1.0] * 8, [2.0] * 8, [3.0] * 8]])
    changed_future = torch.tensor([[[1.0] * 8, [20.0] * 8, [30.0] * 8]])
    with torch.inference_mode():
        original_first = attention(original, padding_mask)[:, 0]
        changed_first = attention(changed_future, padding_mask)[:, 0]
    torch.testing.assert_close(original_first, changed_first)


def test_training_saves_reloadable_weights_config_metrics_and_loss_plot(tmp_path):
    from src.train import train_model

    tokenizer = ProductTokenizer(["Milk", "Bread", "Butter", "Cereal"])
    transactions = [
        ["Milk", "Cereal"],
        ["Bread", "Butter"],
        ["Milk", "Bread"],
        ["Cereal", "Butter"],
        ["Bread", "Cereal"],
        ["Milk", "Butter"],
        ["Milk", "Cereal"],
        ["Bread", "Butter"],
        ["Cereal", "Milk"],
        ["Butter", "Bread"],
    ]
    train, validation, test = split_transactions(transactions, seed=4)
    model_path = tmp_path / "baseline_model.pth"
    vocabulary_path = tmp_path / "product_vocabulary.json"
    metrics_path = tmp_path / "training_metrics.json"
    plot_path = tmp_path / "training_loss.png"
    model, history = train_model(
        tokenizer=tokenizer,
        train_transactions=train,
        validation_transactions=validation,
        test_transactions=test,
        epochs=1,
        batch_size=4,
        model_path=model_path,
        vocabulary_path=vocabulary_path,
        metrics_path=metrics_path,
        plot_path=plot_path,
    )
    weights_path = tmp_path / "baseline_model_state_dict.pth"
    config_path = tmp_path / "baseline_model_config.json"
    assert isinstance(load_model(model_path), RetailLanguageModel)
    restored = load_model_weights(weights_path, config_path)
    assert restored(torch.tensor([[tokenizer.bos_id]]), torch.tensor([1])).shape == (
        1, tokenizer.vocab_size,
    )
    assert history["train_loss"] and history["validation_loss"]
    assert json.loads(metrics_path.read_text())["transaction_split_sizes"] == {
        "train": 8, "validation": 1, "test": 1,
    }
    assert vocabulary_path.exists() and plot_path.exists() and weights_path.exists()


def test_comparison_reports_measured_size_and_accuracy_differences(monkeypatch, tmp_path):
    import src.evaluate as evaluation

    monkeypatch.setattr(evaluation, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(evaluation, "PLOTS_DIR", tmp_path / "plots")
    sample = {
        "model_size_mb": 10.0,
        "top1_accuracy": 0.5,
        "top3_accuracy": 0.8,
        "test_loss": 1.0,
        "average_inference_time_ms": 2.0,
        "parameter_count": 100,
    }
    dynamic = {**sample, "model_size_mb": 5.0, "top1_accuracy": 0.45}
    static = {**sample, "model_size_mb": 7.0, "top1_accuracy": 0.55}
    comparison = build_comparison({
        "baseline": sample, "dynamic_int8": dynamic, "static_int8": static,
    })
    assert comparison["size_reduction_percent"].tolist() == [0.0, 50.0, 30.0]
    assert comparison["absolute_top1_accuracy_difference"].tolist() == pytest.approx([0.0, 0.05, 0.05])
    assert comparison["inference_time_difference_ms"].tolist() == [0.0, 0.0, 0.0]
    assert (tmp_path / "comparison.csv").exists()


def test_quantization_comparison_calculates_signed_changes_and_zero_safely():
    def metrics(size, top1, top3, latency):
        return {
            "model_size_mb": size,
            "checkpoint_size_bytes": int(size * 1_000_000),
            "parameter_count": 100,
            "quantized_linear_count": 0,
            "top1_accuracy": top1,
            "top3_accuracy": top3,
            "test_loss": 1.0,
            "average_inference_time_ms": latency,
            "inference_benchmark": {
                "median_latency_ms": latency,
                "p95_latency_ms": latency,
            },
            "test_examples": 10,
        }

    rows = _comparison_rows({
        "baseline": metrics(1.0, 0.5, 0.8, 2.0),
        "dynamic_int8": metrics(0.5, 0.45, 0.75, 1.0),
        "static_int8": metrics(0.7, 0.55, 0.82, 3.0),
    })
    dynamic = rows[1]
    static = rows[2]
    assert dynamic["size_reduction_percent"] == pytest.approx(50.0)
    assert dynamic["top1_accuracy_change_percentage_points"] == pytest.approx(-5.0)
    assert dynamic["top1_relative_accuracy_change_percent"] == pytest.approx(-10.0)
    assert dynamic["latency_improvement_percent"] == pytest.approx(50.0)
    assert static["top1_accuracy_change_percentage_points"] == pytest.approx(5.0)
    assert static["latency_improvement_percent"] == pytest.approx(-50.0)

    zero_baseline = _comparison_rows({
        "baseline": metrics(1.0, 0.0, 0.0, 0.0),
        "dynamic_int8": metrics(0.5, 0.1, 0.2, 1.0),
        "static_int8": metrics(0.7, 0.05, 0.15, 2.0),
    })
    assert zero_baseline[1]["top1_relative_accuracy_change_percent"] is None
    assert zero_baseline[1]["latency_improvement_percent"] is None
    analysis = _make_analysis(rows)
    assert analysis["greatest_size_reduction"]["model"] == "Dynamic INT8"
    assert analysis["best_quantized_top1_accuracy"]["model"] == "Static INT8"
