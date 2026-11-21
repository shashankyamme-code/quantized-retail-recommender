"""Interactive Streamlit dashboard for the quantized retail recommender."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dataset import (
    PRODUCT_CATEGORIES,
    ProductTokenizer,
    load_transaction_baskets,
    split_transactions,
)
from src.quantization import load_quantized_model
from src.recommend import mine_frequent_itemsets, probe_itemsets, recommend_top_k
from src.train import BASELINE_MODEL_PATH, load_model

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "retail_transactions.csv"
TOKENIZER_PATH = ROOT / "data" / "product_vocabulary.json"
BASELINE_PATH = ROOT / BASELINE_MODEL_PATH
MODEL_PATHS = {
    "Baseline": BASELINE_PATH,
    "Dynamic INT8": ROOT / "models" / "dynamic_int8" / "model.pth",
    "Static INT8": ROOT / "models" / "static_int8" / "model.pth",
}
COMPARISON_JSON_PATH = ROOT / "results" / "quantization_comparison.json"
COMPARISON_CSV_PATH = ROOT / "results" / "quantization_comparison.csv"
ITEMSETS_PATH = ROOT / "results" / "frequent_itemsets.csv"
RULES_PATH = ROOT / "results" / "association_rules.csv"
ASSOCIATION_EVALUATION_PATH = ROOT / "results" / "recommendation_evaluation.json"

st.set_page_config(
    page_title="Quantized Retail Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --retail-ink: #173b3f;
        --retail-green: #167c70;
        --retail-mint: #e9f4f0;
        --retail-sand: #f5f2ea;
    }
    .stApp { background: linear-gradient(180deg, #f8faf8 0%, #ffffff 36%); }
    h1, h2, h3 { color: var(--retail-ink); letter-spacing: -0.025em; }
    h1 { font-weight: 750; }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #e4ece8; border-radius: 14px;
        padding: 16px 18px; box-shadow: 0 3px 14px rgba(23, 59, 63, 0.045);
    }
    [data-testid="stSidebar"] { background: #f2f7f4; }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: 0.35rem; }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: var(--retail-green); border-color: var(--retail-green);
    }
    .dashboard-kicker {
        color: var(--retail-green); font-size: 0.76rem; font-weight: 700;
        letter-spacing: 0.13em; text-transform: uppercase; margin-bottom: -0.5rem;
    }
    .dashboard-subtitle { color: #53686a; font-size: 1.05rem; }
    .dashboard-card {
        background: white; border: 1px solid #e4ece8; border-radius: 14px;
        padding: 1rem 1.15rem; height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_tokenizer(path: str, modified_ns: int) -> ProductTokenizer:
    del modified_ns
    return ProductTokenizer.load(path)


@st.cache_resource
def get_model(model_name: str, path: str, modified_ns: int):
    del modified_ns
    if model_name == "Baseline":
        return load_model(path, map_location="cpu")
    return load_quantized_model(path)


@st.cache_data(show_spinner=False)
def get_dataset(path: str, modified_ns: int) -> tuple[pd.DataFrame, list[list[str]]]:
    del modified_ns
    return pd.read_csv(path), load_transaction_baskets(path)


@st.cache_data(show_spinner=False)
def get_comparison(path: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_json_file(path: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_itemset_artifacts(
    itemsets_path: str,
    rules_path: str,
    itemsets_modified_ns: int,
    rules_modified_ns: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del itemsets_modified_ns, rules_modified_ns
    itemsets = pd.read_csv(itemsets_path)
    rules = pd.read_csv(rules_path)
    itemsets["products"] = itemsets["itemset"].map(
        lambda value: tuple(json.loads(value))
    )
    rules["antecedent_products"] = rules["antecedent"].map(
        lambda value: tuple(json.loads(value))
    )
    rules["consequent_products"] = rules["consequent"].map(
        lambda value: tuple(json.loads(value))
    )
    return itemsets, rules


@st.cache_data(show_spinner="Mining training-split associations…")
def mine_itemset_fallback(
    training_transactions: tuple[tuple[str, ...], ...],
    min_support: float,
    min_confidence: float,
    max_itemset_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return mine_frequent_itemsets(
        [list(basket) for basket in training_transactions],
        min_support=min_support,
        min_confidence=min_confidence,
        max_itemset_size=max_itemset_size,
    )


@st.cache_data(show_spinner=False)
def dataset_summaries(
    transactions: tuple[tuple[str, ...], ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    product_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    basket_sizes: Counter[int] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for basket in transactions:
        product_counts.update(set(basket))
        category_counts.update(
            PRODUCT_CATEGORIES.get(product, "Other") for product in set(basket)
        )
        basket_sizes[len(set(basket))] += 1
        pair_counts.update(itertools.combinations(sorted(set(basket)), 2))
    products = pd.DataFrame(
        [
            {
                "product": product,
                "category": PRODUCT_CATEGORIES.get(product, "Other"),
                "transaction_count": count,
            }
            for product, count in product_counts.most_common()
        ],
    )
    categories = pd.DataFrame(
        [{"category": category, "transaction_count": count}
         for category, count in category_counts.most_common()],
    )
    sizes = pd.DataFrame(
        [{"items_per_basket": size, "transactions": count}
         for size, count in sorted(basket_sizes.items())],
    )
    pairs = pd.DataFrame(
        [
            {
                "combination": f"{left} + {right}",
                "transaction_count": count,
            }
            for (left, right), count in pair_counts.most_common(30)
        ],
    )
    return products, categories, sizes, pairs


def _modified_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


def _page_header(title: str, subtitle: str, section: str) -> None:
    st.markdown(f'<p class="dashboard-kicker">{section}</p>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="dashboard-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def _load_dashboard_data() -> tuple[pd.DataFrame, list[list[str]]] | None:
    if not DATASET_PATH.is_file():
        st.warning(
            "The transaction dataset is missing. Generate it with "
            "`python -m src.pipeline --force-data`."
        )
        return None
    try:
        return get_dataset(str(DATASET_PATH), _modified_ns(DATASET_PATH))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not load the transaction dataset: {error}")
        return None


def _load_dashboard_tokenizer() -> ProductTokenizer | None:
    if not TOKENIZER_PATH.is_file():
        st.warning(
            "The product vocabulary is missing. Run `python -m src.pipeline` to create it."
        )
        return None
    try:
        return get_tokenizer(str(TOKENIZER_PATH), _modified_ns(TOKENIZER_PATH))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not load the product vocabulary: {error}")
        return None


def _load_comparison() -> dict[str, Any] | None:
    if not COMPARISON_JSON_PATH.is_file():
        st.warning(
            "Quantization evaluation is not available. Run "
            "`python -m src.evaluate_quantization` after creating all three checkpoints."
        )
        return None
    try:
        return get_comparison(
            str(COMPARISON_JSON_PATH), _modified_ns(COMPARISON_JSON_PATH),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not load quantization results: {error}")
        return None


def project_overview() -> None:
    _page_header(
        "Quantized Retail Recommender",
        "Learn what sells together, recommend a next item, and inspect the real cost of quantization.",
        "Retail intelligence · model efficiency",
    )
    st.markdown(
        """
        <div class="dashboard-card">
        <strong>Project objective</strong><br>
        Predict the next product in a shopping basket with a compact autoregressive
        Transformer, then compare its original and INT8 variants on a fixed held-out set.
        This project uses synthetic retail data and trains its model from scratch; it does
        not use continued pretraining or paid APIs.
        </div>
        """,
        unsafe_allow_html=True,
    )
    data = _load_dashboard_data()
    tokenizer = _load_dashboard_tokenizer()
    comparison = _load_comparison()

    transactions = data[1] if data else []
    product_count = len(tokenizer.products) if tokenizer else len(PRODUCT_CATEGORIES)
    category_count = len({
        PRODUCT_CATEGORIES.get(product, "Other")
        for product in (tokenizer.products if tokenizer else PRODUCT_CATEGORIES)
    })
    model_rows = {
        model["model_key"]: model for model in comparison.get("comparison", [])
    } if comparison else {}

    st.subheader("At a glance")
    kpi_columns = st.columns(4)
    kpi_columns[0].metric("Transactions", f"{len(transactions):,}" if data else "—")
    kpi_columns[1].metric("Products", f"{product_count:,}")
    kpi_columns[2].metric("Product categories", f"{category_count:,}")
    baseline_size = model_rows.get("baseline", {}).get("model_size_mb")
    kpi_columns[3].metric(
        "Baseline checkpoint",
        f"{baseline_size:.3f} MB" if baseline_size is not None else "Unavailable",
    )

    if model_rows:
        baseline_size = model_rows.get("baseline", {}).get("model_size_mb", 0.0)
        quantized = [
            row for key, row in model_rows.items() if key != "baseline"
        ]
        if quantized and baseline_size:
            smallest = min(quantized, key=lambda row: row["model_size_mb"])
            st.metric(
                f"Maximum measured compression · {smallest['model']}",
                f"{smallest['size_reduction_percent']:.2f}%",
                help="Calculated from actual serialized checkpoint sizes in the latest evaluation.",
            )
        st.subheader("Saved model sizes")
        size_frame = pd.DataFrame([
            {
                "Model": row["model"],
                "Checkpoint size (MB)": row["model_size_mb"],
            }
            for row in model_rows.values()
        ])
        st.plotly_chart(
            px.bar(
                size_frame,
                x="Model",
                y="Checkpoint size (MB)",
                color="Model",
                color_discrete_sequence=["#167c70", "#e7a84b", "#6f8f95"],
            ),
            width="stretch",
        )
    else:
        st.info("Model-size and compression cards populate after quantization evaluation.")

    with st.expander("How the system works"):
        st.write(
            "Synthetic baskets are cleaned and tokenized, whole transactions are split "
            "before next-item examples are built, and a two-layer causal Transformer predicts "
            "the next product. Dynamic and calibrated static INT8 models quantize supported "
            "Linear layers; unsupported Transformer operations remain floating point."
        )


def _format_recommendation_frame(
    predictions: list[dict[str, Any]],
) -> pd.DataFrame:
    frame = pd.DataFrame(predictions)
    if frame.empty:
        return frame
    frame["score"] = frame["score"].map(lambda score: f"{score:.2%}")
    return frame.rename(columns={
        "product": "Recommended product",
        "category": "Category",
        "score": "Model score",
    })[["Recommended product", "Category", "Model score"]]


def shopping_recommendations() -> None:
    _page_header(
        "Shopping basket recommendations",
        "Rank likely next purchases with the trained Transformer. Compare its saved INT8 variants when available.",
        "Next-item prediction",
    )
    tokenizer = _load_dashboard_tokenizer()
    if tokenizer is None:
        return
    available = [
        name for name, path in MODEL_PATHS.items() if path.is_file()
    ]
    if not available:
        st.warning(
            "No model checkpoints were found. Run `python -m src.train` and then "
            "`python -m src.quantize_models` to create the model artifacts."
        )
        return

    selected_products = st.multiselect(
        "Choose products already in the basket",
        options=tokenizer.products,
        help="Already selected products are excluded from model recommendations.",
    )
    typed_products = st.text_input(
        "Or enter product names separated by commas",
        placeholder="Milk, Bread, Butter",
    )
    typed = [product.strip() for product in typed_products.split(",") if product.strip()]
    basket = list(dict.fromkeys([*selected_products, *typed]))
    compare_variants = st.checkbox(
        "Compare baseline, Dynamic INT8, and Static INT8 predictions",
        value=False,
        disabled=len(available) < 3,
    )
    if len(available) < 3:
        st.caption("Model comparison is disabled until all three checkpoint files are present.")

    if st.button("Predict next items", type="primary"):
        if not basket:
            st.info("Choose or enter at least one product to get recommendations.")
            return
        unknown = [product for product in basket if product not in tokenizer.product_to_id]
        if unknown:
            st.error(
                "Unknown product name(s): "
                + ", ".join(unknown)
                + ". Select a product from the list or check its spelling."
            )
            return

        model_names = list(MODEL_PATHS) if compare_variants else ["Baseline"]
        missing_models = [name for name in model_names if not MODEL_PATHS[name].is_file()]
        if missing_models:
            st.error("Missing checkpoint(s): " + ", ".join(missing_models))
            return
        model_predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
        try:
            for model_name in model_names:
                path = MODEL_PATHS[model_name]
                model = get_model(model_name, str(path), _modified_ns(path))
                model_predictions[model_name] = recommend_top_k(
                    basket, model=model, tokenizer=tokenizer,
                )
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
            st.error(f"Could not generate recommendations: {error}")
            return

        st.success(f"Basket contains {len(basket)} unique product(s).")
        st.caption("Prediction scores are Transformer probabilities, not association-rule confidence.")
        tabs = st.tabs(["Top 1", "Top 3", "Top 5"])
        for limit, tab in zip((1, 3, 5), tabs):
            with tab:
                if compare_variants:
                    columns = st.columns(len(model_predictions))
                    for column, (model_name, predictions) in zip(
                        columns, model_predictions.items(),
                    ):
                        with column:
                            st.markdown(f"**{model_name}**")
                            frame = _format_recommendation_frame(
                                predictions[f"top_{limit}"],
                            )
                            if frame.empty:
                                st.info("No eligible products remain.")
                            else:
                                st.dataframe(
                                    frame, hide_index=True, width="stretch",
                                )
                else:
                    predictions = model_predictions["Baseline"][f"top_{limit}"]
                    if predictions:
                        first = predictions[0]
                        st.metric(
                            "Top recommendation",
                            first["product"],
                            f"{first['score']:.2%} model score",
                        )
                        st.dataframe(
                            _format_recommendation_frame(predictions),
                            hide_index=True,
                            width="stretch",
                        )
                    else:
                        st.info("No eligible products remain after excluding the basket.")


def frequently_bought_together() -> None:
    _page_header(
        "Frequently bought together",
        "Explore observed co-purchase patterns and association rules, separately from neural model predictions.",
        "Association-rule analysis",
    )
    data = _load_dashboard_data()
    if data is None:
        return
    baskets = data[1]
    train_transactions, _, _ = split_transactions(baskets, seed=42)
    evaluation = None
    if ASSOCIATION_EVALUATION_PATH.is_file():
        try:
            evaluation = get_json_file(
                str(ASSOCIATION_EVALUATION_PATH),
                _modified_ns(ASSOCIATION_EVALUATION_PATH),
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            st.warning(f"Could not verify the association-rule report: {error}")
    saved_artifacts_exist = (
        ITEMSETS_PATH.is_file()
        and RULES_PATH.is_file()
        and evaluation is not None
        and evaluation.get("association_rules_fit_on") == "training_transactions_only"
    )
    saved_parameters = (
        evaluation.get("itemset_parameters", {}) if saved_artifacts_exist else {}
    )
    support_floor = float(saved_parameters.get("min_support", 0.005))
    confidence_floor = float(saved_parameters.get("min_confidence", 0.20))
    itemset_limit = int(saved_parameters.get("max_itemset_size", 3))
    support_max = max(0.10, support_floor)
    support_default = min(max(0.02, support_floor), support_max)
    confidence_default = min(max(0.30, confidence_floor), 1.0)
    minimum_support = st.slider(
        "Minimum itemset support",
        min_value=support_floor,
        max_value=support_max,
        value=support_default,
        step=0.005,
        format="%.3f",
    )
    minimum_confidence = st.slider(
        "Minimum rule confidence",
        min_value=confidence_floor,
        max_value=1.0,
        value=confidence_default,
        step=0.05,
        format="%.2f",
    )
    max_itemset_size = st.slider(
        "Maximum itemset size",
        min_value=1,
        max_value=max(1, min(5, itemset_limit)),
        value=max(1, min(3, itemset_limit)),
    )

    if saved_artifacts_exist:
        try:
            itemsets, rules = get_itemset_artifacts(
                str(ITEMSETS_PATH),
                str(RULES_PATH),
                _modified_ns(ITEMSETS_PATH),
                _modified_ns(RULES_PATH),
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            st.error(f"Could not read the saved itemset artifacts: {error}")
            return
        st.caption(
            f"Using the saved itemsets and rules, mined on the training split only "
            f"({len(train_transactions):,} transactions). Controls filter the thresholds "
            "recorded in the saved evaluation."
        )
    else:
        training_key = tuple(tuple(basket) for basket in train_transactions)
        itemsets, rules = mine_itemset_fallback(
            training_key,
            minimum_support,
            minimum_confidence,
            max_itemset_size,
        )
        itemsets = itemsets.assign(products=itemsets["itemset"])
        rules = rules.assign(
            antecedent_products=rules["antecedent"],
            consequent_products=rules["consequent"],
        )
        st.info(
            "Saved itemset CSVs were not found. Results are being mined from the training "
            "transaction split only; held-out transactions are not used."
        )

    filtered_itemsets = itemsets[
        (itemsets["support"] >= minimum_support)
        & (itemsets["itemset_size"] <= max_itemset_size)
    ].copy()
    selected_products = st.multiselect(
        "Explore rules for products in a basket",
        options=sorted(PRODUCT_CATEGORIES),
        help="Association rules are mined on training data and never use the test split.",
    )
    if selected_products:
        target = set(selected_products)
        filtered_itemsets = filtered_itemsets[
            filtered_itemsets["products"].map(lambda products: target.issubset(set(products)))
        ]

    st.subheader("Frequent itemsets")
    if filtered_itemsets.empty:
        st.info("No itemsets match these filters. Lower the support threshold or change the products.")
    else:
        display_itemsets = filtered_itemsets.copy()
        display_itemsets["Product combination"] = display_itemsets["products"].map(
            lambda products: " + ".join(products)
        )
        display_itemsets["Support"] = display_itemsets["support"].map(
            lambda support: f"{support:.2%}"
        )
        st.dataframe(
            display_itemsets[[
                "Product combination", "itemset_size", "Support", "transaction_count",
            ]].rename(columns={
                "itemset_size": "Items",
                "transaction_count": "Training transactions",
            }).head(30),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Association rules")
    filtered_rules = rules[
        (rules["support"] >= minimum_support)
        & (rules["confidence"] >= minimum_confidence)
        & (
            rules["antecedent_products"].map(len)
            + rules["consequent_products"].map(len)
            <= max_itemset_size
        )
    ].copy()
    if selected_products:
        try:
            probes = probe_itemsets(selected_products, rules=filtered_rules)
        except ValueError as error:
            st.error(f"Could not probe these products: {error}")
            probes = []
        if probes:
            probe_frame = pd.DataFrame(probes)
            probe_frame["support"] = probe_frame["support"].map(lambda value: f"{value:.2%}")
            probe_frame["confidence"] = probe_frame["confidence"].map(
                lambda value: f"{value:.2%}"
            )
            probe_frame["lift"] = probe_frame["lift"].map(lambda value: f"{value:.2f}")
            st.dataframe(
                probe_frame[[
                    "antecedent", "product", "support", "confidence", "lift",
                ]].rename(columns={
                    "antecedent": "Basket match",
                    "product": "Associated product",
                    "support": "Support",
                    "confidence": "Confidence",
                    "lift": "Lift",
                }),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No associated products meet the selected rule thresholds.")
    elif filtered_rules.empty:
        st.info("No rules match these thresholds.")
    else:
        display_rules = filtered_rules.copy()
        display_rules["If basket contains"] = display_rules["antecedent_products"].map(
            lambda products: " + ".join(products)
        )
        display_rules["Then associated with"] = display_rules["consequent_products"].map(
            lambda products: " + ".join(products)
        )
        for column in ("support", "confidence"):
            display_rules[column] = display_rules[column].map(lambda value: f"{value:.2%}")
        display_rules["lift"] = display_rules["lift"].map(lambda value: f"{value:.2f}")
        st.dataframe(
            display_rules[[
                "If basket contains", "Then associated with",
                "support", "confidence", "lift",
            ]].rename(columns={
                "support": "Support",
                "confidence": "Confidence",
                "lift": "Lift",
            }).head(30),
            hide_index=True,
            width="stretch",
        )
    st.caption(
        "Support = share of training baskets containing the combination; confidence = "
        "share of antecedent baskets also containing the consequent; lift compares that "
        "confidence with the consequent's baseline frequency. These are empirical rules, "
        "not Transformer scores."
    )


def quantization_comparison() -> None:
    _page_header(
        "Quantization comparison",
        "Compare actual checkpoint size, held-out accuracy, and CPU inference timing.",
        "Measured model performance",
    )
    report = _load_comparison()
    if report is None:
        return
    rows = report.get("comparison", [])
    if not rows:
        st.warning("The comparison report contains no model measurements.")
        return
    frame = pd.DataFrame(rows)
    analysis = report.get("analysis", {})
    tradeoff = analysis.get("best_tradeoff", {})
    st.info(
        f"Measured trade-off: **{tradeoff.get('model', 'Unavailable')}** is the preferred "
        "quantized variant under the report's accuracy-first rule. Baseline remains the "
        "latency reference; actual results depend on the CPU and runtime."
    )

    top1_model = analysis.get("best_recommendation_accuracy_overall", {}).get(
        "top1_model", "—",
    )
    smallest = analysis.get("greatest_size_reduction", {})
    best_reduction = smallest.get("size_reduction_percent")
    first, second, third = st.columns(3)
    first.metric("Best measured recommendation accuracy", top1_model)
    second.metric(
        "Greatest size reduction",
        f"{best_reduction:.2f}%" if best_reduction is not None else "—",
        smallest.get("model", ""),
    )
    third.metric(
        "Fastest measured model",
        analysis.get("latency", {}).get("fastest_model_overall", "—"),
    )

    table = frame[[
        "model",
        "model_size_mb",
        "size_reduction_percent",
        "top1_accuracy_percent",
        "top1_accuracy_change_percentage_points",
        "top1_relative_accuracy_change_percent",
        "top3_accuracy_percent",
        "test_loss",
        "average_cpu_inference_latency_ms",
        "latency_improvement_percent",
        "parameter_count",
    ]].rename(columns={
        "model": "Model",
        "model_size_mb": "Checkpoint size (MB)",
        "size_reduction_percent": "Size reduction (%)",
        "top1_accuracy_percent": "Top-1 accuracy (%)",
        "top1_accuracy_change_percentage_points": "Top-1 change (pp)",
        "top1_relative_accuracy_change_percent": "Top-1 relative change (%)",
        "top3_accuracy_percent": "Top-3 accuracy (%)",
        "test_loss": "Test loss",
        "average_cpu_inference_latency_ms": "CPU latency (ms/example)",
        "latency_improvement_percent": "Latency improvement (%)",
        "parameter_count": "Parameters",
    })
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "Accuracy change is shown in percentage points (pp); relative accuracy change is "
        "a percent of baseline accuracy. Positive latency improvement means faster than "
        "baseline; a negative value means slower. Latencies are CPU-only."
    )

    size_column, accuracy_column = st.columns(2)
    with size_column:
        st.plotly_chart(
            px.bar(
                frame, x="model", y="model_size_mb", color="model",
                text="model_size_mb", title="Saved checkpoint size",
                labels={"model": "Model", "model_size_mb": "Checkpoint (MB)"},
            ),
            width="stretch",
        )
    with accuracy_column:
        accuracy_frame = frame.melt(
            id_vars="model",
            value_vars=["top1_accuracy_percent", "top3_accuracy_percent"],
            var_name="Metric",
            value_name="Accuracy (%)",
        )
        accuracy_frame["Metric"] = accuracy_frame["Metric"].map({
            "top1_accuracy_percent": "Top-1",
            "top3_accuracy_percent": "Top-3",
        })
        st.plotly_chart(
            px.bar(
                accuracy_frame, x="model", y="Accuracy (%)", color="Metric",
                barmode="group", title="Held-out recommendation accuracy",
            ),
            width="stretch",
        )
    latency_column, tradeoff_column = st.columns(2)
    with latency_column:
        st.plotly_chart(
            px.bar(
                frame, x="model", y="average_cpu_inference_latency_ms", color="model",
                title="Average CPU inference latency",
                labels={
                    "model": "Model",
                    "average_cpu_inference_latency_ms": "Milliseconds per example",
                },
            ),
            width="stretch",
        )
    with tradeoff_column:
        st.plotly_chart(
            px.scatter(
                frame,
                x="model_size_mb",
                y="top1_accuracy_percent",
                color="model",
                size="top3_accuracy_percent",
                text="model",
                title="Compression versus accuracy",
                labels={
                    "model_size_mb": "Checkpoint size (MB)",
                    "top1_accuracy_percent": "Top-1 accuracy (%)",
                },
            ),
            width="stretch",
        )
    with st.expander("Evaluation methodology and measured interpretation"):
        st.write(
            report.get("methodology", {}).get("transaction_split", "—")
            + ". "
            + report.get("methodology", {}).get("latency", "—")
        )
        st.write(analysis.get("accuracy_cost", "No accuracy analysis available."))
        st.write(
            analysis.get("latency", {}).get(
                "quantized_vs_baseline", "No latency analysis available.",
            )
        )


def dataset_analytics() -> None:
    _page_header(
        "Dataset analytics",
        "Inspect product popularity, category mix, pairings, and basket-size patterns.",
        "Synthetic transaction analysis",
    )
    data = _load_dashboard_data()
    if data is None:
        return
    _, baskets = data
    if not baskets:
        st.warning("The dataset contains no valid transaction baskets.")
        return
    products, categories, basket_sizes, pairs = dataset_summaries(
        tuple(tuple(basket) for basket in baskets)
    )
    average_size = sum(len(set(basket)) for basket in baskets) / len(baskets)
    columns = st.columns(4)
    columns[0].metric("Transactions", f"{len(baskets):,}")
    columns[1].metric("Products purchased", f"{len(products):,}")
    columns[2].metric("Categories", f"{len(categories):,}")
    columns[3].metric("Average basket size", f"{average_size:.2f} items")

    top_n = st.slider("Products to show", min_value=5, max_value=30, value=15, step=1)
    category_options = ["All categories", *categories["category"].tolist()]
    selected_category = st.selectbox("Filter product rankings by category", category_options)
    filtered_products = products if selected_category == "All categories" else products[
        products["category"] == selected_category
    ]
    product_column, category_column = st.columns(2)
    with product_column:
        st.subheader("Most frequently purchased products")
        chart_products = filtered_products.head(top_n).sort_values("transaction_count")
        st.plotly_chart(
            px.bar(
                chart_products,
                x="transaction_count",
                y="product",
                color="category",
                orientation="h",
                labels={
                    "transaction_count": "Transactions containing product",
                    "product": "Product",
                },
            ),
            width="stretch",
        )
    with category_column:
        st.subheader("Category distribution")
        st.plotly_chart(
            px.pie(
                categories,
                names="category",
                values="transaction_count",
                hole=0.48,
                title="Product appearances by category",
            ),
            width="stretch",
        )
    size_column, pair_column = st.columns(2)
    with size_column:
        st.subheader("Basket-size distribution")
        st.plotly_chart(
            px.bar(
                basket_sizes,
                x="items_per_basket",
                y="transactions",
                labels={
                    "items_per_basket": "Unique products per basket",
                    "transactions": "Transactions",
                },
            ),
            width="stretch",
        )
    with pair_column:
        st.subheader("Common product combinations")
        st.plotly_chart(
            px.bar(
                pairs.head(top_n).sort_values("transaction_count"),
                x="transaction_count",
                y="combination",
                orientation="h",
                labels={
                    "transaction_count": "Transactions containing both products",
                    "combination": "Product pair",
                },
            ),
            width="stretch",
        )


PAGES = {
    "Project overview": project_overview,
    "Shopping basket recommendations": shopping_recommendations,
    "Frequently bought together": frequently_bought_together,
    "Quantization comparison": quantization_comparison,
    "Dataset analytics": dataset_analytics,
}

st.sidebar.markdown("## 🛒 Retail Lab")
st.sidebar.caption("Next-item recommendations · INT8 evaluation")
selected_page = st.sidebar.radio("Navigate", list(PAGES))
PAGES[selected_page]()
