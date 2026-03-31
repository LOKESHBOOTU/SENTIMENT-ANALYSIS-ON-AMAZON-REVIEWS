from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import MODELS_DIR, REPORTS_DIR
from src.inference import MLInferencePipeline, RobertaInferencePipeline
from src.utils import load_json


st.set_page_config(page_title="Product Review Sentiment Analysis", page_icon="🛍️", layout="centered")

ML_MODELS = {
    "Logistic Regression": MODELS_DIR / "ml" / "logistic_regression" / "pipeline.joblib",
    "Naive Bayes": MODELS_DIR / "ml" / "naive_bayes" / "pipeline.joblib",
    "SVM": MODELS_DIR / "ml" / "svm" / "pipeline.joblib",
}
ROBERTA_SUMMARY_PATH = REPORTS_DIR / "roberta_summary.json"
BENCHMARK_TABLE_PATH = REPORTS_DIR / "all_model_comparison.csv"


@st.cache_resource
def load_ml_predictor(model_name: str) -> MLInferencePipeline:
    model_path = ML_MODELS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"{model_name} artifact not found at {model_path}.")
    return MLInferencePipeline(model_path)


@st.cache_resource
def load_roberta_predictor() -> RobertaInferencePipeline:
    if not ROBERTA_SUMMARY_PATH.exists():
        raise FileNotFoundError("No RoBERTa summary found. Train or add the transformer artifact first.")
    best_model_dir = load_json(ROBERTA_SUMMARY_PATH)["best_model_dir"]
    return RobertaInferencePipeline(best_model_dir)


def get_available_ml_models() -> list[str]:
    return [model_name for model_name, model_path in ML_MODELS.items() if model_path.exists()]


def roberta_available() -> bool:
    return ROBERTA_SUMMARY_PATH.exists()


def predict_with_available_models(review_text: str) -> tuple[dict | None, list[dict]]:
    primary_prediction = None
    comparison_rows: list[dict] = []

    if roberta_available():
        roberta_prediction = load_roberta_predictor().predict(review_text)
        primary_prediction = {"model": "RoBERTa", **roberta_prediction}
        comparison_rows.append({"Model": "RoBERTa", "Sentiment": roberta_prediction["label"].title(), "Confidence": roberta_prediction["confidence"]})

    for model_name in get_available_ml_models():
        prediction = load_ml_predictor(model_name).predict(review_text)
        comparison_rows.append({"Model": model_name, "Sentiment": prediction["label"].title(), "Confidence": prediction["confidence"]})
        if primary_prediction is None:
            primary_prediction = {"model": model_name, **prediction}

    return primary_prediction, comparison_rows


def load_benchmark_table() -> pd.DataFrame | None:
    if BENCHMARK_TABLE_PATH.exists():
        benchmark_df = pd.read_csv(BENCHMARK_TABLE_PATH)
        benchmark_df["accuracy"] = benchmark_df["accuracy"].map(lambda value: f"{value:.4f}")
        benchmark_df["f1_score"] = benchmark_df["f1_score"].map(lambda value: f"{value:.4f}")
        return benchmark_df[["model", "accuracy", "f1_score"]].rename(
            columns={"model": "Model", "accuracy": "Accuracy", "f1_score": "Weighted F1"}
        )
    return None


def main() -> None:
    st.title("Sentiment Analysis on Product Reviews")
    st.caption("RoBERTa is used as the main predictor when available, with ML baselines shown for side-by-side comparison.")

    available_ml_models = get_available_ml_models()
    if not available_ml_models and not roberta_available():
        st.error("No saved model artifacts were found. Add trained models under the models directory before launching the app.")
        return

    if not roberta_available():
        st.info("RoBERTa is not available in this deployment, so the app is using the saved ML demo models for live inference.")

    review_text = st.text_area(
        "Enter a product review",
        height=180,
        placeholder="This product arrived on time, feels premium, and performs exactly as advertised.",
    )

    if st.button("Predict sentiment", type="primary"):
        if not review_text.strip():
            st.warning("Please enter review text before running a prediction.")
            return

        primary_prediction, comparison_rows = predict_with_available_models(review_text)
        if primary_prediction is None:
            st.error("No inference model could be loaded.")
            return

        st.subheader(f"Primary prediction: {primary_prediction['model']}")
        st.success(f"Predicted sentiment: {primary_prediction['label'].title()}")
        st.metric("Confidence", f"{primary_prediction['confidence'] * 100:.2f}%")

        st.subheader("Primary model class probabilities")
        st.json({label.title(): f"{score:.4f}" for label, score in primary_prediction["probabilities"].items()})

        st.subheader("Live model comparison")
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df["Confidence"] = comparison_df["Confidence"].map(lambda value: f"{value * 100:.2f}%")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    benchmark_df = load_benchmark_table()
    if benchmark_df is not None:
        st.subheader("Saved benchmark comparison")
        st.caption("Held-out evaluation scores from the training pipeline.")
        st.dataframe(benchmark_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
