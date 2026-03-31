from __future__ import annotations

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


def render_probability_bars(probabilities: dict[str, float]) -> None:
    sentiment_order = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    sentiment_styles = {
        "positive": ("Positive", "#15803d"),
        "neutral": ("Neutral", "#a16207"),
        "negative": ("Negative", "#b91c1c"),
    }

    for sentiment, score in sentiment_order:
        label, color = sentiment_styles.get(sentiment, (sentiment.title(), "#0f172a"))
        st.markdown(
            f"""
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem;">
                <span style="font-weight:700;color:{color};">{label}</span>
                <span style="font-weight:600;">{score * 100:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(float(score))


def render_model_comparison(comparison_rows: list[dict]) -> None:
    if not comparison_rows:
        return

    comparison_rows = sorted(comparison_rows, key=lambda row: row["Confidence"], reverse=True)
    top_confidence = comparison_rows[0]["Confidence"]

    sentiment_colors = {
        "Positive": "#15803d",
        "Neutral": "#a16207",
        "Negative": "#b91c1c",
    }

    for row in comparison_rows:
        is_winner = row["Confidence"] == top_confidence
        accent = "#0f766e" if is_winner else "#cbd5e1"
        background = "#ecfdf5" if is_winner else "#f8fafc"
        border = "2px solid #0f766e" if is_winner else "1px solid #e2e8f0"
        sentiment_color = sentiment_colors.get(row["Sentiment"], "#0f172a")
        winner_badge = (
            "<div style='margin-top:0.35rem;'>"
            "<span style='background:#0f766e;color:white;padding:0.18rem 0.55rem;border-radius:999px;font-size:0.8rem;font-weight:700;'>Top confidence</span>"
            "</div>"
            if is_winner
            else ""
        )

        st.markdown(
            (
                f'<div style="border:{border}; background:{background}; border-left:8px solid {accent}; '
                f'border-radius:14px; padding:0.9rem 1rem; margin-bottom:0.8rem;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;">'
                f'<div>'
                f'<div style="font-size:1.05rem; font-weight:800; color:#102a43;">{row["Model"]}</div>'
                f'<div style="margin-top:0.25rem; font-weight:700; color:{sentiment_color};">{row["Sentiment"]}</div>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<div style="font-size:1.2rem; font-weight:800; color:#102a43;">{row["Confidence"] * 100:.2f}%</div>'
                f'{winner_badge}'
                f'</div>'
                f'</div>'
                f'</div>'
            ),
            unsafe_allow_html=True,
        )


def main() -> None:
    st.title("Sentiment Analysis on Product Reviews")
    st.caption("RoBERTa is used as the main predictor when available, with ML baselines shown for side-by-side comparison.")

    if st.session_state.get("clear_review_text", False):
        st.session_state.review_text = ""
        st.session_state.clear_review_text = False

    if "review_text" not in st.session_state:
        st.session_state.review_text = ""

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
        key="review_text",
    )

    action_col, clear_col = st.columns([3, 1])
    predict_clicked = action_col.button("Predict sentiment", type="primary")
    clear_clicked = clear_col.button("Clear")

    if clear_clicked:
        st.session_state.clear_review_text = True
        st.rerun()

    if predict_clicked:
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
        render_probability_bars(primary_prediction["probabilities"])

        st.subheader("Live model comparison")
        render_model_comparison(comparison_rows)


if __name__ == "__main__":
    main()
