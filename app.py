from __future__ import annotations

import streamlit as st

from src.config import MODELS_DIR
from src.inference import MLInferencePipeline, RobertaInferencePipeline
from src.utils import load_json


st.set_page_config(page_title="Product Review Sentiment Analysis", page_icon="🛍️", layout="centered")


@st.cache_resource
def load_inference_model(model_type: str):
    if model_type == "RoBERTa":
        summary_path = MODELS_DIR.parent / "outputs" / "reports" / "roberta_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError("No RoBERTa summary found. Train the transformer model first.")
        best_model_dir = load_json(summary_path)["best_model_dir"]
        return RobertaInferencePipeline(best_model_dir)

    model_path = MODELS_DIR / "ml" / "logistic_regression" / "pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError("No ML pipeline found. Train the traditional models first.")
    return MLInferencePipeline(model_path)


def main() -> None:
    st.title("Sentiment Analysis on Product Reviews")
    st.caption("Interactive inference with traditional ML and transformer sentiment models.")

    model_type = st.selectbox("Inference model", ["RoBERTa", "Logistic Regression"])
    review_text = st.text_area(
        "Enter a product review",
        height=180,
        placeholder="This product arrived on time, feels premium, and performs exactly as advertised.",
    )

    if st.button("Predict sentiment", type="primary"):
        if not review_text.strip():
            st.warning("Please enter review text before running a prediction.")
            return

        try:
            predictor = load_inference_model(model_type)
            prediction = predictor.predict(review_text)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return

        st.success(f"Predicted sentiment: {prediction['label'].title()}")
        st.metric("Confidence", f"{prediction['confidence'] * 100:.2f}%")
        st.subheader("Class probabilities")
        st.json({label.title(): f"{score:.4f}" for label, score in prediction["probabilities"].items()})


if __name__ == "__main__":
    main()
