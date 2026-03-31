from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import ID_TO_LABEL
from src.preprocessing import clean_text


class MLInferencePipeline:
    def __init__(self, model_path: str | Path):
        self.model = joblib.load(model_path)

    def predict(self, text: str) -> dict[str, Any]:
        cleaned = clean_text(text)
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([cleaned])[0]
        else:
            decision = self.model.decision_function([cleaned])
            decision = np.atleast_2d(decision)[0]
            exp = np.exp(decision - np.max(decision))
            probabilities = exp / exp.sum()

        label_id = int(np.argmax(probabilities))
        return {
            "label": ID_TO_LABEL[label_id],
            "confidence": float(np.max(probabilities)),
            "probabilities": {ID_TO_LABEL[idx]: float(prob) for idx, prob in enumerate(probabilities)},
        }


class RobertaInferencePipeline:
    def __init__(self, model_dir: str | Path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> dict[str, Any]:
        encoded = self.tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        label_id = int(np.argmax(probabilities))
        return {
            "label": ID_TO_LABEL[label_id],
            "confidence": float(np.max(probabilities)),
            "probabilities": {ID_TO_LABEL[idx]: float(prob) for idx, prob in enumerate(probabilities)},
        }
