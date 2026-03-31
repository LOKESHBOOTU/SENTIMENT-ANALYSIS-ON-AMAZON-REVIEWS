from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import FIGURES_DIR, ensure_directories


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int], average: str = "weighted") -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def save_classification_report(y_true: Iterable[int], y_pred: Iterable[int], label_names: list[str], output_path: str | Path) -> None:
    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(output_path, index=True)


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: list[int],
    label_names: list[str],
    filename: str,
) -> Path:
    ensure_directories()
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    output_path = FIGURES_DIR / filename
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_model_comparison(results_df: pd.DataFrame, metric: str = "f1_score", filename: str = "model_comparison.png") -> Path:
    ensure_directories()
    output_path = FIGURES_DIR / filename
    ordered = results_df.sort_values(metric, ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=ordered, x="model", y=metric, hue="model", palette="crest", legend=False)
    plt.xticks(rotation=20, ha="right")
    plt.title(f"Model Comparison by {metric.replace('_', ' ').title()}")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
