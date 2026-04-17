from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize

from src.config import FIGURES_DIR, ensure_directories


sns.set_theme(style="whitegrid")


def _to_numpy(values: Iterable[int] | np.ndarray) -> np.ndarray:
    return values if isinstance(values, np.ndarray) else np.asarray(list(values))


def _normalize_probabilities(probabilities: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack([1.0 - scores, scores])
    scores = np.clip(scores, 1e-12, None)
    row_sums = scores.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return scores / row_sums


def _binarize_labels(y_true: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    y_bin = label_binarize(y_true, classes=list(labels))
    if len(labels) == 2 and y_bin.shape[1] == 1:
        y_bin = np.column_stack([1 - y_bin[:, 0], y_bin[:, 0]])
    return y_bin


def multiclass_brier_score(y_true: Iterable[int], probabilities: np.ndarray, labels: Sequence[int]) -> float:
    y_true_array = _to_numpy(y_true)
    y_true_bin = _binarize_labels(y_true_array, labels)
    if y_true_bin.shape[1] != probabilities.shape[1]:
        raise ValueError("Probability matrix and label binarization shape do not match.")
    return float(np.mean(np.sum((probabilities - y_true_bin) ** 2, axis=1)))


def compute_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    average: str = "weighted",
    probabilities: np.ndarray | Iterable[Iterable[float]] | None = None,
    labels: Sequence[int] | None = None,
) -> dict[str, float]:
    y_true_array = _to_numpy(y_true)
    y_pred_array = _to_numpy(y_pred)
    resolved_labels = list(labels) if labels is not None else sorted(np.unique(np.concatenate([y_true_array, y_pred_array])).tolist())

    metrics = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred_array)),
        "precision": float(precision_score(y_true_array, y_pred_array, average=average, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true_array, y_pred_array, average=average, zero_division=0)),
        "precision_macro": float(precision_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true_array, y_pred_array)),
        "cohen_kappa": float(cohen_kappa_score(y_true_array, y_pred_array)),
    }

    if probabilities is None:
        return metrics

    probability_matrix = _normalize_probabilities(probabilities)
    metrics["mean_confidence"] = float(np.max(probability_matrix, axis=1).mean())

    try:
        metrics["top_2_accuracy"] = float(
            top_k_accuracy_score(
                y_true_array,
                probability_matrix,
                k=min(2, probability_matrix.shape[1]),
                labels=resolved_labels,
            )
        )
    except ValueError:
        pass

    try:
        metrics["log_loss"] = float(log_loss(y_true_array, probability_matrix, labels=resolved_labels))
    except ValueError:
        pass

    y_true_bin = _binarize_labels(y_true_array, resolved_labels)
    if y_true_bin.shape[1] != probability_matrix.shape[1]:
        return metrics

    try:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true_bin, probability_matrix, multi_class="ovr", average="macro")
        )
        metrics["roc_auc_ovr_weighted"] = float(
            roc_auc_score(y_true_bin, probability_matrix, multi_class="ovr", average="weighted")
        )
    except ValueError:
        pass

    try:
        metrics["average_precision_macro"] = float(average_precision_score(y_true_bin, probability_matrix, average="macro"))
        metrics["average_precision_weighted"] = float(
            average_precision_score(y_true_bin, probability_matrix, average="weighted")
        )
    except ValueError:
        pass

    try:
        metrics["brier_score"] = multiclass_brier_score(y_true_array, probability_matrix, resolved_labels)
    except ValueError:
        pass

    return metrics


def classification_report_dataframe(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    label_names: list[str],
) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0, output_dict=True)
    return pd.DataFrame(report).transpose()


def save_classification_report(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    label_names: list[str],
    output_path: str | Path,
) -> pd.DataFrame:
    report_df = classification_report_dataframe(y_true, y_pred, label_names)
    report_df.to_csv(output_path, index=True)
    return report_df


def save_prediction_details(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Sequence[int],
    label_names: list[str],
    output_path: str | Path,
    probabilities: np.ndarray | Iterable[Iterable[float]] | None = None,
) -> Path:
    y_true_array = _to_numpy(y_true)
    y_pred_array = _to_numpy(y_pred)
    label_lookup = dict(zip(labels, label_names))
    predictions_df = pd.DataFrame(
        {
            "actual_label_id": y_true_array,
            "actual_label": [label_lookup.get(label_id, str(label_id)) for label_id in y_true_array],
            "predicted_label_id": y_pred_array,
            "predicted_label": [label_lookup.get(label_id, str(label_id)) for label_id in y_pred_array],
            "correct": y_true_array == y_pred_array,
        }
    )

    if probabilities is not None:
        probability_matrix = _normalize_probabilities(probabilities)
        for idx, label_id in enumerate(labels):
            safe_label = label_lookup.get(label_id, str(label_id)).replace(" ", "_").lower()
            predictions_df[f"prob_{safe_label}"] = probability_matrix[:, idx]
        predictions_df["confidence"] = probability_matrix.max(axis=1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    return output_path


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: list[int],
    label_names: list[str],
    filename: str,
    normalize: bool = False,
    title: str | None = None,
) -> Path:
    ensure_directories()
    matrix = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    if normalize:
        row_totals = matrix.sum(axis=1, keepdims=True)
        row_totals[row_totals == 0] = 1.0
        matrix = matrix / row_totals
    output_path = FIGURES_DIR / filename
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else "g",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title or ("Normalized Confusion Matrix" if normalize else "Confusion Matrix"))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_model_comparison(results_df: pd.DataFrame, metric: str = "f1_score", filename: str = "model_comparison.png") -> Path:
    ensure_directories()
    output_path = FIGURES_DIR / filename
    if metric not in results_df.columns:
        plt.figure(figsize=(8, 3))
        plt.axis("off")
        plt.text(0.5, 0.5, f"{metric} is not available in this report.", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return output_path
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


def plot_metric_dashboard(
    results_df: pd.DataFrame,
    metrics: list[str] | None = None,
    filename: str = "model_metric_dashboard.png",
) -> Path:
    ensure_directories()
    metrics = metrics or ["accuracy", "f1_score", "f1_macro", "balanced_accuracy"]
    metrics = [metric for metric in metrics if metric in results_df.columns]
    output_path = FIGURES_DIR / filename

    if not metrics:
        plt.figure(figsize=(8, 3))
        plt.axis("off")
        plt.text(0.5, 0.5, "No compatible metrics were available for this dashboard.", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return output_path

    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
    axes_flat = list(axes.flat)

    for axis, metric in zip(axes_flat, metrics):
        ordered = results_df.sort_values(metric, ascending=False)
        sns.barplot(data=ordered, x="model", y=metric, hue="model", palette="crest", legend=False, ax=axis)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xlabel("Model")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=20)

    for axis in axes_flat[len(metrics):]:
        axis.axis("off")

    fig.suptitle("Model Metrics Dashboard", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_classification_report_bars(report_df: pd.DataFrame, label_names: list[str], filename: str) -> Path:
    ensure_directories()
    output_path = FIGURES_DIR / filename
    available_labels = [label for label in label_names if label in report_df.index]
    class_metrics = report_df.loc[available_labels, ["precision", "recall", "f1-score"]].reset_index(names="label")
    metric_frame = class_metrics.melt(id_vars="label", var_name="metric", value_name="score")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=metric_frame, x="label", y="score", hue="metric", palette="viridis")
    plt.ylim(0, 1.05)
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("Per-Class Precision, Recall, and F1")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_roc_curves(
    y_true: Iterable[int],
    probabilities: np.ndarray | Iterable[Iterable[float]],
    labels: Sequence[int],
    label_names: list[str],
    filename: str,
) -> Path | None:
    ensure_directories()
    y_true_array = _to_numpy(y_true)
    probability_matrix = _normalize_probabilities(probabilities)
    y_true_bin = _binarize_labels(y_true_array, labels)
    if y_true_bin.shape[1] != probability_matrix.shape[1]:
        return None

    output_path = FIGURES_DIR / filename
    plt.figure(figsize=(8, 6))
    plotted_any = False
    for idx, label_name in enumerate(label_names):
        if np.unique(y_true_bin[:, idx]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], probability_matrix[:, idx])
        auc_score = roc_auc_score(y_true_bin[:, idx], probability_matrix[:, idx])
        plt.plot(fpr, tpr, label=f"{label_name.title()} (AUC={auc_score:.3f})")
        plotted_any = True

    if not plotted_any:
        plt.close()
        return None

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_precision_recall_curves(
    y_true: Iterable[int],
    probabilities: np.ndarray | Iterable[Iterable[float]],
    labels: Sequence[int],
    label_names: list[str],
    filename: str,
) -> Path | None:
    ensure_directories()
    y_true_array = _to_numpy(y_true)
    probability_matrix = _normalize_probabilities(probabilities)
    y_true_bin = _binarize_labels(y_true_array, labels)
    if y_true_bin.shape[1] != probability_matrix.shape[1]:
        return None

    output_path = FIGURES_DIR / filename
    plt.figure(figsize=(8, 6))
    plotted_any = False
    for idx, label_name in enumerate(label_names):
        if np.unique(y_true_bin[:, idx]).size < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true_bin[:, idx], probability_matrix[:, idx])
        ap_score = average_precision_score(y_true_bin[:, idx], probability_matrix[:, idx])
        plt.plot(recall, precision, label=f"{label_name.title()} (AP={ap_score:.3f})")
        plotted_any = True

    if not plotted_any:
        plt.close()
        return None

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("One-vs-Rest Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_prediction_confidence(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    probabilities: np.ndarray | Iterable[Iterable[float]],
    filename: str,
) -> Path:
    ensure_directories()
    y_true_array = _to_numpy(y_true)
    y_pred_array = _to_numpy(y_pred)
    probability_matrix = _normalize_probabilities(probabilities)
    confidence = probability_matrix.max(axis=1)
    correctness = y_true_array == y_pred_array

    confidence_df = pd.DataFrame({"confidence": confidence, "correct": correctness})
    bins = np.linspace(0, 1, 11)
    confidence_df["bucket"] = pd.cut(confidence_df["confidence"], bins=bins, include_lowest=True)
    bucket_summary = (
        confidence_df.groupby("bucket", observed=False)
        .agg(accuracy=("correct", "mean"), mean_confidence=("confidence", "mean"))
        .dropna()
    )

    output_path = FIGURES_DIR / filename
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(
        [confidence_df.loc[confidence_df["correct"], "confidence"], confidence_df.loc[~confidence_df["correct"], "confidence"]],
        bins=bins,
        stacked=True,
        color=["#15803d", "#b91c1c"],
        label=["Correct", "Incorrect"],
    )
    axes[0].set_title("Prediction Confidence Distribution")
    axes[0].set_xlabel("Max Predicted Probability")
    axes[0].set_ylabel("Samples")
    axes[0].legend()

    axes[1].plot(bucket_summary["mean_confidence"], bucket_summary["accuracy"], marker="o", label="Observed accuracy")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal calibration")
    axes[1].set_title("Confidence vs Accuracy")
    axes[1].set_xlabel("Mean confidence")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
