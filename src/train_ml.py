from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from src.config import ID_TO_LABEL, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, SEED, ensure_directories, set_seed
from src.evaluate import (
    compute_metrics,
    plot_classification_report_bars,
    plot_confusion_matrix,
    plot_metric_dashboard,
    plot_model_comparison,
    plot_precision_recall_curves,
    plot_prediction_confidence,
    plot_roc_curves,
    save_classification_report,
    save_prediction_details,
)
from src.utils import load_json, save_json


MODEL_SPECS = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED),
        "param_grid": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [2, 5],
            "clf__C": [0.5, 1.0, 2.0],
        },
    },
    "naive_bayes": {
        "estimator": ComplementNB(),
        "param_grid": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [2, 5],
            "clf__alpha": [0.5, 1.0, 1.5],
        },
    },
    "svm": {
        "estimator": LinearSVC(class_weight="balanced", random_state=SEED),
        "param_grid": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [2, 5],
            "clf__C": [0.5, 1.0, 2.0],
        },
    },
}


def scores_to_probabilities(scores: np.ndarray) -> np.ndarray:
    score_matrix = np.asarray(scores, dtype=float)
    if score_matrix.ndim == 1:
        score_matrix = np.column_stack([-score_matrix, score_matrix])
    score_matrix = score_matrix - score_matrix.max(axis=1, keepdims=True)
    exp_scores = np.exp(score_matrix)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def get_prediction_probabilities(model, texts: pd.Series, ordered_labels: list[int]) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(texts), dtype=float)
    elif hasattr(model, "decision_function"):
        probabilities = scores_to_probabilities(model.decision_function(texts))
    else:
        return None

    model_labels = list(getattr(model, "classes_", ordered_labels))
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])
    if model_labels != ordered_labels:
        label_to_index = {label: idx for idx, label in enumerate(model_labels)}
        probabilities = probabilities[:, [label_to_index[label] for label in ordered_labels]]
    return probabilities


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    validation_df = pd.read_csv(PROCESSED_DIR / "validation.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    for frame in [train_df, validation_df, test_df]:
        frame["clean_text"] = frame["clean_text"].fillna("")
    return train_df, validation_df, test_df


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", max_features=40000)),
            ("clf", estimator),
        ]
    )


def run_grid_search(model_name: str, train_df: pd.DataFrame, validation_df: pd.DataFrame, n_jobs: int) -> GridSearchCV:
    spec = MODEL_SPECS[model_name]
    X_combined = pd.concat([train_df["clean_text"], validation_df["clean_text"]], axis=0).reset_index(drop=True)
    y_combined = pd.concat([train_df["label_id"], validation_df["label_id"]], axis=0).reset_index(drop=True)
    test_fold = [-1] * len(train_df) + [0] * len(validation_df)
    split = PredefinedSplit(test_fold=test_fold)

    search = GridSearchCV(
        estimator=build_pipeline(spec["estimator"]),
        param_grid=spec["param_grid"],
        scoring="f1_weighted",
        cv=split,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )
    search.fit(X_combined, y_combined)
    return search


def evaluate_and_save(model_name: str, model, test_df: pd.DataFrame) -> dict[str, float | str]:
    y_true = test_df["label_id"]
    y_pred = model.predict(test_df["clean_text"])
    ordered_labels = sorted(ID_TO_LABEL)
    label_names = [ID_TO_LABEL[idx] for idx in ordered_labels]
    probabilities = get_prediction_probabilities(model, test_df["clean_text"], ordered_labels)
    metrics = compute_metrics(y_true, y_pred, probabilities=probabilities, labels=ordered_labels)

    model_dir = MODELS_DIR / "ml" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "pipeline.joblib")
    save_json(metrics, model_dir / "metrics.json")

    report_df = save_classification_report(
        y_true,
        y_pred,
        label_names=label_names,
        output_path=REPORTS_DIR / f"{model_name}_classification_report.csv",
    )
    save_prediction_details(
        y_true,
        y_pred,
        labels=ordered_labels,
        label_names=label_names,
        probabilities=probabilities,
        output_path=REPORTS_DIR / f"{model_name}_predictions.csv",
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        labels=ordered_labels,
        label_names=label_names,
        filename=f"{model_name}_confusion_matrix.png",
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        labels=ordered_labels,
        label_names=label_names,
        filename=f"{model_name}_normalized_confusion_matrix.png",
        normalize=True,
    )
    plot_classification_report_bars(report_df, label_names=label_names, filename=f"{model_name}_per_class_metrics.png")
    if probabilities is not None:
        plot_roc_curves(
            y_true,
            probabilities=probabilities,
            labels=ordered_labels,
            label_names=label_names,
            filename=f"{model_name}_roc_curves.png",
        )
        plot_precision_recall_curves(
            y_true,
            probabilities=probabilities,
            labels=ordered_labels,
            label_names=label_names,
            filename=f"{model_name}_precision_recall_curves.png",
        )
        plot_prediction_confidence(
            y_true,
            y_pred,
            probabilities=probabilities,
            filename=f"{model_name}_confidence_analysis.png",
        )
    return {"model": model_name, **metrics}


def load_saved_result(model_name: str) -> dict[str, float | str] | None:
    model_dir = MODELS_DIR / "ml" / model_name
    model_path = model_dir / "pipeline.joblib"
    metrics_path = model_dir / "metrics.json"
    if not model_path.exists() or not metrics_path.exists():
        return None
    metrics = load_json(metrics_path)
    return {"model": model_name, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train traditional ML baselines for sentiment analysis.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS.keys()), choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for GridSearchCV. Use 1 for restricted Windows environments.")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain models even if saved artifacts already exist.")
    return parser.parse_args()


def main() -> None:
    set_seed(SEED)
    ensure_directories()
    args = parse_args()
    train_df, validation_df, test_df = load_splits()

    all_results: list[dict[str, float | str]] = []
    best_summary: dict[str, dict] = {}

    for model_name in args.models:
        if not args.force_retrain:
            saved_result = load_saved_result(model_name)
            if saved_result is not None:
                print(f"Reusing saved ML artifact for {model_name} from {MODELS_DIR / 'ml' / model_name}")
                all_results.append(saved_result)
                continue

        search = run_grid_search(model_name, train_df, validation_df, n_jobs=args.n_jobs)
        best_summary[model_name] = {
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
        }
        all_results.append(evaluate_and_save(model_name, search.best_estimator_, test_df))

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(REPORTS_DIR / "ml_model_comparison.csv", index=False)
    plot_model_comparison(results_df, metric="f1_score", filename="ml_model_comparison.png")
    plot_model_comparison(results_df, metric="accuracy", filename="ml_model_accuracy_comparison.png")
    plot_model_comparison(results_df, metric="f1_macro", filename="ml_model_macro_f1_comparison.png")
    plot_metric_dashboard(results_df, filename="ml_metric_dashboard.png")
    existing_summary_path = REPORTS_DIR / "ml_grid_search_summary.json"
    existing_summary = load_json(existing_summary_path) if existing_summary_path.exists() else {}
    existing_summary.update(best_summary)
    save_json(existing_summary, existing_summary_path)
    print(results_df.sort_values("f1_score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
