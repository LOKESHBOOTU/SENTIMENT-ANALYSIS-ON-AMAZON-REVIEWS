from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.config import ID_TO_LABEL, LABEL_TO_ID, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, SEED, ensure_directories, set_seed
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


MODEL_NAME = "roberta-base"
TRAIN_METRICS_SAMPLE_SIZE = 4000
sns.set_theme(style="whitegrid")


@dataclass
class TrialConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    weight_decay: float
    gradient_accumulation_steps: int


@dataclass(frozen=True)
class HardwareProfile:
    has_gpu: bool
    gpu_name: str | None
    cpu_logical: int
    ram_total_gb: float
    ram_available_gb: float


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, label_smoothing_factor: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing_factor = label_smoothing_factor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
            weight=self.class_weights.to(logits.device),
            label_smoothing=self.label_smoothing_factor,
        )
        return (loss, outputs) if return_outputs else loss


class TrainMetricsCallback(TrainerCallback):
    def __init__(self, train_dataset: Dataset):
        self.train_dataset = train_dataset
        self.trainer: Trainer | None = None

    def attach(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control
        train_metrics = self.trainer.evaluate(eval_dataset=self.train_dataset, metric_key_prefix="train")
        train_metrics["epoch"] = state.epoch
        self.trainer.log(train_metrics)
        return control


def detect_hardware_profile() -> HardwareProfile:
    memory = psutil.virtual_memory()
    has_gpu = torch.cuda.is_available()
    return HardwareProfile(
        has_gpu=has_gpu,
        gpu_name=torch.cuda.get_device_name(0) if has_gpu else None,
        cpu_logical=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=round(memory.total / (1024**3), 2),
        ram_available_gb=round(memory.available / (1024**3), 2),
    )


def choose_default_search_space(profile: HardwareProfile, prefer_colab_t4: bool = False) -> dict[str, list | int | float | str | bool]:
    using_gpu_profile = profile.has_gpu or prefer_colab_t4
    gpu_name = (profile.gpu_name or "").lower()
    t4_like_profile = prefer_colab_t4 or "t4" in gpu_name

    if using_gpu_profile:
        return {
            "max_length": 256 if t4_like_profile else 192,
            "learning_rates": [1.5e-5, 2e-5],
            "batch_sizes": [16],
            "epochs": [3, 4],
            "weight_decays": [0.01],
            "gradient_accumulation_steps": [1, 2] if t4_like_profile else [1],
            "dataloader_num_workers": min(2, max(1, profile.cpu_logical // 4)),
            "label_smoothing_factor": 0.05,
            "scheduler_type": "cosine",
            "warmup_ratio": 0.06,
            "selection_metric": "accuracy",
            "gradient_checkpointing": True,
        }

    if profile.ram_available_gb < 5:
        return {
            "max_length": 128,
            "learning_rates": [2e-5],
            "batch_sizes": [4],
            "epochs": [2],
            "weight_decays": [0.01],
            "gradient_accumulation_steps": [4],
            "dataloader_num_workers": 0,
            "label_smoothing_factor": 0.0,
            "scheduler_type": "linear",
            "warmup_ratio": 0.1,
            "selection_metric": "accuracy",
            "gradient_checkpointing": False,
        }

    return {
        "max_length": 128,
        "learning_rates": [2e-5],
        "batch_sizes": [8],
        "epochs": [3],
        "weight_decays": [0.01],
        "gradient_accumulation_steps": [2],
        "dataloader_num_workers": max(1, min(2, profile.cpu_logical // 6)),
        "label_smoothing_factor": 0.03,
        "scheduler_type": "linear",
        "warmup_ratio": 0.1,
        "selection_metric": "accuracy",
        "gradient_checkpointing": False,
    }


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(PROCESSED_DIR / "train.csv"),
        pd.read_csv(PROCESSED_DIR / "validation.csv"),
        pd.read_csv(PROCESSED_DIR / "test.csv"),
    )


def sample_split(df: pd.DataFrame, max_samples: int | None) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df
    fractions = df["label_id"].value_counts(normalize=True).sort_index()
    samples = []
    assigned = 0
    for idx, (label_id, fraction) in enumerate(fractions.items()):
        remaining_classes = len(fractions) - idx
        remaining_budget = max_samples - assigned
        target = max(1, int(round(max_samples * fraction))) if remaining_classes > 1 else remaining_budget
        group = df[df["label_id"] == label_id]
        sampled_group = group.sample(min(target, len(group)), random_state=SEED)
        samples.append(sampled_group)
        assigned += len(sampled_group)
    return pd.concat(samples, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def build_training_arguments(
    trial_output_dir: Path,
    trial: TrialConfig,
    dataloader_num_workers: int,
    selection_metric: str,
    scheduler_type: str,
    warmup_ratio: float,
    gradient_checkpointing: bool,
) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    kwargs = {
        "output_dir": str(trial_output_dir / "checkpoints"),
        "learning_rate": trial.learning_rate,
        "per_device_train_batch_size": trial.batch_size,
        "per_device_eval_batch_size": trial.batch_size,
        "num_train_epochs": trial.epochs,
        "gradient_accumulation_steps": trial.gradient_accumulation_steps,
        "weight_decay": trial.weight_decay,
        "warmup_ratio": warmup_ratio,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": selection_metric,
        "greater_is_better": True,
        "save_total_limit": 2,
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "seed": SEED,
        "fp16": torch.cuda.is_available(),
        "report_to": "none",
    }
    optional_kwargs = {
        "lr_scheduler_type": scheduler_type,
        "optim": "adamw_torch",
        "group_by_length": True,
        "gradient_checkpointing": gradient_checkpointing and torch.cuda.is_available(),
        "dataloader_persistent_workers": dataloader_num_workers > 0,
    }
    for key, value in optional_kwargs.items():
        if key in signature.parameters:
            kwargs[key] = value

    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def build_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(
        df[["review_text", "label_id"]].rename(columns={"label_id": "labels"}),
        preserve_index=False,
    )


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    return dataset.map(
        lambda batch: tokenizer(batch["review_text"], truncation=True, padding=False, max_length=max_length),
        batched=True,
        remove_columns=["review_text"],
    )


def build_metrics_sample(dataset: Dataset, max_samples: int = TRAIN_METRICS_SAMPLE_SIZE) -> Dataset:
    if len(dataset) <= max_samples:
        return dataset
    return dataset.shuffle(seed=SEED).select(range(max_samples))


def hf_compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
    preds = np.argmax(prediction.predictions, axis=1)
    probability_matrix = logits_to_probabilities(prediction.predictions)
    return compute_metrics(prediction.label_ids, preds, probabilities=probability_matrix, labels=sorted(ID_TO_LABEL))


def get_class_weights(labels: pd.Series) -> torch.Tensor:
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(sorted(LABEL_TO_ID.values())),
        y=labels.to_numpy(),
    )
    return torch.tensor(class_weights, dtype=torch.float)


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def save_training_curves(log_history: list[dict], output_dir: Path) -> None:
    history_df = pd.DataFrame(log_history)
    history_df.to_csv(output_dir / "trainer_log_history.csv", index=False)

    if history_df.empty:
        return

    tracked_columns = [
        "epoch",
        "loss",
        "eval_loss",
        "eval_accuracy",
        "eval_f1_score",
        "eval_f1_macro",
        "train_loss",
        "train_accuracy",
        "train_f1_score",
        "train_f1_macro",
        "learning_rate",
    ]
    available_columns = [column for column in tracked_columns if column in history_df.columns]
    curve_df = history_df[available_columns].copy()
    non_epoch_columns = [column for column in available_columns if column != "epoch"]
    if non_epoch_columns:
        curve_df = curve_df.dropna(how="all", subset=non_epoch_columns)
    curve_df.to_csv(output_dir / "training_curves.csv", index=False)

    if "epoch" not in curve_df.columns:
        return

    epoch_metrics = curve_df.dropna(subset=["epoch"]).groupby("epoch", as_index=False).last()
    epoch_metrics.to_csv(output_dir / "epoch_metrics.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_specs = [
        ("train_loss", "eval_loss", "Training vs Validation Loss", "Loss"),
        ("train_accuracy", "eval_accuracy", "Training vs Validation Accuracy", "Accuracy"),
        ("train_f1_score", "eval_f1_score", "Training vs Validation Weighted F1", "Weighted F1"),
        ("train_f1_macro", "eval_f1_macro", "Training vs Validation Macro F1", "Macro F1"),
    ]
    for axis, (train_col, val_col, title, ylabel) in zip(axes.flat, plot_specs):
        plotted = False
        if train_col in epoch_metrics.columns:
            sns.lineplot(data=epoch_metrics, x="epoch", y=train_col, marker="o", label=f"Train {ylabel}", ax=axis)
            plotted = True
        if val_col in epoch_metrics.columns:
            sns.lineplot(data=epoch_metrics, x="epoch", y=val_col, marker="o", label=f"Validation {ylabel}", ax=axis)
            plotted = True
        if not plotted:
            axis.axis("off")
            continue
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)

    figure.tight_layout()
    figure.savefig(output_dir / "training_validation_curves.png", dpi=200)
    plt.close(figure)

    if "learning_rate" in epoch_metrics.columns:
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=epoch_metrics, x="epoch", y="learning_rate", marker="o")
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.tight_layout()
        plt.savefig(output_dir / "learning_rate_schedule.png", dpi=200)
        plt.close()


def trial_output_path(trial: TrialConfig) -> Path:
    return MODELS_DIR / "roberta" / (
        f"lr_{trial.learning_rate}_bs_{trial.batch_size}_ep_{trial.epochs}"
        f"_wd_{trial.weight_decay}_ga_{trial.gradient_accumulation_steps}"
    )


def load_saved_trial_result(trial: TrialConfig) -> dict[str, float | str] | None:
    output_dir = trial_output_path(trial)
    validation_metrics_path = output_dir / "validation_metrics.json"
    model_path = output_dir / "model.safetensors"
    if not output_dir.exists() or not validation_metrics_path.exists() or not model_path.exists():
        return None
    metrics = load_json(validation_metrics_path)
    metrics["model_dir"] = str(output_dir)
    return metrics


def run_trial(
    trial: TrialConfig,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    class_weights: torch.Tensor,
    max_length: int,
    early_stopping_patience: int,
    dataloader_num_workers: int,
    label_smoothing_factor: float,
    selection_metric: str,
    scheduler_type: str,
    warmup_ratio: float,
    gradient_checkpointing: bool,
) -> tuple[dict[str, float], Path]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_tokenized = tokenize_dataset(train_dataset, tokenizer, max_length=max_length)
    validation_tokenized = tokenize_dataset(validation_dataset, tokenizer, max_length=max_length)
    train_metrics_dataset = build_metrics_sample(train_tokenized)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    if gradient_checkpointing and torch.cuda.is_available() and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    trial_output_dir = trial_output_path(trial)
    training_args = build_training_arguments(
        trial_output_dir,
        trial,
        dataloader_num_workers=dataloader_num_workers,
        selection_metric=selection_metric,
        scheduler_type=scheduler_type,
        warmup_ratio=warmup_ratio,
        gradient_checkpointing=gradient_checkpointing,
    )
    train_metrics_callback = TrainMetricsCallback(train_metrics_dataset)

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        label_smoothing_factor=label_smoothing_factor,
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=validation_tokenized,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=hf_compute_metrics,
        callbacks=[train_metrics_callback, EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    train_metrics_callback.attach(trainer)
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(trial_output_dir))
    tokenizer.save_pretrained(str(trial_output_dir))
    save_training_curves(trainer.state.log_history, trial_output_dir)
    save_json(
        {
            key: float(value) if isinstance(value, (np.floating, float)) else value
            for key, value in metrics.items()
        },
        trial_output_dir / "validation_metrics.json",
    )
    return metrics, trial_output_dir


def evaluate_best_model(model_dir: Path, test_df: pd.DataFrame, max_length: int) -> dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    test_dataset = tokenize_dataset(build_dataset(test_df), tokenizer, max_length=max_length)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(model_dir / "eval"),
            report_to="none",
            per_device_eval_batch_size=32 if torch.cuda.is_available() else 16,
        ),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=hf_compute_metrics,
    )
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    probabilities = logits_to_probabilities(predictions.predictions)

    ordered_labels = sorted(ID_TO_LABEL)
    label_names = [ID_TO_LABEL[idx] for idx in ordered_labels]
    metrics = compute_metrics(test_df["label_id"], y_pred, probabilities=probabilities, labels=ordered_labels)

    report_df = save_classification_report(
        test_df["label_id"],
        y_pred,
        label_names=label_names,
        output_path=REPORTS_DIR / "roberta_classification_report.csv",
    )
    save_prediction_details(
        test_df["label_id"],
        y_pred,
        labels=ordered_labels,
        label_names=label_names,
        probabilities=probabilities,
        output_path=REPORTS_DIR / "roberta_test_predictions.csv",
    )
    plot_confusion_matrix(
        test_df["label_id"],
        y_pred,
        labels=ordered_labels,
        label_names=label_names,
        filename="roberta_confusion_matrix.png",
    )
    plot_confusion_matrix(
        test_df["label_id"],
        y_pred,
        labels=ordered_labels,
        label_names=label_names,
        filename="roberta_normalized_confusion_matrix.png",
        normalize=True,
    )
    plot_classification_report_bars(report_df, label_names=label_names, filename="roberta_per_class_metrics.png")
    plot_roc_curves(
        test_df["label_id"],
        probabilities=probabilities,
        labels=ordered_labels,
        label_names=label_names,
        filename="roberta_roc_curves.png",
    )
    plot_precision_recall_curves(
        test_df["label_id"],
        probabilities=probabilities,
        labels=ordered_labels,
        label_names=label_names,
        filename="roberta_precision_recall_curves.png",
    )
    plot_prediction_confidence(
        test_df["label_id"],
        y_pred,
        probabilities=probabilities,
        filename="roberta_confidence_analysis.png",
    )
    save_json(metrics, model_dir / "test_metrics.json")
    return metrics


def load_saved_test_metrics(model_dir: Path) -> dict[str, float] | None:
    metrics_path = model_dir / "test_metrics.json"
    if not metrics_path.exists():
        return None
    return load_json(metrics_path)


def resolve_eval_metric_key(metric_name: str) -> str:
    return metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for sentiment classification.")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=None)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", nargs="+", type=int, default=None)
    parser.add_argument("--weight-decays", nargs="+", type=float, default=None)
    parser.add_argument("--gradient-accumulation-steps", nargs="+", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--selection-metric", choices=["accuracy", "balanced_accuracy", "f1_score", "f1_macro"], default=None)
    parser.add_argument("--scheduler-type", choices=["linear", "cosine", "cosine_with_restarts", "polynomial"], default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--label-smoothing-factor", type=float, default=None)
    parser.add_argument("--colab-t4-profile", action="store_true", help="Use T4-oriented defaults for Google Colab.")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=None)
    parser.add_argument("--force-retrain", action="store_true", help="Retrain trials even if saved artifacts already exist.")
    return parser.parse_args()


def main() -> None:
    set_seed(SEED)
    ensure_directories()
    args = parse_args()
    hardware_profile = detect_hardware_profile()
    defaults = choose_default_search_space(hardware_profile, prefer_colab_t4=args.colab_t4_profile)

    if args.max_length is None:
        args.max_length = defaults["max_length"]
    if args.learning_rates is None:
        args.learning_rates = defaults["learning_rates"]
    if args.batch_sizes is None:
        args.batch_sizes = defaults["batch_sizes"]
    if args.epochs is None:
        args.epochs = defaults["epochs"]
    if args.weight_decays is None:
        args.weight_decays = defaults["weight_decays"]
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = defaults["gradient_accumulation_steps"]
    if args.dataloader_num_workers is None:
        args.dataloader_num_workers = defaults["dataloader_num_workers"]
    if args.selection_metric is None:
        args.selection_metric = defaults["selection_metric"]
    if args.scheduler_type is None:
        args.scheduler_type = defaults["scheduler_type"]
    if args.warmup_ratio is None:
        args.warmup_ratio = defaults["warmup_ratio"]
    if args.label_smoothing_factor is None:
        args.label_smoothing_factor = defaults["label_smoothing_factor"]
    if args.gradient_checkpointing is None:
        args.gradient_checkpointing = defaults["gradient_checkpointing"]

    print(
        json.dumps(
            {
                "hardware_profile": asdict(hardware_profile),
                "resolved_defaults": {
                    "colab_t4_profile": args.colab_t4_profile,
                    "max_length": args.max_length,
                    "learning_rates": args.learning_rates,
                    "batch_sizes": args.batch_sizes,
                    "epochs": args.epochs,
                    "weight_decays": args.weight_decays,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "dataloader_num_workers": args.dataloader_num_workers,
                    "selection_metric": args.selection_metric,
                    "scheduler_type": args.scheduler_type,
                    "warmup_ratio": args.warmup_ratio,
                    "label_smoothing_factor": args.label_smoothing_factor,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "train_metrics_sample_size": TRAIN_METRICS_SAMPLE_SIZE,
                },
            },
            indent=2,
        )
    )

    train_df, validation_df, test_df = load_splits()
    train_df = sample_split(train_df, args.max_train_samples)
    validation_df = sample_split(validation_df, args.max_validation_samples)
    test_df = sample_split(test_df, args.max_test_samples)
    train_dataset = build_dataset(train_df)
    validation_dataset = build_dataset(validation_df)
    class_weights = get_class_weights(train_df["label_id"])
    selection_metric_key = resolve_eval_metric_key(args.selection_metric)

    trial_results: list[dict[str, float | str | int]] = []
    best_selection_score = -1.0
    best_f1 = -1.0
    best_dir: Path | None = None
    best_trial: dict[str, float | int] | None = None

    for learning_rate in args.learning_rates:
        for batch_size in args.batch_sizes:
            for epochs in args.epochs:
                for weight_decay in args.weight_decays:
                    for gradient_accumulation_steps in args.gradient_accumulation_steps:
                        trial = TrialConfig(
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            epochs=epochs,
                            weight_decay=weight_decay,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                        )
                        saved_metrics = None
                        if not args.force_retrain:
                            saved_metrics = load_saved_trial_result(trial)
                            if saved_metrics is not None and selection_metric_key not in saved_metrics:
                                saved_metrics = None

                        if saved_metrics is not None:
                            output_dir = Path(saved_metrics["model_dir"])
                            metrics = saved_metrics
                            print(f"Reusing saved RoBERTa artifact from {output_dir}")
                        else:
                            metrics, output_dir = run_trial(
                                trial=trial,
                                train_dataset=train_dataset,
                                validation_dataset=validation_dataset,
                                class_weights=class_weights,
                                max_length=args.max_length,
                                early_stopping_patience=args.early_stopping_patience,
                                dataloader_num_workers=args.dataloader_num_workers,
                                label_smoothing_factor=args.label_smoothing_factor,
                                selection_metric=args.selection_metric,
                                scheduler_type=args.scheduler_type,
                                warmup_ratio=args.warmup_ratio,
                                gradient_checkpointing=args.gradient_checkpointing,
                            )

                        selection_score = float(metrics[selection_metric_key])
                        validation_f1 = float(metrics.get("eval_f1_score", 0.0))
                        eval_metrics = {
                            key: float(value) if isinstance(value, (np.floating, float)) else value
                            for key, value in metrics.items()
                            if key.startswith("eval_")
                        }
                        trial_results.append(
                            {
                                **asdict(trial),
                                **eval_metrics,
                                "selection_metric": args.selection_metric,
                                "selection_score": selection_score,
                                "model_dir": str(output_dir),
                            }
                        )
                        if selection_score > best_selection_score or (
                            np.isclose(selection_score, best_selection_score) and validation_f1 > best_f1
                        ):
                            best_selection_score = selection_score
                            best_f1 = validation_f1
                            best_dir = output_dir
                            best_trial = asdict(trial)

    if best_dir is None or best_trial is None:
        raise RuntimeError("No RoBERTa trial completed successfully.")

    if args.force_retrain:
        test_metrics = evaluate_best_model(best_dir, test_df, max_length=args.max_length)
    else:
        test_metrics = load_saved_test_metrics(best_dir) or evaluate_best_model(best_dir, test_df, max_length=args.max_length)

    trials_df = pd.DataFrame(trial_results).sort_values(
        by=["selection_score", "eval_f1_score"],
        ascending=[False, False],
    )
    trials_df.to_csv(REPORTS_DIR / "roberta_trials.csv", index=False)
    save_json(
        {
            "best_trial": best_trial,
            "selection_metric": args.selection_metric,
            "best_validation_metric": best_selection_score,
            "best_validation_f1": best_f1,
            "best_model_dir": str(best_dir),
            "hardware_profile": asdict(hardware_profile),
            "resolved_defaults": {
                "colab_t4_profile": args.colab_t4_profile,
                "max_length": args.max_length,
                "learning_rates": args.learning_rates,
                "batch_sizes": args.batch_sizes,
                "epochs": args.epochs,
                "weight_decays": args.weight_decays,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "dataloader_num_workers": args.dataloader_num_workers,
                "scheduler_type": args.scheduler_type,
                "warmup_ratio": args.warmup_ratio,
                "label_smoothing_factor": args.label_smoothing_factor,
                "gradient_checkpointing": args.gradient_checkpointing,
            },
            "test_metrics": test_metrics,
        },
        REPORTS_DIR / "roberta_summary.json",
    )

    ml_comparison_path = REPORTS_DIR / "ml_model_comparison.csv"
    if ml_comparison_path.exists():
        ml_results = pd.read_csv(ml_comparison_path)
        combined_results = pd.concat([ml_results, pd.DataFrame([{"model": "roberta", **test_metrics}])], ignore_index=True)
        combined_results.to_csv(REPORTS_DIR / "all_model_comparison.csv", index=False)
        plot_model_comparison(combined_results, metric="f1_score", filename="all_model_comparison.png")
        plot_model_comparison(combined_results, metric="accuracy", filename="all_model_accuracy_comparison.png")
        plot_model_comparison(combined_results, metric="f1_macro", filename="all_model_macro_f1_comparison.png")
        plot_metric_dashboard(combined_results, filename="all_model_metric_dashboard.png")

    print(
        json.dumps(
            {
                "best_model_dir": str(best_dir),
                "best_trial": best_trial,
                "selection_metric": args.selection_metric,
                "best_validation_metric": best_selection_score,
                "test_metrics": test_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
