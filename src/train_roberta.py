from __future__ import annotations

import argparse
import json
import inspect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
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
    get_linear_schedule_with_warmup,
)

try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW

from src.config import ID_TO_LABEL, LABEL_TO_ID, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, SEED, ensure_directories, set_seed
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_model_comparison
from src.utils import load_json, save_json


MODEL_NAME = "roberta-base"
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
    cpu_logical: int
    ram_total_gb: float
    ram_available_gb: float


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer or self.optimizer,
                num_warmup_steps=max(1, int(0.1 * num_training_steps)),
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
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
    return HardwareProfile(
        has_gpu=torch.cuda.is_available(),
        cpu_logical=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=round(memory.total / (1024**3), 2),
        ram_available_gb=round(memory.available / (1024**3), 2),
    )


def choose_default_search_space(profile: HardwareProfile) -> dict[str, list | int]:
    if profile.has_gpu:
        return {
            "max_length": 256,
            "learning_rates": [2e-5, 3e-5, 5e-5],
            "batch_sizes": [16, 32],
            "epochs": [3, 4, 5],
            "weight_decays": [0.01],
            "gradient_accumulation_steps": [1, 2],
            "dataloader_num_workers": min(4, max(1, profile.cpu_logical // 4)),
        }

    if profile.ram_available_gb < 5:
        return {
            "max_length": 128,
            "learning_rates": [2e-5, 3e-5],
            "batch_sizes": [4],
            "epochs": [3],
            "weight_decays": [0.01],
            "gradient_accumulation_steps": [4],
            "dataloader_num_workers": 0,
        }

    return {
        "max_length": 128,
        "learning_rates": [2e-5, 3e-5],
        "batch_sizes": [8],
        "epochs": [3, 4],
        "weight_decays": [0.01],
        "gradient_accumulation_steps": [2, 4],
        "dataloader_num_workers": max(1, min(2, profile.cpu_logical // 6)),
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
        samples.append(group.sample(min(target, len(group)), random_state=SEED))
        assigned += min(target, len(group))
    return pd.concat(samples, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def build_training_arguments(trial_output_dir: Path, trial: TrialConfig, dataloader_num_workers: int):
    kwargs = {
        "output_dir": str(trial_output_dir / "checkpoints"),
        "learning_rate": trial.learning_rate,
        "per_device_train_batch_size": trial.batch_size,
        "per_device_eval_batch_size": trial.batch_size,
        "num_train_epochs": trial.epochs,
        "gradient_accumulation_steps": trial.gradient_accumulation_steps,
        "weight_decay": trial.weight_decay,
        "warmup_ratio": 0.1,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_score",
        "greater_is_better": True,
        "save_total_limit": 2,
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "seed": SEED,
        "fp16": torch.cuda.is_available(),
        "report_to": "none",
    }
    signature = inspect.signature(TrainingArguments.__init__)
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


def hf_compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
    preds = np.argmax(prediction.predictions, axis=1)
    return compute_metrics(prediction.label_ids, preds)


def get_class_weights(labels: pd.Series) -> torch.Tensor:
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(sorted(LABEL_TO_ID.values())),
        y=labels.to_numpy(),
    )
    return torch.tensor(class_weights, dtype=torch.float)


def save_training_curves(log_history: list[dict], output_dir: Path) -> None:
    history_df = pd.DataFrame(log_history)
    history_df.to_csv(output_dir / "trainer_log_history.csv", index=False)

    if history_df.empty:
        return

    curve_df = history_df[[column for column in history_df.columns if column in {
        "epoch", "loss", "eval_loss", "eval_accuracy", "train_loss", "train_accuracy"
    }]].copy()
    curve_df = curve_df.dropna(how="all")
    curve_df.to_csv(output_dir / "training_curves.csv", index=False)

    if "epoch" not in curve_df.columns:
        return

    epoch_metrics = history_df.groupby("epoch", dropna=True).agg(
        train_loss=("train_loss", "last"),
        val_loss=("eval_loss", "last"),
        train_accuracy=("train_accuracy", "last"),
        val_accuracy=("eval_accuracy", "last"),
    ).reset_index()
    epoch_metrics.to_csv(output_dir / "epoch_metrics.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=epoch_metrics, x="epoch", y="train_loss", marker="o", label="Train Loss")
    sns.lineplot(data=epoch_metrics, x="epoch", y="val_loss", marker="o", label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    sns.lineplot(data=epoch_metrics, x="epoch", y="train_accuracy", marker="o", label="Train Accuracy")
    sns.lineplot(data=epoch_metrics, x="epoch", y="val_accuracy", marker="o", label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(output_dir / "training_validation_curves.png", dpi=200)
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
) -> tuple[dict[str, float], Path]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_tokenized = tokenize_dataset(train_dataset, tokenizer, max_length=max_length)
    validation_tokenized = tokenize_dataset(validation_dataset, tokenizer, max_length=max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    trial_output_dir = trial_output_path(trial)
    training_args = build_training_arguments(trial_output_dir, trial, dataloader_num_workers=dataloader_num_workers)
    train_metrics_callback = TrainMetricsCallback(train_tokenized)

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
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
        {key: float(value) if isinstance(value, (np.floating, float)) else value for key, value in metrics.items()},
        trial_output_dir / "validation_metrics.json",
    )
    return metrics, trial_output_dir


def evaluate_best_model(model_dir: Path, test_df: pd.DataFrame, max_length: int) -> dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    test_dataset = tokenize_dataset(build_dataset(test_df), tokenizer, max_length=max_length)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(model_dir / "eval"), report_to="none", per_device_eval_batch_size=16),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=hf_compute_metrics,
    )
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    metrics = compute_metrics(test_df["label_id"], y_pred)

    ordered_labels = sorted(ID_TO_LABEL)
    plot_confusion_matrix(
        test_df["label_id"],
        y_pred,
        labels=ordered_labels,
        label_names=[ID_TO_LABEL[idx] for idx in ordered_labels],
        filename="roberta_confusion_matrix.png",
    )
    save_json(metrics, model_dir / "test_metrics.json")
    return metrics


def load_saved_test_metrics(model_dir: Path) -> dict[str, float] | None:
    metrics_path = model_dir / "test_metrics.json"
    if not metrics_path.exists():
        return None
    return load_json(metrics_path)


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
    parser.add_argument("--force-retrain", action="store_true", help="Retrain trials even if saved artifacts already exist.")
    return parser.parse_args()


def main() -> None:
    set_seed(SEED)
    ensure_directories()
    args = parse_args()
    hardware_profile = detect_hardware_profile()
    defaults = choose_default_search_space(hardware_profile)

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

    print(
        json.dumps(
            {
                "hardware_profile": {
                    "has_gpu": hardware_profile.has_gpu,
                    "cpu_logical": hardware_profile.cpu_logical,
                    "ram_total_gb": hardware_profile.ram_total_gb,
                    "ram_available_gb": hardware_profile.ram_available_gb,
                },
                "resolved_defaults": {
                    "max_length": args.max_length,
                    "learning_rates": args.learning_rates,
                    "batch_sizes": args.batch_sizes,
                    "epochs": args.epochs,
                    "weight_decays": args.weight_decays,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "dataloader_num_workers": args.dataloader_num_workers,
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

    trial_results: list[dict[str, float | str | int]] = []
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
                        if not args.force_retrain:
                            saved_metrics = load_saved_trial_result(trial)
                        else:
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
                            )

                        validation_f1 = float(metrics["eval_f1_score"])
                        trial_results.append(
                            {
                                "learning_rate": learning_rate,
                                "batch_size": batch_size,
                                "epochs": epochs,
                                "weight_decay": weight_decay,
                                "gradient_accumulation_steps": gradient_accumulation_steps,
                                "eval_accuracy": float(metrics["eval_accuracy"]),
                                "eval_precision": float(metrics["eval_precision"]),
                                "eval_recall": float(metrics["eval_recall"]),
                                "eval_f1_score": validation_f1,
                                "model_dir": str(output_dir),
                            }
                        )
                        if validation_f1 > best_f1:
                            best_f1 = validation_f1
                            best_dir = output_dir
                            best_trial = {
                                "learning_rate": learning_rate,
                                "batch_size": batch_size,
                                "epochs": epochs,
                                "weight_decay": weight_decay,
                                "gradient_accumulation_steps": gradient_accumulation_steps,
                            }

    if best_dir is None or best_trial is None:
        raise RuntimeError("No RoBERTa trial completed successfully.")

    if args.force_retrain:
        test_metrics = evaluate_best_model(best_dir, test_df, max_length=args.max_length)
    else:
        test_metrics = load_saved_test_metrics(best_dir) or evaluate_best_model(best_dir, test_df, max_length=args.max_length)
    pd.DataFrame(trial_results).sort_values("eval_f1_score", ascending=False).to_csv(REPORTS_DIR / "roberta_trials.csv", index=False)
    save_json(
        {
            "best_trial": best_trial,
            "best_validation_f1": best_f1,
            "best_model_dir": str(best_dir),
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
    print(json.dumps({"best_model_dir": str(best_dir), "best_trial": best_trial, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
