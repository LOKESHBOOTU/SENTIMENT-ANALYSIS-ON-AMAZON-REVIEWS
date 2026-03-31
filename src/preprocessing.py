from __future__ import annotations

import argparse
import re
import string
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ID_TO_LABEL,
    LABEL_MAPPING,
    LABEL_TO_ID,
    OPTIONAL_METADATA,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    RATING_COLUMN,
    SEED,
    SplitConfig,
    TEXT_COLUMN,
    TITLE_COLUMN,
    ensure_directories,
    set_seed,
)
from src.utils import save_json


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "had", "has", "have", "i", "if", "in", "is", "it", "its", "my", "of", "on", "or",
    "so", "that", "the", "this", "to", "was", "were", "with", "you", "your",
}


@dataclass
class PreparedData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = [token for token in text.split() if token not in STOP_WORDS]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def map_rating_to_label(rating: int | float | str) -> str:
    try:
        return LABEL_MAPPING[int(float(rating))]
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"Unsupported rating value: {rating}") from exc


def combine_review_fields(df: pd.DataFrame) -> pd.Series:
    title = df[TITLE_COLUMN].fillna("") if TITLE_COLUMN in df.columns else ""
    body = df[TEXT_COLUMN].fillna("")
    if isinstance(title, str):
        return body.astype(str)
    combined = title.astype(str).str.strip() + ". " + body.astype(str).str.strip()
    return combined.str.strip(". ").str.strip()


def load_and_prepare_data(
    data_path: str | Path = RAW_DATA_PATH,
    split_config: SplitConfig = SplitConfig(),
    use_metadata: bool = True,
) -> PreparedData:
    set_seed(SEED)
    ensure_directories()

    df = pd.read_csv(data_path)
    required_columns = {TEXT_COLUMN, RATING_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    df = df.dropna(subset=[TEXT_COLUMN, RATING_COLUMN]).copy()
    df["rating"] = pd.to_numeric(df[RATING_COLUMN], errors="coerce")
    df = df.dropna(subset=["rating"])

    df["sentiment"] = df["rating"].apply(map_rating_to_label)
    df["label_id"] = df["sentiment"].map(LABEL_TO_ID)
    df["review_text"] = combine_review_fields(df)
    df["clean_text"] = df["review_text"].apply(clean_text)
    df["review_length"] = df["review_text"].str.split().str.len()

    if use_metadata:
        for column in OPTIONAL_METADATA:
            if column in df.columns:
                df[column] = df[column].fillna("")

    feature_columns = ["review_text", "clean_text", "sentiment", "label_id", "rating", "review_length"]
    feature_columns += [column for column in OPTIONAL_METADATA if column in df.columns]
    model_df = df[feature_columns].copy()

    train_df, test_df = train_test_split(
        model_df,
        test_size=split_config.test_size,
        random_state=split_config.random_state,
        stratify=model_df["sentiment"],
    )

    adjusted_val_size = split_config.val_size / (1 - split_config.test_size)
    train_df, validation_df = train_test_split(
        train_df,
        test_size=adjusted_val_size,
        random_state=split_config.random_state,
        stratify=train_df["sentiment"],
    )

    return PreparedData(
        train=train_df.reset_index(drop=True),
        validation=validation_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def save_splits(prepared_data: PreparedData) -> None:
    split_paths = {
        "train": PROCESSED_DIR / "train.csv",
        "validation": PROCESSED_DIR / "validation.csv",
        "test": PROCESSED_DIR / "test.csv",
    }
    prepared_data.train.to_csv(split_paths["train"], index=False)
    prepared_data.validation.to_csv(split_paths["validation"], index=False)
    prepared_data.test.to_csv(split_paths["test"], index=False)

    metadata = {
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "split_sizes": {
            "train": len(prepared_data.train),
            "validation": len(prepared_data.validation),
            "test": len(prepared_data.test),
        },
        "random_seed": SEED,
    }
    save_json(metadata, PROCESSED_DIR / "metadata.json")
    joblib.dump(STOP_WORDS, PROCESSED_DIR / "stopwords.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare review data for sentiment analysis.")
    parser.add_argument("--data-path", type=Path, default=RAW_DATA_PATH, help="Path to the raw CSV dataset.")
    parser.add_argument("--no-metadata", action="store_true", help="Skip optional metadata retention.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared_data = load_and_prepare_data(data_path=args.data_path, use_metadata=not args.no_metadata)
    save_splits(prepared_data)
    print(
        f"Saved processed splits to {PROCESSED_DIR} | "
        f"train={len(prepared_data.train)}, validation={len(prepared_data.validation)}, test={len(prepared_data.test)}"
    )


if __name__ == "__main__":
    main()
