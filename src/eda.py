from __future__ import annotations

import argparse
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from src.config import FIGURES_DIR, PROCESSED_DIR, ensure_directories


sns.set_theme(style="whitegrid")


def load_split(split_name: str = "train") -> pd.DataFrame:
    path = PROCESSED_DIR / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed split not found at {path}. Run preprocessing first.")
    return pd.read_csv(path)


def plot_class_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(
        data=df,
        x="sentiment",
        hue="sentiment",
        order=["negative", "neutral", "positive"],
        palette="viridis",
        legend=False,
    )
    ax.set_title("Sentiment Class Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=200)
    plt.close()


def plot_review_length_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x="review_length", hue="sentiment", bins=60, kde=True, element="step")
    plt.title("Review Length Distribution")
    plt.xlabel("Token Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "review_length_distribution.png", dpi=200)
    plt.close()


def plot_top_words(df: pd.DataFrame, top_n: int = 20) -> None:
    for sentiment in ["negative", "neutral", "positive"]:
        plt.figure(figsize=(12, 6))
        tokens = " ".join(df.loc[df["sentiment"] == sentiment, "clean_text"].fillna("")).split()
        top_tokens = Counter(tokens).most_common(top_n)
        sentiment_df = pd.DataFrame(top_tokens, columns=["word", "count"])
        sns.barplot(data=sentiment_df, x="count", y="word", color="#2a9d8f")
        plt.title(f"Most Frequent Words: {sentiment.title()}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"top_words_{sentiment}.png", dpi=200)
        plt.close()


def plot_wordclouds(df: pd.DataFrame) -> None:
    for sentiment in ["negative", "neutral", "positive"]:
        text = " ".join(df.loc[df["sentiment"] == sentiment, "clean_text"].fillna(""))
        wordcloud = WordCloud(width=1200, height=600, background_color="white", colormap="Set2").generate(text)
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud: {sentiment.title()}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"wordcloud_{sentiment}.png", dpi=200)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA charts for sentiment analysis.")
    parser.add_argument("--split", default="train", help="Processed split to visualize.")
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    args = parse_args()
    df = load_split(args.split)
    plot_class_distribution(df)
    plot_review_length_distribution(df)
    plot_top_words(df)
    plot_wordclouds(df)
    print(f"Saved EDA figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
