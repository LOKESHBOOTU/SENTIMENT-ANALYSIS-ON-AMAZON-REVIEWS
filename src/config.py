from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = ROOT_DIR / "reviews.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

TEXT_COLUMN = "body"
RATING_COLUMN = "rating"
TITLE_COLUMN = "title_x"
OPTIONAL_METADATA = ["brand", "verified", TITLE_COLUMN]

NEGATIVE = "negative"
NEUTRAL = "neutral"
POSITIVE = "positive"

LABEL_MAPPING = {
    1: NEGATIVE,
    2: NEGATIVE,
    3: NEUTRAL,
    4: POSITIVE,
    5: POSITIVE,
}

LABEL_TO_ID = {NEGATIVE: 0, NEUTRAL: 1, POSITIVE: 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}

SEED = 42


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = SEED


def ensure_directories() -> None:
    for directory in [DATA_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
