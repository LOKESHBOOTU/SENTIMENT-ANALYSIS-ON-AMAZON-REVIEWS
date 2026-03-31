# Sentiment Analysis on Product Reviews

This project delivers a production-ready, resume-level NLP workflow for classifying product reviews into `negative`, `neutral`, and `positive` sentiment classes. It combines strong traditional ML baselines with a fine-tuned `roberta-base` transformer, adds imbalance handling, exports analysis visuals, and includes a Streamlit app for deployment-ready inference.

## Label Mapping

- `1-2 -> negative`
- `3 -> neutral`
- `4-5 -> positive`

## Project Structure

```text
Sentiment Analysis on Amazon reviews/
├── app.py
├── reviews.csv
├── requirements.txt
├── README.md
├── data/
│   └── processed/
├── models/
├── notebooks/
├── outputs/
│   ├── figures/
│   └── reports/
└── src/
    ├── __init__.py
    ├── config.py
    ├── eda.py
    ├── evaluate.py
    ├── inference.py
    ├── preprocessing.py
    ├── train_ml.py
    ├── train_roberta.py
    └── utils.py
```

## Features

- Robust preprocessing with missing-value handling, text cleaning, token filtering, and stratified splitting
- EDA outputs including class balance, review-length histograms, frequent-word charts, and sentiment-wise word clouds
- Traditional baseline models:
  - Logistic Regression
  - Complement Naive Bayes
  - Linear SVM
- Transformer model:
  - Fine-tuned `roberta-base`
- Hyperparameter tuning:
  - `GridSearchCV` for TF-IDF models
  - learning-rate, batch-size, and epoch sweeps for RoBERTa
- Imbalance handling:
  - class weights for Logistic Regression and Linear SVM
  - class-weighted cross-entropy for RoBERTa
- Evaluation:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrices
  - model comparison bar charts

## Dataset Assumptions

The current repository already contains `reviews.csv` with the following relevant columns:

- `body`
- `rating`
- `title_x`
- `brand`
- `verified`

The preprocessing step combines `title_x` and `body` when possible so short titles can enrich the review representation.

## Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If Windows or OneDrive locks files inside `.venv`, a reliable fallback is to install packages into a local folder and prepend it to `PYTHONPATH`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --target .python_packages -r requirements.txt
$env:PYTHONPATH = (Resolve-Path .python_packages).Path
```

## Run Instructions

### 1. Preprocess the data

```powershell
python -m src.preprocessing
```

Artifacts created:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `data/processed/metadata.json`

### 2. Generate EDA

```powershell
python -m src.eda
```

Artifacts created in `outputs/figures/`:

- class distribution plot
- review length distribution
- top words per sentiment
- word clouds per sentiment

### 3. Train traditional ML models

```powershell
python -m src.train_ml
```

By default, this reuses saved model artifacts if they already exist. To retrain from scratch:

```powershell
python -m src.train_ml --force-retrain
```

Outputs:

- trained pipelines under `models/ml/`
- confusion matrices in `outputs/figures/`
- grid-search summary in `outputs/reports/ml_grid_search_summary.json`
- comparison table in `outputs/reports/ml_model_comparison.csv`
- combined comparison table in `outputs/reports/all_model_comparison.csv` after RoBERTa training

### 4. Fine-tune RoBERTa

```powershell
python -m src.train_roberta
```

By default, this reuses saved RoBERTa trials and test metrics if matching artifacts already exist. To force a fresh run:

```powershell
python -m src.train_roberta --force-retrain
```

Example with explicit tuning options:

```powershell
python -m src.train_roberta --learning-rates 2e-5 3e-5 --batch-sizes 8 16 --epochs 2 3
```

Outputs:

- fine-tuned models under `models/roberta/`
- reusable saved models so inference does not require retraining
- trial leaderboard in `outputs/reports/roberta_trials.csv`
- summary in `outputs/reports/roberta_summary.json`
- confusion matrix in `outputs/figures/roberta_confusion_matrix.png`
- training history in `models/roberta/<trial>/trainer_log_history.csv`
- epoch-wise train/validation metrics in `models/roberta/<trial>/epoch_metrics.csv`
- train vs validation loss/accuracy curves in `models/roberta/<trial>/training_validation_curves.png`

### 5. Launch the Streamlit app

```powershell
streamlit run app.py
```

## Deployment Notes

The Streamlit UI supports:

- free-text review input
- model selection between RoBERTa and Logistic Regression
- predicted sentiment output
- confidence score
- per-class probability display
- loading previously saved model artifacts without retraining

## Reproducibility

- Shared random seed in `src/config.py`
- Centralized label mappings
- Saved metrics and model artifacts
- Deterministic split strategy

## Suggested Future Enhancements

- Add SHAP or LIME for explainability
- Serve the best model through FastAPI
- Add unit tests and CI
- Track experiments with MLflow
