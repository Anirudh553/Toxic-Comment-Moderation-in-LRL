# Toxic Comment Moderation for Low-Resource Languages

Starter repository structure for an NLP project focused on toxic comment detection in low-resource language settings.

## Default dataset

This repository is now wired for the Multilingual Toxic Comment Classification dataset:

- Source: https://github.com/AkshayaANANTAM/Multilingual-Toxic-comments-classification
- Raw columns supported out of the box: `comment_text`, `toxic`, `lang`, `id`
- Canonical columns produced by the project pipeline: `text`, `label`, optional subtype tags (`abusive`, `hate_targeted`, `threat`), `language`, `id`

## Project layout

```text
NLP_final_project/
|-- data/
|   |-- raw/
|   |-- interim/
|   |-- processed/
|   `-- external/
|-- experiments/
|   `-- configs/
|-- notebooks/
|-- reports/
|   |-- figures/
|   `-- results/
|-- scripts/
|-- src/
|   |-- config/
|   |-- data/
|   |-- features/
|   |-- models/
|   |-- training/
|   |-- evaluation/
|   |-- inference/
|   `-- utils/
|-- tests/
|-- requirements.txt
`-- README.md
```

## What goes where

- `data/raw/`: original datasets collected from Kaggle, GitHub, or papers.
- `data/interim/`: cleaned but not fully transformed data.
- `data/processed/`: train, validation, and test files ready for modeling.
- `data/external/`: lexicons, embeddings, and other third-party resources.
- `src/data/`: loaders, preprocessing, and dataset preparation.
- `src/features/`: tokenization and feature engineering helpers.
- `src/models/`: baseline ML models and transformer model wrappers.
- `src/training/`: training pipelines.
- `src/evaluation/`: metrics and evaluation logic.
- `src/inference/`: prediction utilities.
- `src/utils/`: shared helper functions.
- `experiments/configs/`: experiment-specific YAML configuration files.
- `reports/`: plots, tables, and final outputs.
- `tests/`: unit tests for preprocessing, data loading, and metrics.

## Suggested workflow

1. Download the multilingual toxic comments dataset and place the CSV in `data/raw/multilingual_toxic_comments.csv`.
2. Prepare cleaned data and train/validation/test splits with `python -m src.data.prepare`.
3. Review the generated summary in `data/processed/multilingual_toxic_comments/summary.json`.
4. Train the baseline from the processed splits or from a raw CSV.
5. Fine-tune a smaller multilingual pretrained model first, then compare against larger checkpoints only if needed.
6. Track experiment settings in `experiments/configs/`.

## Prepare the data

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Prepare the default multilingual dataset:

```powershell
python -m src.data.prepare `
  --source-csv data/raw/multilingual_toxic_comments.csv `
  --dataset-name multilingual_toxic_comments
```

Generated files:

- cleaned full dataset: `data/interim/multilingual_toxic_comments_clean.csv`
- training split: `data/processed/multilingual_toxic_comments/train.csv`
- validation split: `data/processed/multilingual_toxic_comments/validation.csv`
- test split: `data/processed/multilingual_toxic_comments/test.csv`
- dataset stats: `data/processed/multilingual_toxic_comments/summary.json`

If your raw data uses richer moderation labels such as `neutral`, `abusive`, `hate-targeted`, or `threat`, the preparation step preserves or derives canonical subtype columns named `abusive`, `hate_targeted`, and `threat`. The binary `label` column is still kept for backward compatibility.

## Run the baseline

Train and evaluate from prepared splits:

```powershell
python -m src.training.train `
  --train-csv data/processed/multilingual_toxic_comments/train.csv `
  --eval-csv data/processed/multilingual_toxic_comments/validation.csv `
  --dataset-name multilingual_toxic_comments
```

Or train directly from a raw CSV and let the script create an evaluation split:

```powershell
python -m src.training.train `
  --data-csv data/raw/multilingual_toxic_comments.csv `
  --dataset-name multilingual_toxic_comments
```

If the prepared split files already exist at the paths in `experiments/configs/baseline.yaml`, you can also run:

```powershell
python -m src.training.train
```

Outputs:

- trained model: `artifacts/baseline/model.joblib`
- evaluation metrics: `artifacts/baseline/metrics.json`

## Fine-Tune a Small Multilingual Model

The project now uses the smaller `distilbert-base-multilingual-cased` checkpoint as the default fine-tuning target. It keeps `XLM-RoBERTa-base` and `MuRIL` available when you want a larger multilingual encoder or a Hinglish-focused alternative.

Train from prepared splits:

```powershell
python -m src.training.train_transformer `
  --train-csv data/processed/multilingual_toxic_comments/train.csv `
  --eval-csv data/processed/multilingual_toxic_comments/validation.csv `
  --dataset-name multilingual_toxic_comments `
  --model-name slm `
  --output-dir artifacts/slm
```

Merge multiple prepared training files by repeating the same flag:

```powershell
python -m src.training.train_transformer `
  --train-csv data/processed/source_a/train.csv `
  --train-csv data/processed/source_b/train.csv `
  --eval-csv data/processed/source_a/validation.csv `
  --eval-csv data/processed/source_b/validation.csv `
  --model-name slm `
  --output-dir artifacts/slm_multi
```

The repeated-file path assumes the input CSVs can all be standardized with the same dataset settings or column overrides.

Switch to `XLM-RoBERTa-base` when you have more compute:

```powershell
python -m src.training.train_transformer `
  --train-csv data/processed/multilingual_toxic_comments/train.csv `
  --eval-csv data/processed/multilingual_toxic_comments/validation.csv `
  --dataset-name multilingual_toxic_comments `
  --model-name xlm-roberta-base `
  --output-dir artifacts/xlm-r
```

Switch to MuRIL for Hinglish-heavy experiments:

```powershell
python -m src.training.train_transformer `
  --train-csv data/processed/multilingual_toxic_comments/train.csv `
  --eval-csv data/processed/multilingual_toxic_comments/validation.csv `
  --dataset-name multilingual_toxic_comments `
  --model-name muril `
  --output-dir artifacts/muril
```

You can also run the config-driven default:

```powershell
python -m src.training.train_transformer
```

Transformer outputs:

- saved model + tokenizer: `artifacts/transformer/`
- evaluation metrics: `artifacts/transformer/metrics.json`
- resolved model metadata: `artifacts/transformer/model_config.json`

For CPU-only runs, the trainer now auto-tunes DataLoader workers and uses a larger evaluation batch size. If you need a quicker benchmark or a faster iteration loop, add flags like `--max-train-samples 4096 --max-eval-samples 1024 --logging-steps 200`.

To train a subtype-aware multilabel model from prepared CSVs that include `abusive`, `hate_targeted`, and `threat`, run:

```powershell
python -m src.training.train_transformer `
  --train-csv data/processed/multilingual_toxic_comments/train.csv `
  --eval-csv data/processed/multilingual_toxic_comments/validation.csv `
  --dataset-name multilingual_toxic_comments `
  --label-mode subtype_multilabel `
  --output-dir artifacts/transformer-subtypes
```
