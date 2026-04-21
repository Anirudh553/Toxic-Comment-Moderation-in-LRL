import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import yaml
from sklearn.model_selection import train_test_split

from src.data.dataset import load_dataset
from src.data.preprocessing import normalize_text
from src.evaluation.metrics import classification_metrics
from src.models.baseline import build_baseline_pipeline
from src.utils.seed import set_seed


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config if it exists, otherwise return an empty dict."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _load_and_normalize_dataset(
    csv_path: str,
    text_column: Optional[str],
    label_column: Optional[str],
    language_column: Optional[str],
    id_column: Optional[str],
    dataset_name: Optional[str],
):
    df = load_dataset(
        csv_path,
        text_column=text_column,
        label_column=label_column,
        language_column=language_column,
        id_column=id_column,
        dataset_name=dataset_name,
    )
    df["text"] = df["text"].map(normalize_text)
    return df


def _resolve_data_paths(args: argparse.Namespace, data_config: Dict[str, Any]):
    configured_raw_csv = data_config.get("raw_csv")
    configured_train_csv = data_config.get("train_csv")
    configured_eval_csv = data_config.get("validation_csv") or data_config.get("test_csv")

    use_explicit_prepared_splits = bool(args.train_csv)
    use_configured_prepared_splits = bool(
        configured_train_csv
        and configured_eval_csv
        and Path(configured_train_csv).exists()
        and Path(configured_eval_csv).exists()
    )

    if use_explicit_prepared_splits:
        return None, args.train_csv, args.eval_csv

    if use_configured_prepared_splits:
        return None, configured_train_csv, args.eval_csv or configured_eval_csv

    return args.data_csv or configured_raw_csv, None, args.eval_csv


def train_baseline(
    data_csv: Optional[str] = None,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    test_size: float = 0.2,
    output_dir: str = "artifacts/baseline",
    seed: int = 42,
    train_csv: Optional[str] = None,
    eval_csv: Optional[str] = None,
    language_column: Optional[str] = None,
    id_column: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Dict[str, float]:
    """Train the baseline classifier, save artifacts, and return metrics."""
    set_seed(seed)
    source_csv = data_csv or train_csv
    if not source_csv:
        raise ValueError("Provide either data_csv or train_csv.")

    if eval_csv:
        train_df = _load_and_normalize_dataset(
            source_csv,
            text_column=text_column,
            label_column=label_column,
            language_column=language_column,
            id_column=id_column,
            dataset_name=dataset_name,
        )
        test_df = _load_and_normalize_dataset(
            eval_csv,
            text_column=text_column,
            label_column=label_column,
            language_column=language_column,
            id_column=id_column,
            dataset_name=dataset_name,
        )
    else:
        df = _load_and_normalize_dataset(
            source_csv,
            text_column=text_column,
            label_column=label_column,
            language_column=language_column,
            id_column=id_column,
            dataset_name=dataset_name,
        )

        stratify = df["label"] if df["label"].nunique() > 1 else None
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )

    pipeline = build_baseline_pipeline()
    pipeline.fit(train_df["text"], train_df["label"])

    predictions = pipeline.predict(test_df["text"])
    metrics = classification_metrics(test_df["label"], predictions)

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, artifact_dir / "model.joblib")
    with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the toxic comment baseline model.")
    parser.add_argument("--config", default="experiments/configs/baseline.yaml")
    parser.add_argument("--data-csv", help="Path to a CSV dataset that will be split for training and evaluation.")
    parser.add_argument("--train-csv", help="Path to a prepared training CSV.")
    parser.add_argument("--eval-csv", help="Path to a prepared evaluation CSV.")
    parser.add_argument("--dataset-name", help="Dataset registry name used for column inference.")
    parser.add_argument("--text-column", help="Text column name in the CSV.")
    parser.add_argument("--label-column", help="Label column name in the CSV.")
    parser.add_argument("--language-column", help="Language column name in the CSV.")
    parser.add_argument("--id-column", help="Identifier column name in the CSV.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction used for evaluation.")
    parser.add_argument("--output-dir", default="artifacts/baseline", help="Where to save the model and metrics.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_config = config.get("data", {})
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    data_csv, train_csv, eval_csv = _resolve_data_paths(args, data_config)

    if not data_csv and not train_csv:
        raise ValueError("No dataset path provided. Use --data-csv or --train-csv.")

    metrics = train_baseline(
        data_csv=data_csv,
        train_csv=train_csv,
        eval_csv=eval_csv,
        text_column=args.text_column or data_config.get("text_column"),
        label_column=args.label_column or data_config.get("label_column"),
        language_column=args.language_column or data_config.get("language_column"),
        id_column=args.id_column or data_config.get("id_column"),
        dataset_name=args.dataset_name or data_config.get("dataset_name"),
        test_size=args.test_size,
        output_dir=args.output_dir,
        seed=seed,
    )

    print("Baseline training complete.")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"Artifacts saved to: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
