import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.catalog import DATASET_REGISTRY
from src.data.dataset import load_dataset
from src.data.labels import SUBTYPE_LABEL_COLUMNS, available_label_modes
from src.data.preprocessing import normalize_text
from src.utils.seed import set_seed


def _stratify_labels(df: pd.DataFrame, label_column: str) -> Optional[pd.Series]:
    counts = df[label_column].value_counts()
    if df[label_column].nunique() < 2 or counts.min() < 2:
        return None
    return df[label_column]


def _split_frame(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    label_column: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_size <= 0:
        return df.reset_index(drop=True), df.iloc[0:0].copy()

    stratify = _stratify_labels(df, label_column)
    try:
        first, second = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        first, second = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )

    return first.reset_index(drop=True), second.reset_index(drop=True)


def build_dataset_summary(
    raw_rows: int,
    clean_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    source_csv: str,
    source_url: str,
) -> Dict[str, object]:
    label_distribution = {
        str(label): int(count)
        for label, count in clean_df["label"].value_counts().sort_index().to_dict().items()
    }
    summary: Dict[str, object] = {
        "dataset_name": dataset_name,
        "source_csv": source_csv,
        "source_url": source_url,
        "row_counts": {
            "raw": raw_rows,
            "clean": len(clean_df),
            "train": len(train_df),
            "validation": len(validation_df),
            "test": len(test_df),
        },
        "label_distribution": label_distribution,
        "available_label_modes": list(available_label_modes(clean_df.columns)),
    }

    subtype_columns = [column for column in SUBTYPE_LABEL_COLUMNS if column in clean_df.columns]
    if subtype_columns:
        summary["subtype_distribution"] = {
            column: int(clean_df[column].sum())
            for column in subtype_columns
        }

    if "language" in clean_df.columns:
        summary["language_distribution"] = {
            str(language): int(count)
            for language, count in clean_df["language"]
            .fillna("unknown")
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        }

    return summary


def prepare_dataset(
    source_csv: str,
    output_dir: str = "data",
    dataset_name: str = "multilingual_toxic_comments",
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    language_column: Optional[str] = None,
    id_column: Optional[str] = None,
    validation_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
) -> Dict[str, object]:
    """Clean a raw dataset, create splits, and write dataset metadata."""
    if validation_size < 0 or test_size < 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be non-negative and sum to less than 1.")

    set_seed(seed)
    raw_df = pd.read_csv(source_csv)
    clean_df = load_dataset(
        source_csv,
        text_column=text_column,
        label_column=label_column,
        language_column=language_column,
        id_column=id_column,
        dataset_name=dataset_name,
    )

    clean_df["text"] = clean_df["text"].map(normalize_text)
    clean_df = clean_df[clean_df["text"].str.len() > 0].copy()

    dedupe_columns = ["text", "label"]
    dedupe_columns.extend([column for column in SUBTYPE_LABEL_COLUMNS if column in clean_df.columns])
    if "language" in clean_df.columns:
        dedupe_columns.append("language")
    clean_df = clean_df.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)

    train_validation_df, test_df = _split_frame(clean_df, test_size=test_size, seed=seed)

    remaining_validation_size = 0.0
    if validation_size > 0:
        remaining_validation_size = validation_size / (1 - test_size)
    train_df, validation_df = _split_frame(
        train_validation_df,
        test_size=remaining_validation_size,
        seed=seed,
    )

    output_root = Path(output_dir)
    interim_dir = output_root / "interim"
    processed_dir = output_root / "processed" / dataset_name
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    clean_path = interim_dir / f"{dataset_name}_clean.csv"
    train_path = processed_dir / "train.csv"
    validation_path = processed_dir / "validation.csv"
    test_path = processed_dir / "test.csv"
    summary_path = processed_dir / "summary.json"

    clean_df.to_csv(clean_path, index=False)
    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)

    spec = DATASET_REGISTRY.get(dataset_name)
    summary = build_dataset_summary(
        raw_rows=len(raw_df),
        clean_df=clean_df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        dataset_name=dataset_name,
        source_csv=str(Path(source_csv)),
        source_url=spec.source_url if spec else "",
    )
    summary["files"] = {
        "clean_csv": str(clean_path),
        "train_csv": str(train_path),
        "validation_csv": str(validation_path),
        "test_csv": str(test_path),
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a toxic comment dataset for modeling.")
    parser.add_argument("--source-csv", required=True, help="Path to the raw CSV dataset.")
    parser.add_argument("--output-dir", default="data", help="Root directory for interim and processed data.")
    parser.add_argument(
        "--dataset-name",
        default="multilingual_toxic_comments",
        help="Dataset registry name used for column inference and output folder naming.",
    )
    parser.add_argument("--text-column", help="Raw text column name.")
    parser.add_argument("--label-column", help="Raw label column name.")
    parser.add_argument("--language-column", help="Raw language column name.")
    parser.add_argument("--id-column", help="Raw identifier column name.")
    parser.add_argument("--validation-size", type=float, default=0.1, help="Fraction reserved for validation.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction reserved for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(
        source_csv=args.source_csv,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        text_column=args.text_column,
        label_column=args.label_column,
        language_column=args.language_column,
        id_column=args.id_column,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    print("Dataset preparation complete.")
    print(json.dumps(summary["row_counts"], indent=2))
    print(f"Processed files written to: {Path(summary['files']['train_csv']).parent}")


if __name__ == "__main__":
    main()
