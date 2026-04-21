from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from src.data.catalog import DATASET_REGISTRY
from src.data.labels import (
    SUBTYPE_COLUMN_CANDIDATES,
    SUBTYPE_LABEL_COLUMNS,
    available_label_modes,
    build_subtype_frame,
    coerce_subtype_indicator,
    parse_label_value,
)


def _resolve_column(
    columns: Iterable[str],
    requested_column: Optional[str],
    candidate_columns: Iterable[str],
    field_name: str,
    required: bool,
) -> Optional[str]:
    available = list(columns)

    if requested_column:
        if requested_column not in available:
            raise ValueError(f"Column '{requested_column}' was requested for {field_name} but is missing.")
        return requested_column

    for candidate in candidate_columns:
        if candidate in available:
            return candidate

    if required:
        raise ValueError(f"Missing required column for {field_name}. Available columns: {available}")

    return None


def _resolve_subtype_columns(columns: Iterable[str]) -> Dict[str, Optional[str]]:
    available = list(columns)
    resolved_columns: Dict[str, Optional[str]] = {}
    for canonical_name, candidates in SUBTYPE_COLUMN_CANDIDATES.items():
        resolved_columns[canonical_name] = next((candidate for candidate in candidates if candidate in available), None)
    return resolved_columns


def standardize_dataset(
    df: pd.DataFrame,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    language_column: Optional[str] = None,
    id_column: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """Map dataset-specific columns to the project's canonical schema."""
    spec = DATASET_REGISTRY.get(dataset_name) if dataset_name else None

    text_candidates = spec.text_columns if spec else ("text", "comment_text", "comment")
    label_candidates = spec.label_columns if spec else ("label", "toxic", "target")
    language_candidates = spec.language_columns if spec else ("language", "lang")
    id_candidates = spec.id_columns if spec else ("id", "comment_id")

    resolved_text_column = _resolve_column(df.columns, text_column, text_candidates, "text", required=True)
    resolved_subtype_columns = _resolve_subtype_columns(df.columns)
    has_subtype_columns = any(resolved_subtype_columns.values())

    resolved_label_column = _resolve_column(
        df.columns,
        label_column,
        label_candidates,
        "label",
        required=not has_subtype_columns,
    )
    resolved_language_column = _resolve_column(
        df.columns,
        language_column,
        language_candidates,
        "language",
        required=False,
    )
    resolved_id_column = _resolve_column(df.columns, id_column, id_candidates, "id", required=False)

    subtype_rows = [{label: 0 for label in SUBTYPE_LABEL_COLUMNS} for _ in range(len(df))]
    if has_subtype_columns:
        for label_name, column_name in resolved_subtype_columns.items():
            if not column_name:
                continue
            values = df[column_name].map(coerce_subtype_indicator).tolist()
            for index, value in enumerate(values):
                subtype_rows[index][label_name] = value

    binary_labels = None
    has_derived_subtype_schema = False
    if resolved_label_column:
        parsed_labels = df[resolved_label_column].map(parse_label_value).tolist()
        binary_labels = pd.Series([item["binary_label"] for item in parsed_labels], index=df.index)
        has_derived_subtype_schema = any(item["used_subtype_schema"] for item in parsed_labels)
        for index, parsed in enumerate(parsed_labels):
            for label_name, value in parsed["subtype_flags"].items():
                subtype_rows[index][label_name] = max(subtype_rows[index][label_name], value)

    subtype_frame = build_subtype_frame(subtype_rows)
    has_any_subtype_signal = bool(subtype_frame.to_numpy().sum())
    has_subtype_schema = has_subtype_columns or has_any_subtype_signal or has_derived_subtype_schema
    derived_binary_labels = subtype_frame.max(axis=1).astype(int) if has_subtype_schema else None

    if binary_labels is not None and derived_binary_labels is not None:
        label_series = pd.concat([binary_labels.astype(int), derived_binary_labels], axis=1).max(axis=1).astype(int)
    elif binary_labels is not None:
        label_series = binary_labels.astype(int)
    elif derived_binary_labels is not None:
        label_series = derived_binary_labels.astype(int)
    else:
        raise ValueError("Missing required label columns. Provide a binary label or subtype labels.")

    standardized = pd.DataFrame({"text": df[resolved_text_column], "label": label_series})

    if has_subtype_schema:
        for label_name in SUBTYPE_LABEL_COLUMNS:
            standardized[label_name] = subtype_frame[label_name].astype(int)

    if resolved_language_column:
        standardized["language"] = df[resolved_language_column].astype(str).str.strip()

    if resolved_id_column:
        standardized["id"] = df[resolved_id_column]

    standardized["text"] = standardized["text"].fillna("").astype(str).str.strip()
    standardized = standardized[standardized["text"] != ""].reset_index(drop=True)

    ordered_columns = ["text", "label"]
    if any(column in standardized.columns for column in SUBTYPE_LABEL_COLUMNS):
        ordered_columns.extend([column for column in SUBTYPE_LABEL_COLUMNS if column in standardized.columns])
    if "language" in standardized.columns:
        ordered_columns.append("language")
    if "id" in standardized.columns:
        ordered_columns.append("id")

    return standardized[ordered_columns]


def load_dataset(
    csv_path: str,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    language_column: Optional[str] = None,
    id_column: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load a raw CSV dataset and map it to canonical training columns."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    standardized = standardize_dataset(
        df,
        text_column=text_column,
        label_column=label_column,
        language_column=language_column,
        id_column=id_column,
        dataset_name=dataset_name,
    )
    standardized.attrs["available_label_modes"] = available_label_modes(standardized.columns)
    return standardized
