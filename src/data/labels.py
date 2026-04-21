import re
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import pandas as pd

BINARY_LABEL_MODE = "binary"
SUBTYPE_MULTILABEL_MODE = "subtype_multilabel"
LABEL_MODE_CHOICES = (BINARY_LABEL_MODE, SUBTYPE_MULTILABEL_MODE)

DEFAULT_BINARY_LABEL_NAMES = ("non-toxic", "toxic")
SUBTYPE_LABEL_COLUMNS = ("abusive", "hate_targeted", "threat")
NEUTRAL_LABEL = "neutral"

SUBTYPE_COLUMN_CANDIDATES: Mapping[str, Tuple[str, ...]] = {
    "abusive": ("abusive", "abuse", "offensive", "insult"),
    "hate_targeted": (
        "hate_targeted",
        "hate-targeted",
        "hate targeted",
        "identity_hate",
        "hate_speech",
        "hate speech",
    ),
    "threat": ("threat", "threatening", "violent_threat"),
}

_TOKEN_SPLIT_PATTERN = re.compile(r"[,;/|]+")
_NORMALIZE_PATTERN = re.compile(r"[\s\-]+")

_BINARY_VALUE_MAP: Mapping[str, int] = {
    "0": 0,
    "1": 1,
    "false": 0,
    "true": 1,
    "no": 0,
    "yes": 1,
    "neutral": 0,
    "none": 0,
    "non_toxic": 0,
    "clean": 0,
    "safe": 0,
    "toxic": 1,
    "abusive": 1,
    "offensive": 1,
    "hate_targeted": 1,
    "hate_speech": 1,
    "identity_hate": 1,
    "threat": 1,
    "threatening": 1,
}

_CATEGORY_TO_SUBTYPES: Mapping[str, Tuple[str, ...]] = {
    "neutral": (),
    "none": (),
    "non_toxic": (),
    "clean": (),
    "safe": (),
    "toxic": (),
    "abusive": ("abusive",),
    "offensive": ("abusive",),
    "insult": ("abusive",),
    "hate_targeted": ("hate_targeted",),
    "identity_hate": ("hate_targeted",),
    "hate_speech": ("hate_targeted",),
    "threat": ("threat",),
    "threatening": ("threat",),
    "violent_threat": ("threat",),
}


def normalize_label_token(value: Any) -> str:
    return _NORMALIZE_PATTERN.sub("_", str(value).strip().lower())


def coerce_binary_label(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("Missing label value.")

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_value = int(value)
        if numeric_value not in (0, 1):
            raise ValueError(f"Unsupported label value: {value}")
        return numeric_value

    normalized = normalize_label_token(value)
    if normalized not in _BINARY_VALUE_MAP:
        raise ValueError(f"Unsupported label value: {value}")
    return _BINARY_VALUE_MAP[normalized]


def coerce_subtype_indicator(value: Any) -> int:
    if pd.isna(value):
        return 0

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_value = int(value)
        if numeric_value not in (0, 1):
            raise ValueError(f"Unsupported subtype value: {value}")
        return numeric_value

    normalized = normalize_label_token(value)
    if normalized in {"", "0", "false", "no", "none", "neutral", "non_toxic", "clean", "safe"}:
        return 0
    if normalized in {"1", "true", "yes", "present", "toxic", *SUBTYPE_LABEL_COLUMNS, "offensive", "insult"}:
        return 1
    raise ValueError(f"Unsupported subtype value: {value}")


def parse_label_value(value: Any) -> Dict[str, Any]:
    subtype_flags = {label: 0 for label in SUBTYPE_LABEL_COLUMNS}

    if pd.isna(value):
        raise ValueError("Missing label value.")

    if isinstance(value, bool) or isinstance(value, (int, float)):
        return {
            "binary_label": coerce_binary_label(value),
            "subtype_flags": subtype_flags,
            "used_subtype_schema": False,
        }

    normalized_value = str(value).strip()
    if not normalized_value:
        raise ValueError("Missing label value.")

    normalized_token = normalize_label_token(normalized_value)
    if normalized_token in _BINARY_VALUE_MAP:
        for label_name in _CATEGORY_TO_SUBTYPES.get(normalized_token, ()):
            subtype_flags[label_name] = 1
        return {
            "binary_label": _BINARY_VALUE_MAP[normalized_token],
            "subtype_flags": subtype_flags,
            "used_subtype_schema": normalized_token in {
                "neutral",
                "none",
                "abusive",
                "offensive",
                "insult",
                "hate_targeted",
                "identity_hate",
                "hate_speech",
                "threat",
                "threatening",
                "violent_threat",
            },
        }

    binary_label = 0
    recognized_token = False
    for raw_token in _TOKEN_SPLIT_PATTERN.split(normalized_value):
        token = normalize_label_token(raw_token)
        if not token:
            continue
        if token in _CATEGORY_TO_SUBTYPES:
            recognized_token = True
            for label_name in _CATEGORY_TO_SUBTYPES[token]:
                subtype_flags[label_name] = 1
            if _CATEGORY_TO_SUBTYPES[token] or token == "toxic":
                binary_label = 1
            continue
        raise ValueError(f"Unsupported label value: {value}")

    if not recognized_token:
        raise ValueError(f"Unsupported label value: {value}")

    return {
        "binary_label": binary_label,
        "subtype_flags": subtype_flags,
        "used_subtype_schema": True,
    }


def build_subtype_frame(rows: Sequence[Mapping[str, int]]):
    return pd.DataFrame(rows, columns=list(SUBTYPE_LABEL_COLUMNS)).fillna(0).astype(int)


def label_names_for_mode(label_mode: str) -> Tuple[str, ...]:
    if label_mode == BINARY_LABEL_MODE:
        return DEFAULT_BINARY_LABEL_NAMES
    if label_mode == SUBTYPE_MULTILABEL_MODE:
        return SUBTYPE_LABEL_COLUMNS
    supported = ", ".join(LABEL_MODE_CHOICES)
    raise ValueError(f"Unsupported label mode '{label_mode}'. Supported values: {supported}")


def available_label_modes(columns: Iterable[str]) -> Tuple[str, ...]:
    available_columns = set(columns)
    modes = [BINARY_LABEL_MODE]
    if all(column in available_columns for column in SUBTYPE_LABEL_COLUMNS):
        modes.append(SUBTYPE_MULTILABEL_MODE)
    return tuple(modes)
