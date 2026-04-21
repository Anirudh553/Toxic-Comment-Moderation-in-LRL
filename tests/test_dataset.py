import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data.dataset import load_dataset
from src.data.prepare import prepare_dataset


def test_load_dataset_infers_multilingual_toxic_schema():
    test_dir = Path("tests") / f"dataset-{uuid4().hex}"
    test_dir.mkdir(parents=True, exist_ok=True)
    csv_path = test_dir / "multilingual_toxic_comments.csv"
    pd.DataFrame(
        {
            "id": [11, 12],
            "comment_text": ["Namaste", "You are awful"],
            "toxic": [0, 1],
            "lang": ["hi", "en"],
        }
    ).to_csv(csv_path, index=False)

    loaded = load_dataset(str(csv_path), dataset_name="multilingual_toxic_comments")

    assert list(loaded.columns) == ["text", "label", "language", "id"]
    assert loaded.attrs["available_label_modes"] == ("binary",)
    assert loaded.to_dict(orient="records")[1]["label"] == 1
    assert loaded.to_dict(orient="records")[0]["language"] == "hi"


def test_load_dataset_derives_subtype_columns_from_categorical_labels():
    test_dir = Path("tests") / f"dataset-{uuid4().hex}"
    test_dir.mkdir(parents=True, exist_ok=True)
    csv_path = test_dir / "moderation_labels.csv"
    pd.DataFrame(
        {
            "comment_text": [
                "Helpful note",
                "You are disgusting",
                "I will find you",
                "Go back to your country",
            ],
            "label": ["neutral", "abusive", "threat", "hate-targeted"],
        }
    ).to_csv(csv_path, index=False)

    loaded = load_dataset(str(csv_path), text_column="comment_text", label_column="label")

    assert list(loaded.columns) == ["text", "label", "abusive", "hate_targeted", "threat"]
    assert loaded.attrs["available_label_modes"] == ("binary", "subtype_multilabel")
    assert loaded.loc[0, "label"] == 0
    assert loaded.loc[1, "abusive"] == 1
    assert loaded.loc[2, "threat"] == 1
    assert loaded.loc[3, "hate_targeted"] == 1


def test_prepare_dataset_writes_clean_and_split_files():
    test_dir = Path("tests") / f"prepare-{uuid4().hex}"
    test_dir.mkdir(parents=True, exist_ok=True)
    csv_path = test_dir / "multilingual_toxic_comments.csv"
    output_dir = test_dir / "data"

    pd.DataFrame(
        {
            "id": list(range(1, 11)),
            "comment_text": [
                "Helpful advice",
                "You are terrible",
                "What a great answer",
                "Such a toxic reply",
                "Very kind response",
                "Awful and rude",
                "Thanks for sharing",
                "Horrible behavior",
                "This is useful",
                "You idiot",
            ],
            "toxic": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "lang": ["en", "en", "hi", "hi", "es", "es", "fr", "fr", "de", "de"],
        }
    ).to_csv(csv_path, index=False)

    summary = prepare_dataset(
        source_csv=str(csv_path),
        output_dir=str(output_dir),
        dataset_name="multilingual_toxic_comments",
        validation_size=0.2,
        test_size=0.2,
        seed=7,
    )

    train_csv = output_dir / "processed" / "multilingual_toxic_comments" / "train.csv"
    validation_csv = output_dir / "processed" / "multilingual_toxic_comments" / "validation.csv"
    test_csv = output_dir / "processed" / "multilingual_toxic_comments" / "test.csv"
    summary_json = output_dir / "processed" / "multilingual_toxic_comments" / "summary.json"
    clean_csv = output_dir / "interim" / "multilingual_toxic_comments_clean.csv"

    assert clean_csv.exists()
    assert train_csv.exists()
    assert validation_csv.exists()
    assert test_csv.exists()
    assert summary["row_counts"]["clean"] == 10

    saved_summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert saved_summary["row_counts"]["train"] == 6
    assert saved_summary["row_counts"]["validation"] == 2
    assert saved_summary["row_counts"]["test"] == 2


def test_prepare_dataset_records_subtype_distribution():
    test_dir = Path("tests") / f"prepare-{uuid4().hex}"
    test_dir.mkdir(parents=True, exist_ok=True)
    csv_path = test_dir / "moderation_labels.csv"
    output_dir = test_dir / "data"

    pd.DataFrame(
        {
            "text": [
                "Helpful advice",
                "You idiot",
                "I will hurt you",
                "Those people are filthy",
                "Very kind response",
                "You are pathetic",
                "Thanks for sharing",
                "Watch your back",
                "This is useful",
                "Go away forever",
            ],
            "label": [
                "neutral",
                "abusive",
                "threat",
                "hate_targeted",
                "neutral",
                "abusive",
                "neutral",
                "threat",
                "neutral",
                "hate-targeted",
            ],
        }
    ).to_csv(csv_path, index=False)

    summary = prepare_dataset(
        source_csv=str(csv_path),
        output_dir=str(output_dir),
        dataset_name="multilingual_toxic_comments",
        text_column="text",
        label_column="label",
        validation_size=0.2,
        test_size=0.2,
        seed=5,
    )

    assert summary["available_label_modes"] == ["binary", "subtype_multilabel"]
    assert summary["subtype_distribution"] == {
        "abusive": 2,
        "hate_targeted": 2,
        "threat": 2,
    }
