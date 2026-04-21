import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data.prepare import prepare_dataset
from src.training.train import _resolve_data_paths as resolve_baseline_data_paths
from src.training.train import train_baseline
from src.training.train_transformer import _build_train_eval_frames
from src.training.train_transformer import _resolve_data_paths as resolve_transformer_data_paths
from src.training.train_transformer import _sample_frame


def test_train_baseline_saves_model_and_metrics():
    test_dir = Path("tests") / f"test-{uuid4().hex}"
    test_dir.mkdir(parents=True, exist_ok=True)
    csv_path = test_dir / "sample.csv"
    output_dir = test_dir / "artifacts"

    df = pd.DataFrame(
        {
            "text": [
                "you are horrible",
                "thank you so much",
                "idiot comment",
                "have a nice day",
                "awful behavior",
                "this was helpful",
            ],
            "label": [1, 0, 1, 0, 1, 0],
        }
    )
    df.to_csv(csv_path, index=False)

    metrics = train_baseline(
        data_csv=str(csv_path),
        text_column="text",
        label_column="label",
        test_size=0.33,
        output_dir=str(output_dir),
        seed=42,
    )

    assert set(metrics) == {"accuracy", "precision_macro", "recall_macro", "f1_macro"}
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()

    saved_metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics["accuracy"] >= 0.0


def test_train_baseline_accepts_prepared_train_and_eval_splits():
    test_dir = Path("tests") / f"prepared-{uuid4().hex}"
    raw_dir = test_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "multilingual_toxic_comments.csv"
    output_dir = test_dir / "model"

    pd.DataFrame(
        {
            "id": list(range(1, 11)),
            "comment_text": [
                "helpful answer",
                "you are disgusting",
                "thank you friend",
                "idiot comment",
                "what a kind note",
                "terrible message",
                "great support",
                "awful reply",
                "very useful",
                "pathetic troll",
            ],
            "toxic": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "lang": ["en", "en", "hi", "hi", "es", "es", "fr", "fr", "de", "de"],
        }
    ).to_csv(csv_path, index=False)

    prepare_dataset(
        source_csv=str(csv_path),
        output_dir=str(test_dir / "data"),
        dataset_name="multilingual_toxic_comments",
        validation_size=0.2,
        test_size=0.2,
        seed=21,
    )

    processed_dir = test_dir / "data" / "processed" / "multilingual_toxic_comments"
    metrics = train_baseline(
        train_csv=str(processed_dir / "train.csv"),
        eval_csv=str(processed_dir / "test.csv"),
        dataset_name="multilingual_toxic_comments",
        output_dir=str(output_dir),
        seed=21,
    )

    assert set(metrics) == {"accuracy", "precision_macro", "recall_macro", "f1_macro"}
    assert (output_dir / "model.joblib").exists()


def test_baseline_config_prefers_validation_split():
    test_dir = Path("tests") / f"paths-{uuid4().hex}"
    test_dir.mkdir(parents=True, exist_ok=True)
    train_csv = test_dir / "train.csv"
    validation_csv = test_dir / "validation.csv"
    test_csv = test_dir / "test.csv"
    for path in (train_csv, validation_csv, test_csv):
        path.write_text("text,label\nhello,0\n", encoding="utf-8")

    args = type("Args", (), {"data_csv": None, "train_csv": None, "eval_csv": None})()
    data_csv, resolved_train_csv, resolved_eval_csv = resolve_baseline_data_paths(
        args,
        {
            "train_csv": str(train_csv),
            "validation_csv": str(validation_csv),
            "test_csv": str(test_csv),
        },
    )

    assert data_csv is None
    assert resolved_train_csv == str(train_csv)
    assert resolved_eval_csv == str(validation_csv)


def test_transformer_config_prefers_validation_split():
    test_dir = Path("tests") / f"transformer-paths-{uuid4().hex}"
    test_dir.mkdir(parents=True, exist_ok=True)
    train_csv = test_dir / "train.csv"
    validation_csv = test_dir / "validation.csv"
    test_csv = test_dir / "test.csv"
    for path in (train_csv, validation_csv, test_csv):
        path.write_text("text,label\nhello,0\n", encoding="utf-8")

    args = type("Args", (), {"data_csv": None, "train_csv": None, "eval_csv": None})()
    data_csv, resolved_train_csv, resolved_eval_csv = resolve_transformer_data_paths(
        args,
        {
            "train_csv": str(train_csv),
            "validation_csv": str(validation_csv),
            "test_csv": str(test_csv),
        },
    )

    assert data_csv is None
    assert resolved_train_csv == [str(train_csv)]
    assert resolved_eval_csv == [str(validation_csv)]


def test_transformer_config_supports_multiple_configured_splits():
    test_dir = Path("tests") / f"transformer-multi-paths-{uuid4().hex}"
    test_dir.mkdir(parents=True, exist_ok=True)
    train_csv_a = test_dir / "train-a.csv"
    train_csv_b = test_dir / "train-b.csv"
    validation_csv_a = test_dir / "validation-a.csv"
    validation_csv_b = test_dir / "validation-b.csv"
    for path in (train_csv_a, train_csv_b, validation_csv_a, validation_csv_b):
        path.write_text("text,label\nhello,0\n", encoding="utf-8")

    args = type("Args", (), {"data_csv": None, "train_csv": None, "eval_csv": None})()
    data_csv, resolved_train_csv, resolved_eval_csv = resolve_transformer_data_paths(
        args,
        {
            "train_csvs": [str(train_csv_a), str(train_csv_b)],
            "validation_csvs": [str(validation_csv_a), str(validation_csv_b)],
        },
    )

    assert data_csv is None
    assert resolved_train_csv == [str(train_csv_a), str(train_csv_b)]
    assert resolved_eval_csv == [str(validation_csv_a), str(validation_csv_b)]


def test_sample_frame_caps_rows_without_changing_requested_size():
    frame = pd.DataFrame({"text": [f"row-{idx}" for idx in range(10)], "label": [idx % 2 for idx in range(10)]})

    sampled_frame = _sample_frame(frame, max_rows=4, seed=11)

    assert len(sampled_frame) == 4
    assert sampled_frame.index.tolist() == [0, 1, 2, 3]


def test_build_train_eval_frames_merges_multiple_csvs():
    test_dir = Path("tests") / f"transformer-merge-{uuid4().hex}"
    test_dir.mkdir(parents=True, exist_ok=True)
    train_csv_a = test_dir / "train-a.csv"
    train_csv_b = test_dir / "train-b.csv"
    eval_csv = test_dir / "eval.csv"

    pd.DataFrame(
        {
            "text": ["you are awful", "thank you"],
            "label": [1, 0],
        }
    ).to_csv(train_csv_a, index=False)
    pd.DataFrame(
        {
            "text": ["you are awful", "idiot comment"],
            "label": [1, 1],
        }
    ).to_csv(train_csv_b, index=False)
    pd.DataFrame(
        {
            "text": ["have a nice day", "terrible reply"],
            "label": [0, 1],
        }
    ).to_csv(eval_csv, index=False)

    train_df, eval_df = _build_train_eval_frames(
        data_csv=None,
        train_csv=[str(train_csv_a), str(train_csv_b)],
        eval_csv=[str(eval_csv)],
        text_column="text",
        label_column="label",
        language_column=None,
        id_column=None,
        dataset_name=None,
        test_size=0.2,
        seed=5,
    )

    assert len(train_df) == 3
    assert sorted(train_df["text"].tolist()) == ["idiot comment", "thank you", "you are awful"]
    assert len(eval_df) == 2
