import os
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from src.inference.predict import PROJECT_ROOT, format_prediction, predict_result, predict_text, resolve_model_dir
from src.models.transformer import resolve_transformer_config


def test_resolve_transformer_config_defaults_to_slm():
    config = resolve_transformer_config()

    assert config.key == "slm"
    assert config.hf_model_name == "distilbert-base-multilingual-cased"
    assert config.max_length == 256


def test_resolve_transformer_config_supports_xlmr_alias():
    config = resolve_transformer_config(model_name="xlm-r", max_length=192)

    assert config.key == "xlm-r"
    assert config.hf_model_name == "xlm-roberta-base"
    assert config.max_length == 192


def test_resolve_transformer_config_supports_muril_alias():
    config = resolve_transformer_config(model_name="muril", max_length=192)

    assert config.key == "muril"
    assert config.hf_model_name == "google/muril-base-cased"
    assert config.max_length == 192


def test_resolve_transformer_config_rejects_unknown_model():
    with pytest.raises(ValueError):
        resolve_transformer_config(model_name="unknown-transformer")


def test_predict_text_supports_sklearn_style_models():
    class DummySklearnModel:
        def predict(self, texts):
            assert texts == ["Hello <USER>"]
            return [1]

    assert predict_text(DummySklearnModel(), "Hello @friend") == 1


def test_predict_text_supports_callable_transformer_like_pipelines():
    class DummyPipeline:
        def __call__(self, text):
            assert text == "Visit <URL>"
            return [{"label": "LABEL_1", "score": 0.99}]

    assert predict_text(DummyPipeline(), "Visit https://example.com") == "toxic"


def test_predict_result_returns_score_for_transformer_like_pipelines():
    class DummyPipeline:
        def __call__(self, text, top_k=None):
            assert top_k is None
            assert text == "Visit <URL>"
            return [
                {"label": "LABEL_0", "score": 0.75},
                {"label": "LABEL_1", "score": 0.25},
            ]

    result = predict_result(DummyPipeline(), "Visit https://example.com")

    assert result == {
        "label": "non-toxic",
        "score": 0.75,
        "scores": {
            "non-toxic": 0.75,
            "toxic": 0.25,
        },
        "raw_prediction": [
            {"label": "non-toxic", "score": 0.75},
            {"label": "toxic", "score": 0.25},
        ],
    }


def test_predict_result_normalizes_multiword_hinglish_abuse_before_inference():
    class DummyPipeline:
        def __call__(self, text, top_k=None):
            assert top_k is None
            assert text == "bhen ke lode motherfucker"
            return [
                {"label": "LABEL_1", "score": 0.9},
                {"label": "LABEL_0", "score": 0.1},
            ]

    result = predict_result(DummyPipeline(), "bhen ke lode")

    assert result["label"] == "toxic"
    assert result["score"] == 0.9


def test_predict_result_deobfuscates_leetspeak_hate_slurs_before_inference():
    class DummyPipeline:
        def __call__(self, text, top_k=None):
            assert top_k is None
            assert text == "nigger"
            return [
                {"label": "LABEL_1", "score": 0.91},
                {"label": "LABEL_0", "score": 0.09},
            ]

    result = predict_result(DummyPipeline(), "n1gger")

    assert result["label"] == "toxic"
    assert result["score"] == 0.91


def test_predict_result_enriches_implicit_toxic_phrase_before_inference():
    class DummyPipeline:
        def __call__(self, text, top_k=None):
            assert top_k is None
            assert text == "you belong in the kitchen sexist"
            return [
                {"label": "LABEL_1", "score": 0.84},
                {"label": "LABEL_0", "score": 0.16},
            ]

    result = predict_result(DummyPipeline(), "you belong to the kitchen")

    assert result["label"] == "toxic"
    assert result["score"] == 0.84


def test_predict_result_enriches_targeted_racial_slur_before_inference():
    class DummyPipeline:
        def __call__(self, text, top_k=None):
            assert top_k is None
            assert text == "pajeet racist slur"
            return [
                {"label": "LABEL_1", "score": 0.88},
                {"label": "LABEL_0", "score": 0.12},
            ]

    result = predict_result(DummyPipeline(), "pajeet")

    assert result["label"] == "toxic"
    assert result["score"] == 0.88


def test_predict_result_enriches_hinglish_insult_phrase_before_inference():
    class DummyPipeline:
        def __call__(self, text, top_k=None):
            assert top_k is None
            assert text == "tum pagal ho idiot"
            return [
                {"label": "LABEL_1", "score": 0.87},
                {"label": "LABEL_0", "score": 0.13},
            ]

    result = predict_result(DummyPipeline(), "tum pagal ho")

    assert result["label"] == "toxic"
    assert result["score"] == 0.87


def test_predict_result_supports_multilabel_transformer_like_pipelines():
    class DummyConfig:
        problem_type = "multi_label_classification"

    class DummyModel:
        config = DummyConfig()

    class DummyPipeline:
        model = DummyModel()

        def __call__(self, text, top_k=None):
            assert top_k is None
            assert text == "Get lost"
            return [
                {"label": "abusive", "score": 0.91},
                {"label": "hate_targeted", "score": 0.22},
                {"label": "threat", "score": 0.71},
            ]

    result = predict_result(DummyPipeline(), "Get lost")

    assert result["label"] == "abusive, threat"
    assert result["labels"] == ["abusive", "threat"]
    assert result["scores"]["abusive"] == 0.91
    assert result["scores"]["threat"] == 0.71


def _write_dummy_transformer_artifacts(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for file_name in ("config.json", "tokenizer_config.json"):
        (model_dir / file_name).write_text("{}", encoding="utf-8")


def test_resolve_model_dir_prefers_newest_valid_saved_model(monkeypatch):
    test_root = PROJECT_ROOT / "tests" / f"predict-model-dir-{uuid4().hex}"
    artifacts_dir = test_root / "artifacts"
    older_model_dir = artifacts_dir / "transformer"
    newer_model_dir = artifacts_dir / "combined_hate_speech_slm"
    invalid_model_dir = artifacts_dir / "baseline"

    try:
        _write_dummy_transformer_artifacts(older_model_dir)
        _write_dummy_transformer_artifacts(newer_model_dir)
        invalid_model_dir.mkdir(parents=True, exist_ok=True)

        older_timestamp = 1_700_000_000
        newer_timestamp = older_timestamp + 60
        for artifact_path in older_model_dir.iterdir():
            os.utime(artifact_path, (older_timestamp, older_timestamp))
        for artifact_path in newer_model_dir.iterdir():
            os.utime(artifact_path, (newer_timestamp, newer_timestamp))

        monkeypatch.setattr("src.inference.predict.DEFAULT_MODEL_DIR_CANDIDATES", (older_model_dir,))

        assert resolve_model_dir() == newer_model_dir.resolve()
    finally:
        shutil.rmtree(test_root, ignore_errors=True)


def test_resolve_model_dir_prefers_better_scoring_model_over_newer_one(monkeypatch):
    test_root = PROJECT_ROOT / "tests" / f"predict-model-quality-{uuid4().hex}"
    artifacts_dir = test_root / "artifacts"
    better_model_dir = artifacts_dir / "transformer"
    newer_weaker_model_dir = artifacts_dir / "combined_hate_speech_slm"

    try:
        _write_dummy_transformer_artifacts(better_model_dir)
        _write_dummy_transformer_artifacts(newer_weaker_model_dir)

        (better_model_dir / "metrics.json").write_text('{"f1_macro": 0.91}', encoding="utf-8")
        (newer_weaker_model_dir / "metrics.json").write_text('{"f1_macro": 0.72}', encoding="utf-8")

        older_timestamp = 1_700_000_000
        newer_timestamp = older_timestamp + 60
        for artifact_path in better_model_dir.iterdir():
            os.utime(artifact_path, (older_timestamp, older_timestamp))
        for artifact_path in newer_weaker_model_dir.iterdir():
            os.utime(artifact_path, (newer_timestamp, newer_timestamp))

        monkeypatch.setattr("src.inference.predict.DEFAULT_MODEL_DIR_CANDIDATES", (better_model_dir,))

        assert resolve_model_dir() == better_model_dir.resolve()
    finally:
        shutil.rmtree(test_root, ignore_errors=True)


def test_resolve_model_dir_accepts_explicit_existing_path():
    model_dir = PROJECT_ROOT / "artifacts" / "transformer"

    assert resolve_model_dir(str(model_dir)) == model_dir.resolve()


def test_format_prediction_hides_binary_class_scores_from_default_output():
    formatted = format_prediction(
        {
            "label": "toxic",
            "score": 0.9,
            "scores": {
                "non-toxic": 0.1,
                "toxic": 0.9,
            },
        }
    )

    assert formatted == "Prediction: toxic (confidence: 0.9000)"
