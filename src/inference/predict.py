import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import logging

from src.data.labels import DEFAULT_BINARY_LABEL_NAMES, NEUTRAL_LABEL, SUBTYPE_LABEL_COLUMNS
from src.data.preprocessing import normalize_text

logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR_CANDIDATES = (
    PROJECT_ROOT / "artifacts" / "slm",
    PROJECT_ROOT / "artifacts" / "transformer",
)
EXIT_WORDS = {"exit", "quit"}


def _has_transformer_artifacts(model_dir: Path) -> bool:
    required_files = ("config.json", "tokenizer_config.json")
    return all((model_dir / file_name).exists() for file_name in required_files)


def _candidate_model_dirs():
    seen_paths = set()

    for candidate in DEFAULT_MODEL_DIR_CANDIDATES:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in seen_paths:
            continue
        seen_paths.add(resolved_candidate)
        yield resolved_candidate

    search_roots = []
    for candidate in DEFAULT_MODEL_DIR_CANDIDATES:
        candidate_parent = candidate.resolve().parent
        if candidate_parent not in search_roots:
            search_roots.append(candidate_parent)

    for search_root in search_roots:
        if not search_root.exists():
            continue
        for child in search_root.iterdir():
            if not child.is_dir():
                continue
            resolved_child = child.resolve()
            if resolved_child in seen_paths:
                continue
            seen_paths.add(resolved_child)
            yield resolved_child


def _model_artifact_mtime(model_dir: Path) -> float:
    timestamps = [model_dir.stat().st_mtime]
    for file_name in ("config.json", "tokenizer_config.json", "model_config.json"):
        artifact_path = model_dir / file_name
        if artifact_path.exists():
            timestamps.append(artifact_path.stat().st_mtime)
    return max(timestamps)


def _model_quality_score(model_dir: Path) -> Optional[float]:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None

    for metric_name in ("f1_macro", "eval_f1_macro", "f1_micro", "eval_f1_micro", "accuracy", "eval_accuracy"):
        metric_value = metrics.get(metric_name)
        if isinstance(metric_value, (int, float)):
            return float(metric_value)
    return None


def _model_selection_rank(model_dir: Path):
    quality_score = _model_quality_score(model_dir)
    return (
        quality_score is not None,
        quality_score if quality_score is not None else float("-inf"),
        _model_artifact_mtime(model_dir),
    )


def resolve_model_dir(model_dir: Optional[str] = None) -> Path:
    """Resolve an explicit model path or fall back to the newest saved transformer model."""
    if model_dir:
        resolved = Path(model_dir).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Model directory not found: {resolved}")
        return resolved

    valid_candidates = [
        candidate
        for candidate in _candidate_model_dirs()
        if _has_transformer_artifacts(candidate)
    ]
    if valid_candidates:
        return max(valid_candidates, key=_model_selection_rank)

    searched = ", ".join(str(path) for path in DEFAULT_MODEL_DIR_CANDIDATES)
    raise FileNotFoundError(
        "No saved transformer model was found automatically. "
        f"Train a model first or pass --model-dir. Searched: {searched}"
    )


def normalize_prediction_label(label: Any) -> Any:
    """Map raw Hugging Face labels to friendly class names when possible."""
    label_map = {
        "LABEL_0": "non-toxic",
        "LABEL_1": "toxic",
    }
    return label_map.get(str(label), label)


def _normalize_pipeline_output(prediction: Any):
    if isinstance(prediction, list) and len(prediction) == 1 and isinstance(prediction[0], list):
        return prediction[0]
    return prediction


def _normalize_scored_prediction_list(prediction):
    normalized_scores = []
    for item in prediction:
        if not isinstance(item, dict) or "label" not in item:
            continue
        normalized_scores.append(
            {
                "label": normalize_prediction_label(item["label"]),
                "score": float(item.get("score", 0.0)),
            }
        )
    normalized_scores.sort(key=lambda item: item["score"], reverse=True)
    return normalized_scores


def _ensure_model_label_mapping(hf_model) -> None:
    id2label = getattr(hf_model.config, "id2label", {}) or {}
    normalized_values = {str(value) for value in id2label.values()}
    default_values = {f"LABEL_{index}" for index in range(getattr(hf_model.config, "num_labels", 0))}
    if getattr(hf_model.config, "num_labels", 0) == 2 and normalized_values.issubset(default_values):
        hf_model.config.id2label = {
            0: DEFAULT_BINARY_LABEL_NAMES[0],
            1: DEFAULT_BINARY_LABEL_NAMES[1],
        }
        hf_model.config.label2id = {
            DEFAULT_BINARY_LABEL_NAMES[0]: 0,
            DEFAULT_BINARY_LABEL_NAMES[1]: 1,
        }


def _is_multilabel_pipeline(model) -> bool:
    model_config = getattr(getattr(model, "model", None), "config", None)
    if not model_config:
        return False
    return getattr(model_config, "problem_type", "") == "multi_label_classification"


def _is_binary_score_payload(prediction) -> bool:
    labels = {item["label"] for item in prediction}
    return set(DEFAULT_BINARY_LABEL_NAMES).issubset(labels)


def load_transformer_pipeline(model_dir: str, device: int = -1):
    """Load a saved Hugging Face text classification pipeline."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    _ensure_model_label_mapping(hf_model)
    return pipeline(
        "text-classification",
        model=hf_model,
        tokenizer=tokenizer,
        truncation=True,
        device=device,
    )


def predict_result(model, text: str) -> Dict[str, Any]:
    """Run inference and return a consistent result payload."""
    normalized_text = normalize_text(text)

    if callable(model):
        try:
            prediction = model(normalized_text, top_k=None)
        except TypeError:
            prediction = model(normalized_text)
        prediction = _normalize_pipeline_output(prediction)
        if _is_multilabel_pipeline(model) and isinstance(prediction, list):
            normalized_scores = _normalize_scored_prediction_list(prediction)
            predicted_labels = [
                item["label"]
                for item in normalized_scores
                if item["label"] in SUBTYPE_LABEL_COLUMNS and item["score"] >= 0.5
            ]
            if not predicted_labels:
                predicted_labels = [NEUTRAL_LABEL]
            return {
                "label": ", ".join(predicted_labels),
                "labels": predicted_labels,
                "score": normalized_scores[0]["score"] if normalized_scores else None,
                "scores": {
                    item["label"]: float(item["score"])
                    for item in normalized_scores
                    if item["label"] in SUBTYPE_LABEL_COLUMNS
                },
                "raw_prediction": normalized_scores,
            }
        if isinstance(prediction, list) and prediction:
            normalized_scores = _normalize_scored_prediction_list(prediction)
            if normalized_scores:
                top_result = normalized_scores[0]
                result = {
                    "label": top_result["label"],
                    "score": top_result["score"],
                    "raw_prediction": normalized_scores,
                }
                if _is_binary_score_payload(normalized_scores):
                    result["scores"] = {
                        item["label"]: item["score"]
                        for item in normalized_scores
                        if item["label"] in DEFAULT_BINARY_LABEL_NAMES
                    }
                return result
            first_result = prediction[0]
            if isinstance(first_result, dict) and "label" in first_result:
                raw_label = first_result["label"]
                return {
                    "label": normalize_prediction_label(raw_label),
                    "score": first_result.get("score"),
                    "raw_prediction": first_result,
                }
            return {
                "label": normalize_prediction_label(first_result),
                "score": None,
                "raw_prediction": first_result,
            }
        return {
            "label": normalize_prediction_label(prediction),
            "score": None,
            "raw_prediction": prediction,
        }

    if hasattr(model, "predict"):
        raw_prediction = model.predict([normalized_text])[0]
        return {
            "label": normalize_prediction_label(raw_prediction),
            "score": None,
            "raw_prediction": raw_prediction,
        }

    raise TypeError("Unsupported model type for predict_text.")


def predict_text(model, text: str):
    """Return the predicted label only."""
    return predict_result(model, text)["label"]


def format_prediction(result: Dict[str, Any]) -> str:
    """Convert a prediction payload into user-friendly output."""
    label = result["label"]
    subtype_scores = result.get("scores")
    if subtype_scores and any(label_name in SUBTYPE_LABEL_COLUMNS for label_name in subtype_scores):
        formatted_scores = ", ".join(
            f"{subtype}={score:.4f}"
            for subtype, score in sorted(subtype_scores.items())
        )
        return f"Prediction: {label} | subtype_scores: {formatted_scores}"
    score = result.get("score")
    if score is None:
        return f"Prediction: {label}"
    return f"Prediction: {label} (confidence: {score:.4f})"


def run_interactive_session(model) -> None:
    """Prompt repeatedly for text until the user exits."""
    print("Type the text to predict and press Enter.")
    print("Press Enter on a blank line, or type exit, to close.")

    while True:
        text = input("\nText: ").strip()
        if not text or text.lower() in EXIT_WORDS:
            print("Closing predictor.")
            return

        result = predict_result(model, text)
        print(format_prediction(result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run toxic comment prediction.")
    parser.add_argument("--model-dir", help="Path to the saved model directory.")
    parser.add_argument("--text", help="Text to predict in one-shot mode.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open an interactive prompt instead of requiring --text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = resolve_model_dir(args.model_dir)
    model = load_transformer_pipeline(str(model_dir))

    if args.interactive or not args.text:
        print(f"Using model: {model_dir}")
        run_interactive_session(model)
        return

    result = predict_result(model, args.text)
    print(format_prediction(result))


if __name__ == "__main__":
    main()
