import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from src.data.dataset import load_dataset
from src.data.labels import (
    BINARY_LABEL_MODE,
    LABEL_MODE_CHOICES,
    SUBTYPE_LABEL_COLUMNS,
    SUBTYPE_MULTILABEL_MODE,
    label_names_for_mode,
)
from src.data.preprocessing import normalize_text
from src.evaluation.metrics import classification_metrics, multilabel_classification_metrics
from src.models.transformer import build_transformer_components, resolve_transformer_config
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


def _coerce_csv_paths(csv_paths: Optional[str | Sequence[str]]) -> list[str]:
    if csv_paths is None:
        return []
    if isinstance(csv_paths, (str, os.PathLike)):
        normalized_paths = [str(csv_paths)]
    else:
        normalized_paths = [str(path) for path in csv_paths if path]
    return list(dict.fromkeys(normalized_paths))


def _resolve_configured_csv_paths(config: Dict[str, Any], *keys: str) -> list[str]:
    for key in keys:
        if key not in config:
            continue
        resolved_paths = _coerce_csv_paths(config.get(key))
        if resolved_paths:
            return resolved_paths
    return []


def _all_paths_exist(csv_paths: Sequence[str]) -> bool:
    return all(Path(csv_path).exists() for csv_path in csv_paths)


def _merge_dataset_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("At least one dataset frame is required.")
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)

    merged_frame = pd.concat(frames, ignore_index=True)
    dedupe_columns = ["text", "label"]
    dedupe_columns.extend([column for column in SUBTYPE_LABEL_COLUMNS if column in merged_frame.columns])
    if "language" in merged_frame.columns:
        dedupe_columns.append("language")
    return merged_frame.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)


def _load_and_normalize_datasets(
    csv_paths: Optional[str | Sequence[str]],
    text_column: Optional[str],
    label_column: Optional[str],
    language_column: Optional[str],
    id_column: Optional[str],
    dataset_name: Optional[str],
):
    resolved_paths = _coerce_csv_paths(csv_paths)
    if not resolved_paths:
        raise ValueError("Provide at least one CSV path.")

    frames = [
        _load_and_normalize_dataset(
            csv_path,
            text_column=text_column,
            label_column=label_column,
            language_column=language_column,
            id_column=id_column,
            dataset_name=dataset_name,
        )
        for csv_path in resolved_paths
    ]
    return _merge_dataset_frames(frames)


def _build_train_eval_frames(
    data_csv: Optional[str | Sequence[str]],
    train_csv: Optional[str | Sequence[str]],
    eval_csv: Optional[str | Sequence[str]],
    text_column: Optional[str],
    label_column: Optional[str],
    language_column: Optional[str],
    id_column: Optional[str],
    dataset_name: Optional[str],
    test_size: float,
    seed: int,
):
    from sklearn.model_selection import train_test_split

    data_paths = _coerce_csv_paths(data_csv)
    train_paths = _coerce_csv_paths(train_csv)
    eval_paths = _coerce_csv_paths(eval_csv)

    source_paths = data_paths or train_paths
    if not source_paths:
        raise ValueError("Provide either data_csv or train_csv.")

    if eval_paths:
        train_df = _load_and_normalize_datasets(
            source_paths,
            text_column=text_column,
            label_column=label_column,
            language_column=language_column,
            id_column=id_column,
            dataset_name=dataset_name,
        )
        eval_df = _load_and_normalize_datasets(
            eval_paths,
            text_column=text_column,
            label_column=label_column,
            language_column=language_column,
            id_column=id_column,
            dataset_name=dataset_name,
        )
        return train_df, eval_df

    df = _load_and_normalize_datasets(
        source_paths,
        text_column=text_column,
        label_column=label_column,
        language_column=language_column,
        id_column=id_column,
        dataset_name=dataset_name,
    )
    label_counts = df["label"].value_counts()
    stratify = df["label"] if df["label"].nunique() > 1 and label_counts.min() >= 2 else None
    try:
        train_df, eval_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_df, eval_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def _resolve_data_paths(args: argparse.Namespace, data_config: Dict[str, Any]):
    configured_raw_csv = _resolve_configured_csv_paths(data_config, "raw_csvs", "raw_csv")
    configured_train_csv = _resolve_configured_csv_paths(data_config, "train_csvs", "train_csv")
    configured_eval_csv = _resolve_configured_csv_paths(
        data_config,
        "validation_csvs",
        "validation_csv",
        "test_csvs",
        "test_csv",
    )

    explicit_data_csv = _coerce_csv_paths(args.data_csv)
    explicit_train_csv = _coerce_csv_paths(args.train_csv)
    explicit_eval_csv = _coerce_csv_paths(args.eval_csv)

    use_explicit_prepared_splits = bool(explicit_train_csv)
    use_configured_prepared_splits = bool(
        configured_train_csv
        and configured_eval_csv
        and _all_paths_exist(configured_train_csv)
        and _all_paths_exist(configured_eval_csv)
    )

    if use_explicit_prepared_splits:
        return None, explicit_train_csv, explicit_eval_csv or None

    if use_configured_prepared_splits:
        return None, configured_train_csv, explicit_eval_csv or configured_eval_csv

    resolved_data_csv = explicit_data_csv or configured_raw_csv
    resolved_eval_csv = explicit_eval_csv or None
    return resolved_data_csv or None, None, resolved_eval_csv


def _resolve_task_labels(df, label_mode: str) -> Sequence[str]:
    if label_mode == BINARY_LABEL_MODE:
        return ("label",)
    if label_mode == SUBTYPE_MULTILABEL_MODE:
        missing_columns = [column for column in SUBTYPE_LABEL_COLUMNS if column not in df.columns]
        if missing_columns:
            missing_list = ", ".join(missing_columns)
            raise ValueError(
                "Subtype multilabel training requires prepared data with subtype columns. "
                f"Missing: {missing_list}"
            )
        return SUBTYPE_LABEL_COLUMNS
    supported = ", ".join(LABEL_MODE_CHOICES)
    raise ValueError(f"Unsupported label mode '{label_mode}'. Supported values: {supported}")


def _frame_to_dataset(df, label_mode: str, task_labels: Sequence[str]):
    from datasets import Dataset

    dataset_frame = df[["text"]].reset_index(drop=True).copy()
    if label_mode == BINARY_LABEL_MODE:
        dataset_frame["label"] = df["label"].reset_index(drop=True)
    else:
        dataset_frame["labels"] = (
            df[list(task_labels)]
            .reset_index(drop=True)
            .astype(np.float32)
            .values
            .tolist()
        )
    return Dataset.from_pandas(dataset_frame, preserve_index=False)


def _sample_frame(df, max_rows: Optional[int], seed: int):
    if not max_rows or max_rows >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def _resolve_device_type() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return "mps"
    return "cpu"


def _default_dataloader_workers(device_type: str) -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 0
    if device_type == "cpu":
        return min(4, max(1, cpu_count // 3))
    return min(8, max(2, cpu_count // 2))


def _configure_torch_runtime(device_type: str) -> None:
    import torch

    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except RuntimeError:
            pass

    if device_type != "cpu":
        return
    if os.environ.get("OMP_NUM_THREADS") or os.environ.get("MKL_NUM_THREADS"):
        return

    target_threads = max(1, (os.cpu_count() or 1) - 1)
    if torch.get_num_threads() != target_threads:
        torch.set_num_threads(target_threads)

    target_interop_threads = min(4, target_threads)
    try:
        if torch.get_num_interop_threads() != target_interop_threads:
            torch.set_num_interop_threads(target_interop_threads)
    except RuntimeError:
        # PyTorch only allows changing inter-op threads before the pool is initialized.
        pass


def train_transformer(
    model_name: str = "slm",
    data_csv: Optional[str | Sequence[str]] = None,
    train_csv: Optional[str | Sequence[str]] = None,
    eval_csv: Optional[str | Sequence[str]] = None,
    dataset_name: Optional[str] = None,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    language_column: Optional[str] = None,
    id_column: Optional[str] = None,
    label_mode: str = BINARY_LABEL_MODE,
    max_length: int = 256,
    num_labels: int = 2,
    batch_size: int = 8,
    eval_batch_size: Optional[int] = None,
    learning_rate: float = 2e-5,
    epochs: int = 3,
    weight_decay: float = 0.01,
    test_size: float = 0.2,
    output_dir: str = "artifacts/transformer",
    dataloader_workers: Optional[int] = None,
    tokenization_workers: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    logging_steps: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """Train and save a transformer text classifier."""
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    set_seed(seed)
    if label_mode not in LABEL_MODE_CHOICES:
        supported = ", ".join(LABEL_MODE_CHOICES)
        raise ValueError(f"Unsupported label mode '{label_mode}'. Supported values: {supported}")

    resolved_num_labels = len(SUBTYPE_LABEL_COLUMNS) if label_mode == SUBTYPE_MULTILABEL_MODE else num_labels
    config = resolve_transformer_config(
        model_name=model_name,
        max_length=max_length,
        num_labels=resolved_num_labels,
    )
    train_df, eval_df = _build_train_eval_frames(
        data_csv=data_csv,
        train_csv=train_csv,
        eval_csv=eval_csv,
        text_column=text_column,
        label_column=label_column,
        language_column=language_column,
        id_column=id_column,
        dataset_name=dataset_name,
        test_size=test_size,
        seed=seed,
    )
    train_df = _sample_frame(train_df, max_train_samples, seed)
    eval_df = _sample_frame(eval_df, max_eval_samples, seed)
    task_labels = _resolve_task_labels(train_df, label_mode)
    label_names = label_names_for_mode(label_mode)

    device_type = _resolve_device_type()
    _configure_torch_runtime(device_type)

    tokenizer, model = build_transformer_components(config)
    model.config.id2label = {index: label_name for index, label_name in enumerate(label_names)}
    model.config.label2id = {label_name: index for index, label_name in enumerate(label_names)}
    model.config.problem_type = (
        "multi_label_classification"
        if label_mode == SUBTYPE_MULTILABEL_MODE
        else "single_label_classification"
    )
    resolved_eval_batch_size = eval_batch_size or min(max(batch_size * 2, batch_size), 32)
    resolved_dataloader_workers = (
        _default_dataloader_workers(device_type)
        if dataloader_workers is None
        else max(dataloader_workers, 0)
    )
    resolved_tokenization_workers = (
        tokenization_workers if tokenization_workers and tokenization_workers > 1 else None
    )

    train_dataset = _frame_to_dataset(train_df, label_mode=label_mode, task_labels=task_labels)
    eval_dataset = _frame_to_dataset(eval_df, label_mode=label_mode, task_labels=task_labels)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_length,
            return_length=True,
        )

    tokenized_train_dataset = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        num_proc=resolved_tokenization_workers,
        desc="Tokenizing training split",
    )
    tokenized_eval_dataset = eval_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        num_proc=resolved_tokenization_workers,
        desc="Tokenizing evaluation split",
    )
    
    # Create data collator with tokenizer for compatibility across transformers versions
    try:
        # Try newer API first (transformers >= 4.25)
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 if device_type == "cuda" else None,
        )
    except TypeError:
        # Fall back to older API if tokenizer param not supported
        data_collator = DataCollatorWithPadding(
            pad_to_multiple_of=8 if device_type == "cuda" else None,
        )

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        if isinstance(logits, tuple):
            logits = logits[0]
        if label_mode == SUBTYPE_MULTILABEL_MODE:
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            predictions = (probabilities >= 0.5).astype(int)
            return multilabel_classification_metrics(labels, predictions)
        predictions = np.argmax(logits, axis=-1)
        return classification_metrics(labels, predictions)

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Runtime settings: "
        f"device={device_type}, train_samples={len(train_df)}, eval_samples={len(eval_df)}, "
        f"train_batch_size={batch_size}, eval_batch_size={resolved_eval_batch_size}, "
        f"dataloader_workers={resolved_dataloader_workers}, max_length={config.max_length}, "
        f"label_mode={label_mode}"
    )

    training_arguments_signature = inspect.signature(TrainingArguments.__init__)

    def build_training_args(worker_count: int) -> TrainingArguments:
        training_arg_kwargs = {
            "output_dir": str(artifact_dir),
            "learning_rate": learning_rate,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": resolved_eval_batch_size,
            "num_train_epochs": epochs,
            "weight_decay": weight_decay,
            "save_strategy": "epoch",
            "logging_strategy": "steps",
            "logging_steps": logging_steps,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1_macro",
            "greater_is_better": True,
            "report_to": "none",
            "save_total_limit": 1,
            "seed": seed,
        }
        if "evaluation_strategy" in training_arguments_signature.parameters:
            training_arg_kwargs["evaluation_strategy"] = "epoch"
        else:
            training_arg_kwargs["eval_strategy"] = "epoch"

        optional_training_args = {
            "dataloader_num_workers": worker_count,
            "dataloader_pin_memory": device_type == "cuda",
            "group_by_length": True,
            "length_column_name": "length",
            "skip_memory_metrics": True,
            "use_cpu": device_type == "cpu",
        }
        if worker_count > 0:
            optional_training_args["dataloader_persistent_workers"] = True
            optional_training_args["dataloader_prefetch_factor"] = 4

        for argument_name, argument_value in optional_training_args.items():
            if argument_name in training_arguments_signature.parameters:
                training_arg_kwargs[argument_name] = argument_value

        return TrainingArguments(**training_arg_kwargs)

    def build_trainer(worker_count: int) -> Trainer:
        # Trainer only accepts specific parameters
        # Do NOT pass tokenizer, data_collator should handle padding
        trainer_kwargs = {
            "model": model,
            "args": build_training_args(worker_count),
            "train_dataset": tokenized_train_dataset,
            "eval_dataset": tokenized_eval_dataset,
            "data_collator": data_collator,
            "compute_metrics": compute_metrics,
        }
        return Trainer(**trainer_kwargs)

    trainer = build_trainer(resolved_dataloader_workers)
    try:
        trainer.train()
    except PermissionError:
        if resolved_dataloader_workers <= 0:
            raise
        print(
            "Multiprocessing data loading is unavailable in this environment. "
            "Retrying with dataloader_workers=0."
        )
        trainer = build_trainer(0)
        trainer.train()
    metrics = trainer.evaluate()
    metric_keys = {
        "eval_accuracy",
        "eval_precision_macro",
        "eval_recall_macro",
        "eval_f1_macro",
    }
    if label_mode == SUBTYPE_MULTILABEL_MODE:
        metric_keys.update(
            {
                "eval_precision_micro",
                "eval_recall_micro",
                "eval_f1_micro",
                "eval_hamming_loss",
            }
        )
    cleaned_metrics = {
        key.replace("eval_", ""): float(value)
        for key, value in metrics.items()
        if key in metric_keys
    }

    trainer.save_model(str(artifact_dir))
    tokenizer.save_pretrained(str(artifact_dir))
    with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(cleaned_metrics, file, indent=2)
    with (artifact_dir / "model_config.json").open("w", encoding="utf-8") as file:
        model_metadata = config.to_dict()
        model_metadata["label_mode"] = label_mode
        model_metadata["label_names"] = list(label_names)
        json.dump(model_metadata, file, indent=2)

    return cleaned_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transformer toxic comment classifier.")
    parser.add_argument("--config", default="experiments/configs/transformer.yaml")
    parser.add_argument("--model-name", help="Transformer alias or Hugging Face model name preset.")
    parser.add_argument(
        "--data-csv",
        action="append",
        help="Path to a CSV dataset that will be split for training and evaluation. Repeat to merge multiple files.",
    )
    parser.add_argument(
        "--train-csv",
        action="append",
        help="Path to a prepared training CSV. Repeat to merge multiple training files.",
    )
    parser.add_argument(
        "--eval-csv",
        action="append",
        help="Path to a prepared evaluation CSV. Repeat to merge multiple evaluation files.",
    )
    parser.add_argument("--dataset-name", help="Dataset registry name used for column inference.")
    parser.add_argument("--text-column", help="Text column name in the CSV.")
    parser.add_argument("--label-column", help="Label column name in the CSV.")
    parser.add_argument("--language-column", help="Language column name in the CSV.")
    parser.add_argument("--id-column", help="Identifier column name in the CSV.")
    parser.add_argument(
        "--label-mode",
        choices=LABEL_MODE_CHOICES,
        help="Use binary toxicity labels or subtype multilabel tags.",
    )
    parser.add_argument("--max-length", type=int, help="Maximum tokenized sequence length.")
    parser.add_argument("--num-labels", type=int, help="Number of output labels.")
    parser.add_argument("--batch-size", type=int, help="Per-device batch size.")
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        help="Per-device evaluation batch size. Defaults to 2x the train batch size, capped at 32.",
    )
    parser.add_argument("--learning-rate", type=float, help="Optimizer learning rate.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--weight-decay", type=float, help="Weight decay for AdamW.")
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        help="Number of DataLoader worker processes. Auto-tuned when omitted.",
    )
    parser.add_argument(
        "--tokenization-workers",
        type=int,
        help="Number of worker processes to use while tokenizing the dataset.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        help="Optional cap on the number of training rows, mainly for debugging or quick benchmarks.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        help="Optional cap on the number of evaluation rows, mainly for debugging or quick benchmarks.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        help="How often to emit scalar training logs.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction used for evaluation if splitting.")
    parser.add_argument("--output-dir", default="artifacts/transformer", help="Where to save model artifacts.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    data_csv, train_csv, eval_csv = _resolve_data_paths(args, data_config)

    if not data_csv and not train_csv:
        raise ValueError("No dataset path provided. Use --data-csv or --train-csv.")

    metrics = train_transformer(
        model_name=args.model_name or model_config.get("transformer_name", "slm"),
        data_csv=data_csv,
        train_csv=train_csv,
        eval_csv=eval_csv,
        dataset_name=args.dataset_name or data_config.get("dataset_name"),
        text_column=args.text_column or data_config.get("text_column"),
        label_column=args.label_column or data_config.get("label_column"),
        language_column=args.language_column or data_config.get("language_column"),
        id_column=args.id_column or data_config.get("id_column"),
        label_mode=args.label_mode or model_config.get("label_mode", BINARY_LABEL_MODE),
        max_length=args.max_length if args.max_length is not None else model_config.get("max_length", 256),
        num_labels=args.num_labels if args.num_labels is not None else model_config.get("num_labels", 2),
        batch_size=args.batch_size if args.batch_size is not None else training_config.get("batch_size", 8),
        eval_batch_size=(
            args.eval_batch_size
            if args.eval_batch_size is not None
            else training_config.get("eval_batch_size")
        ),
        learning_rate=args.learning_rate or training_config.get("learning_rate", 2e-5),
        epochs=args.epochs if args.epochs is not None else training_config.get("epochs", 3),
        weight_decay=args.weight_decay or training_config.get("weight_decay", 0.01),
        test_size=args.test_size,
        output_dir=args.output_dir,
        dataloader_workers=(
            args.dataloader_workers
            if args.dataloader_workers is not None
            else training_config.get("dataloader_workers")
        ),
        tokenization_workers=(
            args.tokenization_workers
            if args.tokenization_workers is not None
            else training_config.get("tokenization_workers")
        ),
        max_train_samples=(
            args.max_train_samples
            if args.max_train_samples is not None
            else training_config.get("max_train_samples")
        ),
        max_eval_samples=(
            args.max_eval_samples
            if args.max_eval_samples is not None
            else training_config.get("max_eval_samples")
        ),
        logging_steps=(
            args.logging_steps if args.logging_steps is not None else training_config.get("logging_steps", 100)
        ),
        seed=seed,
    )

    print("Transformer training complete.")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"Artifacts saved to: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
