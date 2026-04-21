from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for a pretrained sequence classification backbone."""

    key: str
    hf_model_name: str
    max_length: int = 256
    num_labels: int = 2

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


TRANSFORMER_PRESETS: Dict[str, TransformerConfig] = {
    "slm": TransformerConfig(
        key="slm",
        hf_model_name="distilbert-base-multilingual-cased",
    ),
    "distilbert-multilingual": TransformerConfig(
        key="slm",
        hf_model_name="distilbert-base-multilingual-cased",
    ),
    "distilbert-base-multilingual-cased": TransformerConfig(
        key="slm",
        hf_model_name="distilbert-base-multilingual-cased",
    ),
    "xlm-r": TransformerConfig(key="xlm-r", hf_model_name="xlm-roberta-base"),
    "xlm-roberta-base": TransformerConfig(key="xlm-r", hf_model_name="xlm-roberta-base"),
    "muril": TransformerConfig(key="muril", hf_model_name="google/muril-base-cased"),
    "google/muril-base-cased": TransformerConfig(
        key="muril",
        hf_model_name="google/muril-base-cased",
    ),
}


def resolve_transformer_config(
    model_name: str = "slm",
    max_length: int = 256,
    num_labels: int = 2,
) -> TransformerConfig:
    """Resolve a model alias into a concrete Hugging Face checkpoint config."""
    normalized_name = model_name.strip().lower()
    if normalized_name not in TRANSFORMER_PRESETS:
        supported = ", ".join(sorted(TRANSFORMER_PRESETS))
        raise ValueError(f"Unsupported transformer model '{model_name}'. Supported values: {supported}")

    preset = TRANSFORMER_PRESETS[normalized_name]
    return TransformerConfig(
        key=preset.key,
        hf_model_name=preset.hf_model_name,
        max_length=max_length,
        num_labels=num_labels,
    )


def _resolve_local_model_path(model_name: str) -> str:
    cache_dir_name = f"models--{model_name.replace('/', '--')}"
    snapshot_root = Path.home() / ".cache" / "huggingface" / "hub" / cache_dir_name / "snapshots"
    if not snapshot_root.exists():
        return model_name

    snapshots = [path for path in snapshot_root.iterdir() if path.is_dir()]
    if not snapshots:
        return model_name

    latest_snapshot = max(snapshots, key=lambda path: path.stat().st_mtime)
    return str(latest_snapshot)


def build_transformer_components(config: TransformerConfig):
    """Create tokenizer and sequence classifier for the resolved config."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    load_target = _resolve_local_model_path(config.hf_model_name)
    load_kwargs = {"local_files_only": True} if load_target != config.hf_model_name else {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(load_target, **load_kwargs)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(
            config.hf_model_name,
            local_files_only=True,
            use_fast=False,
        )

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            load_target,
            num_labels=config.num_labels,
            **load_kwargs,
        )
    except OSError:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.hf_model_name,
            num_labels=config.num_labels,
            local_files_only=True,
        )
    return tokenizer, model
