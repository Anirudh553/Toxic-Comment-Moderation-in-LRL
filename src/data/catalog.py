from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata and column hints for supported datasets."""

    name: str
    source_url: str
    description: str
    text_columns: Tuple[str, ...]
    label_columns: Tuple[str, ...]
    language_columns: Tuple[str, ...] = ()
    id_columns: Tuple[str, ...] = ()
    default_raw_filename: str = ""


MULTILINGUAL_TOXIC_COMMENTS = DatasetSpec(
    name="multilingual_toxic_comments",
    source_url="https://github.com/AkshayaANANTAM/Multilingual-Toxic-comments-classification",
    description="Multilingual toxic comment classification dataset with comment text, binary toxicity labels, and language tags.",
    text_columns=("comment_text", "text", "comment"),
    label_columns=("toxic", "label", "target"),
    language_columns=("lang", "language"),
    id_columns=("id", "comment_id"),
    default_raw_filename="multilingual_toxic_comments.csv",
)


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    MULTILINGUAL_TOXIC_COMMENTS.name: MULTILINGUAL_TOXIC_COMMENTS,
}
