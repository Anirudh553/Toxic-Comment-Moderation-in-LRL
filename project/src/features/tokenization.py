from typing import Dict, List


def whitespace_tokenize(text: str) -> List[str]:
    """Simple baseline tokenizer."""
    return text.split()


def build_token_record(text: str) -> Dict[str, object]:
    """Return a minimal token record for quick experiments."""
    tokens = whitespace_tokenize(text)
    return {"tokens": tokens, "length": len(tokens)}

