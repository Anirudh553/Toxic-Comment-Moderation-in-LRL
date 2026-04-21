import re
import unicodedata

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
USER_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")
ROMANIZED_PROFANITY_VARIANTS = (
    (re.compile(r"\bgaa+ndu+\b", flags=re.IGNORECASE), "gandu"),
    (re.compile(r"\bgaa+nd\b", flags=re.IGNORECASE), "gand"),
    (re.compile(r"\bma+d+a+r+ch+o+d+\b", flags=re.IGNORECASE), "madarchod"),
    (re.compile(r"\bbe+h+n+ch+o+d+\b", flags=re.IGNORECASE), "bhenchod"),
    (re.compile(r"\bch+u+t+i?y+a+\b", flags=re.IGNORECASE), "chutiya"),
    (re.compile(r"\bch+u+t+i?y+e+\b", flags=re.IGNORECASE), "chutiya"),
    (re.compile(r"\bbh?e+h?e?n\s+k(?:e)?\s+lo+d+e+\b", flags=re.IGNORECASE), "bhen ke lode"),
    (re.compile(r"\bma+a+\s+k[ie]\s+ch(?:u+|oo+)t+\b", flags=re.IGNORECASE), "maa ki chut"),
    (re.compile(r"\bn[i1!][g69]{2,}(?:e|3)r\b", flags=re.IGNORECASE), "nigger"),
    (re.compile(r"\byou\s+belong\s+(?:to|in)\s+the\s+kitchen\b", flags=re.IGNORECASE), "you belong in the kitchen"),
    (re.compile(r"\b(?:tu|tum|aap)\s+pagal\s+(?:ho|hai|he)\b", flags=re.IGNORECASE), "tum pagal ho"),
)
ROMANIZED_PROFANITY_GLOSSES = {
    "bhenchod": "motherfucker",
    "bhen ke lode": "motherfucker",
    "chutiya": "idiot",
    "maa ki chut": "cunt",
    "madarchod": "motherfucker",
    "pajeet": "racist slur",
    "kanglu": "racist slur",
    "tum pagal ho": "idiot",
    "you belong in the kitchen": "sexist",
}


def _normalize_profanity_variants(text: str) -> str:
    normalized_text = text
    for pattern, replacement in ROMANIZED_PROFANITY_VARIANTS:
        normalized_text = pattern.sub(replacement, normalized_text)
    return normalized_text


def _enrich_toxic_cues(text: str) -> str:
    enriched_text = text
    for token, gloss in ROMANIZED_PROFANITY_GLOSSES.items():
        enriched_text = re.sub(
            rf"\b{re.escape(token)}\b",
            f"{token} {gloss}",
            enriched_text,
            flags=re.IGNORECASE,
        )
    return enriched_text


def normalize_text(text: str) -> str:
    """Apply lightweight normalization that is useful for noisy social text."""

    text = unicodedata.normalize("NFKC", str(text))

    # replace URLs and usernames
    text = URL_PATTERN.sub(" <URL> ", text)
    text = USER_PATTERN.sub(" <USER> ", text)

    # normalize elongated characters
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)

    # collapse common romanized profanity spelling variants.
    text = _normalize_profanity_variants(text)
    text = _enrich_toxic_cues(text)

    # separate hashtags
    text = text.replace("#", " #")

    # normalize whitespace
    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()
