from src.data.preprocessing import normalize_text


def test_normalize_text_replaces_urls_and_users():
    text = "Hello @user check https://example.com"
    normalized = normalize_text(text)
    assert "<USER>" in normalized
    assert "<URL>" in normalized


def test_normalize_text_collapses_whitespace():
    text = "hello   world"
    assert normalize_text(text) == "hello world"


def test_normalize_text_collapses_romanized_profanity_variants():
    assert normalize_text("hello gaandu") == "hello gandu"
    assert normalize_text("hello maadarchoood") == "hello madarchod motherfucker"


def test_normalize_text_enriches_romanized_hinglish_abuse_with_english_gloss():
    assert normalize_text("hello chutiye") == "hello chutiya idiot"


def test_normalize_text_enriches_multiword_hinglish_abuse_with_english_gloss():
    assert normalize_text("bhen ke lode") == "bhen ke lode motherfucker"
    assert normalize_text("maa ki chut") == "maa ki chut cunt"


def test_normalize_text_deobfuscates_leetspeak_hate_slurs():
    assert normalize_text("n1gger") == "nigger"
    assert normalize_text("n!gger") == "nigger"
    assert normalize_text("ni99er") == "nigger"


def test_normalize_text_enriches_targeted_racial_slurs():
    assert normalize_text("pajeet") == "pajeet racist slur"
    assert normalize_text("kanglu") == "kanglu racist slur"


def test_normalize_text_enriches_implicit_toxic_phrases():
    assert normalize_text("you belong to the kitchen") == "you belong in the kitchen sexist"
    assert normalize_text("tum pagal ho") == "tum pagal ho idiot"
