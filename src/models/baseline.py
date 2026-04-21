from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_baseline_pipeline() -> Pipeline:
    """Create a strong and simple TF-IDF plus Logistic Regression baseline."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
