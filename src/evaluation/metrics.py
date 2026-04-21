from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import hamming_loss


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return standard text classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def multilabel_classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return exact-match and macro/micro metrics for multilabel moderation tags."""
    y_true_array = np.asarray(y_true).astype(int)
    y_pred_array = np.asarray(y_pred).astype(int)

    return {
        "accuracy": accuracy_score(y_true_array, y_pred_array),
        "precision_macro": precision_score(y_true_array, y_pred_array, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_array, y_pred_array, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true_array, y_pred_array, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true_array, y_pred_array, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true_array, y_pred_array, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true_array, y_pred_array, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true_array, y_pred_array),
    }
