"""Small metrics helpers without a training-only sklearn dependency."""

import numpy as np


def classification_metrics(targets, predicted, classes: int) -> dict:
    targets = np.asarray(targets, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if targets.ndim != 1 or targets.shape != predicted.shape or not targets.size:
        raise ValueError("Targets and predictions must be nonempty matching vectors.")
    if classes < 1 or np.any(targets < 0) or np.any(targets >= classes):
        raise ValueError("Target index is outside the class list.")
    if np.any(predicted < 0) or np.any(predicted >= classes):
        raise ValueError("Prediction index is outside the class list.")
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    true_positive = confusion.diagonal().astype(np.float64)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    precision = np.divide(
        true_positive, predicted_count, out=np.zeros(classes), where=predicted_count > 0
    )
    recall = np.divide(true_positive, support, out=np.zeros(classes), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator, out=np.zeros(classes), where=denominator > 0
    )
    return {
        "samples": int(targets.size),
        "accuracy": float(true_positive.sum() / targets.size),
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": confusion.tolist(),
    }
