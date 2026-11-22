"""Accuracy, loss, precision, recall, F1 and ROC-AUC.

Precision/recall/F1/ROC-AUC are macro-averaged: computed per class, then averaged with equal
weight, so a small class (mountain_hiking) counts as much as a large one.
"""

from __future__ import annotations

from typing import Any

import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, log_loss,
                             precision_recall_fscore_support, roc_auc_score)


def predict_dataset(model: keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """(true labels, predicted probabilities) for every example in an (images, labels) dataset."""
    labels, probs = [], []
    for x, y in dataset:
        probs.append(model(x, training=False).numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """All headline metrics, per-class scores and the confusion matrix, as plain JSON-able values."""
    labels = list(range(len(class_names)))
    y_pred = y_prob.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                               average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "loss": round(float(log_loss(y_true, y_prob, labels=labels)), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "roc_auc_macro": round(float(roc_auc_score(y_true, y_prob, labels=labels, multi_class="ovr",
                                                   average="macro")), 4),
        "per_class": {c: {k: round(float(report[c][k]), 4) for k in ("precision", "recall", "f1-score")}
                      | {"support": int(report[c]["support"])} for c in class_names},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate_model(model: keras.Model, dataset: tf.data.Dataset, class_names: list[str]) -> dict[str, Any]:
    """classification_metrics for a model on a dataset."""
    y_true, y_prob = predict_dataset(model, dataset)
    return classification_metrics(y_true, y_prob, class_names)
