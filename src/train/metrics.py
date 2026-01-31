from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support


def safe_auc(y_true, y_prob) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def compute_bin_metrics(y_true, y_prob, thresh: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thresh).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    auc = safe_auc(y_true, y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "acc": acc,
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def aggregate_study_level(study_ids, y_true_img, y_prob_img):
    df = pd.DataFrame({
        "study": study_ids,
        "y_true": y_true_img,
        "y_prob": y_prob_img,
    })
    g = df.groupby("study", as_index=False).mean(numeric_only=True)
    y_true_st = np.rint(g["y_true"].values).astype(int)
    y_prob_st = g["y_prob"].values.astype(float)
    return g["study"].values, y_true_st, y_prob_st
