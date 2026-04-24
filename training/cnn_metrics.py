"""Detailed accuracy / classification metrics for CNN training (SED + emotion)."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader


def gather_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return y_true, y_pred as int64 arrays."""
    model.eval()
    ys: List[int] = []
    ps: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(dim=1).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(pred.tolist())
    return np.array(ys, dtype=np.int64), np.array(ps, dtype=np.int64)


def full_metrics_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    average_none: bool = True,
) -> dict[str, Any]:
    """
    Accuracy, balanced accuracy, macro/micro/weighted precision, recall, F1,
    per-class report, confusion matrix.
    """
    labels = list(range(len(class_names))) if class_names is not None else None
    out: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0, labels=labels)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
        ),
    }
    target_names = list(class_names) if class_names is not None else None
    out["classification_report"] = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        labels=labels,
        digits=4,
        zero_division=0,
    )
    out["classification_report_dict"] = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out["confusion_matrix"] = cm.tolist()
    if class_names is not None:
        out["per_class"] = {}
        for i, name in enumerate(class_names):
            mask = y_true == i
            if mask.sum() == 0:
                out["per_class"][name] = {"support": 0, "recall_on_class": None}
            else:
                rec = (y_pred[mask] == i).sum() / mask.sum()
                out["per_class"][name] = {
                    "support": int(mask.sum()),
                    "recall_on_class": float(rec),
                }
    return out


def print_metrics_summary(m: dict[str, Any]) -> None:
    print("--- Summary ---")
    print(f"Samples:     {m['n_samples']}")
    print(f"Accuracy:    {m['accuracy']:.4f}")
    print(f"Balanced acc:{m['balanced_accuracy']:.4f}")
    print(f"Macro F1:    {m['macro_f1']:.4f}")
    print(f"Weighted F1: {m['weighted_f1']:.4f}")
    print("\n--- Classification report ---\n")
    print(m["classification_report"])
    print("\n--- Confusion matrix (rows=true, cols=pred) ---")
    for row in m["confusion_matrix"]:
        print(row)
