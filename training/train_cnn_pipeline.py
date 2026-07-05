"""
End-to-end CNN training with full validation metrics (for notebooks and scripts).

Returns dicts suitable for JSON export and the merged checkpoint step.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from training.cnn_metrics import full_metrics_dict, gather_predictions, print_metrics_summary
from training.data_utils import MelSpecCNN, set_seed
from training.train_emotion import (
    EMOTION_NAMES,
    RavdessMelDataset,
    collect_ravdess_files,
    find_ravdess_root,
)
from training.train_sed import ESC50MelDataset, _build_label_map, _label_key

BASE = Path(__file__).resolve().parent.parent


def _jsonable(o: Any) -> Any:
    """Convert numpy/sklearn-nested structures for json.dump."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _esc50_class_names(ds, label_ids: list[int]) -> list[str]:
    key = _label_key(ds)
    col = ds[key]
    feat = ds.features[key]
    if hasattr(feat, "names") and feat.names is not None:
        return list(feat.names)
    sample = col[0]
    if isinstance(sample, str):
        return sorted(set(col))
    n = max(label_ids) + 1
    return [str(i) for i in range(n)]


def train_esc50_sed(
    *,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    use_class_weights: bool = True,
    early_stopping_patience: int = 8,
    min_lr: float = 1e-6,
    seed: int = 42,
    n_mels: int = 64,
    time_frames: int = 128,
    val_ratio: float = 0.2,
    output_pt: Path | None = None,
    output_metrics_json: Path | None = None,
) -> dict[str, Any]:
    """Train SED CNN on ESC-50; save checkpoint + detailed val metrics on best epoch."""
    from sklearn.model_selection import train_test_split

    output_pt = output_pt or (BASE / "outputs" / "sed_cnn.pt")
    output_metrics_json = output_metrics_json or (BASE / "outputs" / "sed_metrics.json")

    set_seed(seed)
    from training.esc50_loader import load_esc50_for_training

    loaded, label_ids, num_classes, source = load_esc50_for_training()
    if source == "hf":
        ds = loaded
        class_names = _esc50_class_names(ds, label_ids)
        n = len(ds)
        idx_all = list(range(n))
        strat = label_ids if len(set(label_ids)) > 1 else None
        tr_idx, va_idx = train_test_split(
            idx_all,
            test_size=val_ratio,
            random_state=seed,
            stratify=strat,
        )
        tr_labels = [label_ids[i] for i in tr_idx]
        va_labels = [label_ids[i] for i in va_idx]
        train_set = ESC50MelDataset(ds, tr_idx, n_mels, time_frames, tr_labels, center_crop=False)
        val_set = ESC50MelDataset(ds, va_idx, n_mels, time_frames, va_labels, center_crop=True)
    else:
        from training.esc50_files import Esc50FileMelDataset

        tr_rows, va_rows, class_names, num_classes = loaded
        tr_labels = [r["target"] for r in tr_rows]
        train_set = Esc50FileMelDataset(tr_rows, n_mels, time_frames, center_crop=False)
        val_set = Esc50FileMelDataset(va_rows, n_mels, time_frames, center_crop=True)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MelSpecCNN(num_classes=num_classes, n_mels=n_mels, time_frames=time_frames)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3, min_lr=min_lr
    )
    class_weights = None
    if use_class_weights:
        bincount = np.bincount(np.asarray(tr_labels), minlength=num_classes).astype(np.float32)
        bincount[bincount == 0.0] = 1.0
        inv = 1.0 / bincount
        class_weights = torch.tensor(inv / inv.mean(), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    best_acc = 0.0
    best_macro_f1 = -1.0
    best_state = None
    history: list[dict[str, Any]] = []
    epochs_without_improve = 0
    output_pt.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_set)

        model.eval()
        yt, yp = gather_predictions(model, val_loader, device)
        val_metrics = full_metrics_dict(yt, yp, class_names=class_names)
        acc = val_metrics["accuracy"]
        macro_f1 = val_metrics["macro_f1"]
        scheduler.step(acc)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_accuracy": acc,
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": macro_f1,
            "val_weighted_f1": val_metrics["weighted_f1"],
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_acc={acc:.4f}  macro_f1={macro_f1:.4f}  lr={opt.param_groups[0]['lr']:.2e}"
        )

        improved = (acc > best_acc) or (acc >= best_acc and macro_f1 > best_macro_f1)
        if improved:
            best_acc = acc
            best_macro_f1 = macro_f1
            epochs_without_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model": best_state,
                    "num_classes": num_classes,
                    "n_mels": n_mels,
                    "time_frames": time_frames,
                    "kind": "sed_esc50",
                    "class_names": class_names,
                    "best_val_accuracy": best_acc,
                    "best_val_macro_f1": best_macro_f1,
                },
                output_pt,
            )
            print(f"  saved best to {output_pt}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= early_stopping_patience:
                print(
                    f"  early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)"
                )
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    yt, yp = gather_predictions(model, val_loader, device)
    final_metrics = full_metrics_dict(yt, yp, class_names=class_names)
    print_metrics_summary(final_metrics)

    result = {
        "task": "sed_esc50",
        "dataset": "ashraq/esc50",
        "num_train": len(train_set),
        "num_val": len(val_set),
        "num_classes": num_classes,
        "class_names": class_names,
        "device": str(device),
        "best_val_accuracy": float(best_acc),
        "history": history,
        "final_validation_metrics": final_metrics,
        "checkpoint": str(output_pt.resolve()),
    }

    payload = {
        "summary": {k: v for k, v in result.items() if k != "final_validation_metrics"},
        "final_validation_metrics": {
            **{k: v for k, v in final_metrics.items() if k not in ("classification_report",)},
            "classification_report": final_metrics["classification_report"],
        },
    }
    output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_json, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)

    print(f"Metrics JSON: {output_metrics_json}")
    return result


def train_ravdess_emotion(
    *,
    data_root: Path | None = None,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    use_class_weights: bool = True,
    early_stopping_patience: int = 10,
    min_lr: float = 1e-6,
    seed: int = 42,
    n_mels: int = 64,
    time_frames: int = 128,
    val_ratio: float = 0.2,
    output_pt: Path | None = None,
    output_metrics_json: Path | None = None,
) -> dict[str, Any]:
    """Train emotion CNN on RAVDESS; save checkpoint + detailed val metrics."""
    from sklearn.model_selection import train_test_split

    data_root = data_root or (BASE / "data" / "datasets" / "ravdess")
    output_pt = output_pt or (BASE / "outputs" / "emotion_cnn.pt")
    output_metrics_json = output_metrics_json or (BASE / "outputs" / "emotion_metrics.json")

    set_seed(seed)
    root = find_ravdess_root(data_root.resolve())
    files = collect_ravdess_files(root)
    if len(files) < 50:
        raise RuntimeError(f"Too few RAVDESS speech files ({len(files)}). Download/extract first.")

    labels = [y for _, y in files]
    strat = labels if len(set(labels)) > 1 else None
    tr, va = train_test_split(
        list(range(len(files))),
        test_size=val_ratio,
        random_state=seed,
        stratify=strat,
    )
    train_items = [files[i] for i in tr]
    val_items = [files[i] for i in va]

    train_set = RavdessMelDataset(train_items, n_mels, time_frames, center_crop=False)
    val_set = RavdessMelDataset(val_items, n_mels, time_frames, center_crop=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    num_classes = 8
    class_names = list(EMOTION_NAMES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MelSpecCNN(num_classes=num_classes, n_mels=n_mels, time_frames=time_frames)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3, min_lr=min_lr
    )
    class_weights = None
    if use_class_weights:
        bincount = np.bincount(np.asarray(labels)[tr], minlength=num_classes).astype(np.float32)
        bincount[bincount == 0.0] = 1.0
        inv = 1.0 / bincount
        class_weights = torch.tensor(inv / inv.mean(), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    best_acc = 0.0
    best_macro_f1 = -1.0
    best_state = None
    history: list[dict[str, Any]] = []
    epochs_without_improve = 0
    output_pt.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_set)

        model.eval()
        yt, yp = gather_predictions(model, val_loader, device)
        val_metrics = full_metrics_dict(yt, yp, class_names=class_names)
        acc = val_metrics["accuracy"]
        macro_f1 = val_metrics["macro_f1"]
        scheduler.step(acc)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_accuracy": acc,
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": macro_f1,
                "val_weighted_f1": val_metrics["weighted_f1"],
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )
        print(
            f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_acc={acc:.4f}  macro_f1={macro_f1:.4f}  lr={opt.param_groups[0]['lr']:.2e}"
        )

        improved = (acc > best_acc) or (acc >= best_acc and macro_f1 > best_macro_f1)
        if improved:
            best_acc = acc
            best_macro_f1 = macro_f1
            epochs_without_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model": best_state,
                    "num_classes": num_classes,
                    "n_mels": n_mels,
                    "time_frames": time_frames,
                    "emotion_names": EMOTION_NAMES,
                    "kind": "emotion_ravdess",
                    "best_val_accuracy": best_acc,
                    "best_val_macro_f1": best_macro_f1,
                },
                output_pt,
            )
            print(f"  saved best to {output_pt}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= early_stopping_patience:
                print(
                    f"  early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)"
                )
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    yt, yp = gather_predictions(model, val_loader, device)
    final_metrics = full_metrics_dict(yt, yp, class_names=class_names)
    print_metrics_summary(final_metrics)

    result = {
        "task": "emotion_ravdess",
        "dataset": "RAVDESS Speech (03-*)",
        "num_train": len(train_set),
        "num_val": len(val_set),
        "num_classes": num_classes,
        "class_names": class_names,
        "device": str(device),
        "best_val_accuracy": float(best_acc),
        "history": history,
        "final_validation_metrics": final_metrics,
        "checkpoint": str(output_pt.resolve()),
    }

    payload = {
        "summary": {k: v for k, v in result.items() if k != "final_validation_metrics"},
        "final_validation_metrics": {
            **{k: v for k, v in final_metrics.items() if k not in ("classification_report",)},
            "classification_report": final_metrics["classification_report"],
        },
    }
    output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_json, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)

    print(f"Metrics JSON: {output_metrics_json}")
    return result
