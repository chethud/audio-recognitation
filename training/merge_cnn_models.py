"""
Merge separate SED and emotion CNN checkpoints into one file for deployment.

Usage:
  python -m training.merge_cnn_models
  python -m training.merge_cnn_models --sed outputs/sed_cnn.pt --emotion outputs/emotion_cnn.pt
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
DEFAULT_MERGED = BASE / "outputs" / "alm_cnn_merged.pt"


def merge_cnn_checkpoints(
    sed_path: Path,
    emotion_path: Path,
    out_path: Path,
    extra_meta: dict | None = None,
    sed_metrics_json: Path | None = None,
    emotion_metrics_json: Path | None = None,
    merged_metrics_json: Path | None = None,
) -> dict:
    """Load two .pt files and save combined dict under keys `sed` and `emotion`."""
    sed_path, emotion_path = sed_path.resolve(), emotion_path.resolve()
    if not sed_path.is_file():
        raise FileNotFoundError(sed_path)
    if not emotion_path.is_file():
        raise FileNotFoundError(emotion_path)

    sed = torch.load(sed_path, map_location="cpu")
    emo = torch.load(emotion_path, map_location="cpu")

    meta = dict(extra_meta or {})
    if sed_metrics_json and Path(sed_metrics_json).is_file():
        with open(sed_metrics_json, encoding="utf-8") as f:
            meta["sed_metrics"] = json.load(f)
    if emotion_metrics_json and Path(emotion_metrics_json).is_file():
        with open(emotion_metrics_json, encoding="utf-8") as f:
            meta["emotion_metrics"] = json.load(f)

    merged = {
        "format": "alm_cnn_merged_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sed": sed,
        "emotion": emo,
        "meta": meta,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out_path)

    if merged_metrics_json is not None:
        summary = {
            "format": "alm_cnn_metrics_merged_v1",
            "created_utc": merged["created_utc"],
            "checkpoints": {"sed": str(sed_path), "emotion": str(emotion_path), "merged_pt": str(out_path)},
            "sed_metrics": meta.get("sed_metrics"),
            "emotion_metrics": meta.get("emotion_metrics"),
        }
        merged_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with open(merged_metrics_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return merged


def load_merged(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sed", type=Path, default=BASE / "outputs" / "sed_cnn.pt")
    p.add_argument("--emotion", type=Path, default=BASE / "outputs" / "emotion_cnn.pt")
    p.add_argument("--out", type=Path, default=DEFAULT_MERGED)
    p.add_argument("--sed-metrics", type=Path, default=BASE / "outputs" / "sed_metrics.json")
    p.add_argument("--emotion-metrics", type=Path, default=BASE / "outputs" / "emotion_metrics.json")
    p.add_argument("--merged-metrics", type=Path, default=BASE / "outputs" / "alm_cnn_merged_metrics.json")
    args = p.parse_args()
    merge_cnn_checkpoints(
        args.sed,
        args.emotion,
        args.out,
        sed_metrics_json=args.sed_metrics,
        emotion_metrics_json=args.emotion_metrics,
        merged_metrics_json=args.merged_metrics,
    )
    print(f"Merged model saved to: {args.out}")
    print(f"Merged metrics JSON: {args.merged_metrics}")


if __name__ == "__main__":
    main()
