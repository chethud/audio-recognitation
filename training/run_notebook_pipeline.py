"""
Automated equivalent of ALM_Lite_CNN_Training.ipynb:
  download datasets -> train SED + emotion CNNs -> merge -> verify checkpoints

  python -m training.run_notebook_pipeline
  python -m training.run_notebook_pipeline --quick
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

BASE = Path(__file__).resolve().parent.parent


def _verify_checkpoints(merged: Path) -> None:
    import torch

    from src.cnn import cnn_checkpoints_exist, cnn_models_available, predict_emotion_cnn, predict_sed_cnn

    print("\n--- Verification ---")
    print("checkpoints_exist:", cnn_checkpoints_exist())
    if not merged.is_file():
        print("FAIL: merged checkpoint missing")
        sys.exit(1)

    data = torch.load(merged, map_location="cpu", weights_only=False)
    assert data.get("format") == "alm_cnn_merged_v1", data.get("format")
    assert "sed" in data and "emotion" in data
    print("merged format OK:", merged)

    import numpy as np

    wav = np.random.randn(16000).astype(np.float32) * 0.01
    sed = predict_sed_cnn(wav, top_k=3, threshold=0.0)
    emo = predict_emotion_cnn(wav)
    print("SED sample:", sed[:3] if sed else "(empty)")
    print("Emotion sample:", emo)
    print("cnn_models_available:", cnn_models_available())
    print("Verification passed.")


def main() -> None:
    p = argparse.ArgumentParser(description="Run ALM-Lite CNN notebook pipeline")
    p.add_argument("--quick", action="store_true", help="Few epochs for smoke test (default)")
    p.add_argument("--epochs-sed", type=int, default=None)
    p.add_argument("--epochs-emo", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()

    epochs_sed = args.epochs_sed if args.epochs_sed is not None else (8 if args.quick else 40)
    epochs_emo = args.epochs_emo if args.epochs_emo is not None else (12 if args.quick else 60)

    out = BASE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    data_root = BASE / "data" / "datasets"

    if not args.skip_download:
        from training.download_datasets import download_ravdess_zip
        from training.esc50_files import download_esc50_files

        print("Step 1/4: Download datasets (if missing)...")
        try:
            download_ravdess_zip(data_root)
        except Exception as exc:
            print(f"RAVDESS download warning: {exc}")
        try:
            download_esc50_files(data_root / "esc50")
        except Exception as exc:
            print(f"ESC-50 download warning: {exc}")

    from training.merge_cnn_models import merge_cnn_checkpoints
    from training.train_cnn_pipeline import train_esc50_sed, train_ravdess_emotion

    sed_pt = out / "sed_cnn.pt"
    emo_pt = out / "emotion_cnn.pt"
    merged_pt = out / "alm_cnn_merged.pt"

    print(f"Step 2/4: Train SED CNN ({epochs_sed} epochs)...")
    train_esc50_sed(
        epochs=epochs_sed,
        batch_size=args.batch_size,
        output_pt=sed_pt,
        output_metrics_json=out / "sed_metrics.json",
    )

    print(f"Step 3/4: Train emotion CNN ({epochs_emo} epochs)...")
    train_ravdess_emotion(
        data_root=data_root / "ravdess",
        epochs=epochs_emo,
        batch_size=args.batch_size,
        output_pt=emo_pt,
        output_metrics_json=out / "emotion_metrics.json",
    )

    print("Step 4/4: Merge checkpoints...")
    merge_cnn_checkpoints(
        sed_pt,
        emo_pt,
        merged_pt,
        sed_metrics_json=out / "sed_metrics.json",
        emotion_metrics_json=out / "emotion_metrics.json",
        merged_metrics_json=out / "alm_cnn_merged_metrics.json",
    )

    _verify_checkpoints(merged_pt)
    print("\nDone. Restart backend to use CNN inference.")


if __name__ == "__main__":
    main()
