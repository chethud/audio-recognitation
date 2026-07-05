"""
Quick CNN training for local smoke tests (few epochs). Full quality: use ALM_Lite_CNN_Training.ipynb.

  python -m training.train_cnn_quick
  python -m training.train_cnn_quick --epochs-sed 15 --epochs-emo 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

from training.merge_cnn_models import merge_cnn_checkpoints
from training.train_cnn_pipeline import train_esc50_sed, train_ravdess_emotion

BASE = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description="Train and merge ALM-Lite CNN checkpoints")
    p.add_argument("--epochs-sed", type=int, default=12, help="ESC-50 SED epochs (default 12)")
    p.add_argument("--epochs-emo", type=int, default=15, help="RAVDESS emotion epochs (default 15)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--skip-merge", action="store_true")
    args = p.parse_args()

    out = BASE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    sed_pt = out / "sed_cnn.pt"
    emo_pt = out / "emotion_cnn.pt"
    merged_pt = out / "alm_cnn_merged.pt"

    print(f"Training SED CNN ({args.epochs_sed} epochs) -> {sed_pt}")
    train_esc50_sed(
        epochs=args.epochs_sed,
        batch_size=args.batch_size,
        output_pt=sed_pt,
        output_metrics_json=out / "sed_metrics.json",
    )

    print(f"Training emotion CNN ({args.epochs_emo} epochs) -> {emo_pt}")
    train_ravdess_emotion(
        data_root=BASE / "data" / "datasets" / "ravdess",
        epochs=args.epochs_emo,
        batch_size=args.batch_size,
        output_pt=emo_pt,
        output_metrics_json=out / "emotion_metrics.json",
    )

    if not args.skip_merge:
        print(f"Merging -> {merged_pt}")
        merge_cnn_checkpoints(
            sed_pt,
            emo_pt,
            merged_pt,
            sed_metrics_json=out / "sed_metrics.json",
            emotion_metrics_json=out / "emotion_metrics.json",
            merged_metrics_json=out / "alm_cnn_merged_metrics.json",
        )

    print("Done. Restart the backend to load CNN inference.")


if __name__ == "__main__":
    main()
