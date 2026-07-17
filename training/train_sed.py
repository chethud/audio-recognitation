"""
Train a CNN on an AudioSet **subset** (ESC-50 via Hugging Face) for environmental sound classification.

Full AudioSet is not bundled; ESC-50 provides 50 environmental classes suitable for a student /
production prototype. Run `python -m training.download_datasets --sed` first to cache the dataset.

Usage:
  python -m training.download_datasets --sed
  python -m training.train_sed --epochs 25 --batch-size 16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from training.data_utils import MelSpecCNN, normalize_mel, pad_or_crop_time, set_seed, waveform_to_mel

BASE = Path(__file__).resolve().parent.parent
DEFAULT_OUT = BASE / "outputs" / "sed_cnn.pt"


class ESC50MelDataset(Dataset):
    """HF ESC-50 rows -> fixed-size log-mel tensors."""

    def __init__(self, hf_ds, indices, n_mels: int, target_frames: int, label_ids, center_crop: bool = False, target_sr: int = 16000):
        self.ds = hf_ds
        self.indices = indices
        self.n_mels = n_mels
        self.target_frames = target_frames
        self.label_ids = label_ids
        self.center_crop = center_crop
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.indices)

    def _load_wav(self, audio_obj):
        """Support array, file path, or raw bytes (HF Audio decode=False)."""
        import io

        import soundfile as sf

        if isinstance(audio_obj, dict) and audio_obj.get("array") is not None:
            wav = np.asarray(audio_obj["array"], dtype=np.float32)
            sr = int(audio_obj.get("sampling_rate", 44100))
            return wav, sr
        if isinstance(audio_obj, dict):
            if audio_obj.get("bytes") is not None:
                raw = audio_obj["bytes"]
                bio = io.BytesIO(raw if isinstance(raw, bytes) else bytes(raw))
                wav, sr = sf.read(bio, dtype="float32", always_2d=False)
                if isinstance(wav, np.ndarray) and wav.ndim > 1:
                    wav = np.mean(wav, axis=1)
                return wav.astype(np.float32), int(sr)
            path = audio_obj.get("path")
            if path:
                wav, sr = sf.read(path, dtype="float32", always_2d=False)
                if isinstance(wav, np.ndarray) and wav.ndim > 1:
                    wav = np.mean(wav, axis=1)
                return wav.astype(np.float32), int(sr)
        raise ValueError("Cannot load audio row")

    def __getitem__(self, i: int):
        row = self.ds[self.indices[i]]
        audio = row["audio"]
        wav, sr = self._load_wav(audio)
        if sr != self.target_sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        mel = waveform_to_mel(wav, sr, n_mels=self.n_mels)
        mel = pad_or_crop_time(mel, self.target_frames, center=self.center_crop)
        t = torch.from_numpy(mel).unsqueeze(0)
        t = normalize_mel(t)
        y = self.label_ids[i]
        return t, torch.tensor(y, dtype=torch.long)


def _label_key(ds):
    for k in ("label", "category", "labels"):
        if k in ds.features:
            return k
    raise KeyError("No label column found (expected label/category)")


def _build_label_map(ds):
    key = _label_key(ds)
    feat = ds.features
    lab = feat[key]
    col = ds[key]
    if hasattr(lab, "names") and lab.names is not None:
        names = list(lab.names)
        str2id = {n: i for i, n in enumerate(names)}
        ids = []
        for v in col:
            if isinstance(v, str):
                ids.append(str2id[v])
            else:
                ids.append(int(v))
        num_classes = len(names)
        return ids, num_classes
    sample = col[0]
    if isinstance(sample, str):
        uniq = sorted(set(col))
        str2id = {s: i for i, s in enumerate(uniq)}
        ids = [str2id[v] for v in col]
        return ids, len(uniq)
    ids = [int(v) for v in col]
    num_classes = max(ids) + 1
    return ids, num_classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--time-frames", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    set_seed(args.seed)

    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    print("Loading ESC-50 (AudioSet subset) from Hugging Face...")
    from datasets import Audio

    ds = load_dataset("ashraq/esc50", split="train")
    # Avoid torchcodec: read file paths with soundfile in the Dataset
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=False))
    label_ids, num_classes = _build_label_map(ds)
    n = len(ds)
    idx_all = list(range(n))
    strat = label_ids if len(set(label_ids)) > 1 else None
    tr_idx, va_idx = train_test_split(
        idx_all,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=strat,
    )
    tr_labels = [label_ids[i] for i in tr_idx]
    va_labels = [label_ids[i] for i in va_idx]

    train_set = ESC50MelDataset(ds, tr_idx, args.n_mels, args.time_frames, tr_labels, center_crop=False)
    val_set = ESC50MelDataset(ds, va_idx, args.n_mels, args.time_frames, va_labels, center_crop=True)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MelSpecCNN(num_classes=num_classes, n_mels=args.n_mels, time_frames=args.time_frames)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_set)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1)
        print(f"epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  val_acc={acc:.4f}")

        if acc >= best_acc:
            best_acc = acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_classes": num_classes,
                    "n_mels": args.n_mels,
                    "time_frames": args.time_frames,
                    "kind": "sed_esc50",
                },
                args.output,
            )
            print(f"  saved best to {args.output}")

    print(f"Done. Best val acc ~ {best_acc:.4f}")


if __name__ == "__main__":
    main()
