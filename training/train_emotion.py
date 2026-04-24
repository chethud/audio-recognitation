"""
Train a CNN on RAVDESS Speech for emotion (8 classes).

Download data first:
  python -m training.download_datasets --ravdess

Usage:
  python -m training.train_emotion --epochs 30 --batch-size 32
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
DEFAULT_DATA = BASE / "data" / "datasets" / "ravdess"
DEFAULT_OUT = BASE / "outputs" / "emotion_cnn.pt"

# RAVDESS filename: Modality-VocalChannel-Emotion-...
# Speech files use modality 03; emotion is the 3rd field (1–8).
EMOTION_NAMES = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


def parse_emotion_from_name(name: str):
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    if parts[0] != "03":
        return None
    try:
        emo = int(parts[2])
    except ValueError:
        return None
    if 1 <= emo <= 8:
        return emo - 1
    return None


def collect_ravdess_files(root: Path):
    """Return list of (wav_path, label 0..7) for speech clips."""
    out: list[tuple[Path, int]] = []
    for wav in sorted(root.rglob("*.wav")):
        lab = parse_emotion_from_name(wav.name)
        if lab is not None:
            out.append((wav, lab))
    return out


class RavdessMelDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], n_mels: int, target_frames: int, sr: int = 16000, center_crop: bool = False):
        self.items = items
        self.n_mels = n_mels
        self.target_frames = target_frames
        self.sr = sr
        self.center_crop = center_crop

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        import soundfile as sf

        path, y = self.items[i]
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if isinstance(wav, np.ndarray) and wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        if sr != self.sr:
            import librosa

            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=self.sr)
        mel = waveform_to_mel(wav, self.sr, n_mels=self.n_mels)
        mel = pad_or_crop_time(mel, self.target_frames, center=self.center_crop)
        t = torch.from_numpy(mel).unsqueeze(0)
        t = normalize_mel(t)
        return t, torch.tensor(y, dtype=torch.long)


def find_ravdess_root(base: Path) -> Path:
    """Locate folder that contains Actor_* after Zenodo extract."""
    if not base.exists():
        raise FileNotFoundError(f"RAVDESS not found at {base}. Run: python -m training.download_datasets --ravdess")
    # Direct children sometimes Audio_Speech_Actors_01-24
    for p in base.rglob("Actor_01"):
        if p.is_dir():
            return p.parent
    raise FileNotFoundError(
        f"No Actor_* folders under {base}. Extract RAVDESS Speech zip there."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--time-frames", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    set_seed(args.seed)

    from sklearn.model_selection import train_test_split

    root = find_ravdess_root(args.data.resolve())
    files = collect_ravdess_files(root)
    if len(files) < 50:
        raise RuntimeError(
            f"Too few RAVDESS speech files ({len(files)}). Check extraction under {args.data}"
        )

    paths = [p for p, _ in files]
    labels = [y for _, y in files]
    strat = labels if len(set(labels)) > 1 else None
    tr, va = train_test_split(
        list(range(len(files))),
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=strat,
    )
    train_items = [files[i] for i in tr]
    val_items = [files[i] for i in va]

    train_set = RavdessMelDataset(train_items, args.n_mels, args.time_frames, center_crop=False)
    val_set = RavdessMelDataset(val_items, args.n_mels, args.time_frames, center_crop=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = 8
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
        correct = total = 0
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
                    "emotion_names": EMOTION_NAMES,
                    "kind": "emotion_ravdess",
                },
                args.output,
            )
            print(f"  saved best to {args.output}")

    print(f"Done. Best val acc ~ {best_acc:.4f}")


if __name__ == "__main__":
    main()
