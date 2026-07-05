"""File-based ESC-50 (GitHub zip) — lazy wav loading, low RAM."""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from training.data_utils import normalize_mel, pad_or_crop_time, waveform_to_mel

BASE = Path(__file__).resolve().parent.parent
DEFAULT_ESC50_ROOT = BASE / "data" / "datasets" / "esc50"
ESC50_ZIP_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"


class Esc50FileMelDataset(Dataset):
    """Lazy ESC-50: reads one wav per __getitem__."""

    def __init__(
        self,
        rows: list[dict],
        n_mels: int,
        time_frames: int,
        center_crop: bool = False,
    ):
        self.rows = rows
        self.n_mels = n_mels
        self.time_frames = time_frames
        self.center_crop = center_crop

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import soundfile as sf

        row = self.rows[i]
        wav, sr = sf.read(row["path"], dtype="float32", always_2d=False)
        if isinstance(wav, np.ndarray) and wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        wav = wav.astype(np.float32)
        mel = waveform_to_mel(wav, int(sr), n_mels=self.n_mels)
        mel = pad_or_crop_time(mel, self.time_frames, center=self.center_crop)
        t = torch.from_numpy(mel).unsqueeze(0)
        t = normalize_mel(t)
        return t, torch.tensor(row["target"], dtype=torch.long)


def _find_esc50_root(root: Path) -> Optional[Path]:
    if (root / "audio").is_dir() and (root / "meta" / "esc50.csv").is_file():
        return root
    for sub in root.iterdir() if root.is_dir() else []:
        if sub.is_dir() and (sub / "audio").is_dir() and (sub / "meta" / "esc50.csv").is_file():
            return sub
    return None


def download_esc50_files(root: Path | None = None) -> Path:
    """Download and extract ESC-50 wav files from GitHub (~600 MB zip)."""
    import requests
    from tqdm import tqdm

    root = (root or DEFAULT_ESC50_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    found = _find_esc50_root(root)
    if found:
        print(f"ESC-50 files already present: {found}")
        return found

    marker = root / ".esc50_download_ok"
    zip_path = root / "esc50_master.zip"
    print(f"Downloading ESC-50 from GitHub -> {zip_path}")
    with requests.get(ESC50_ZIP_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="esc50.zip"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    print("Extracting ESC-50...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    found = _find_esc50_root(root)
    if not found:
        raise RuntimeError(f"ESC-50 extract failed under {root}")
    marker.write_text(str(found), encoding="utf-8")
    print(f"ESC-50 ready: {found}")
    return found


def load_esc50_file_rows(root: Path | None = None) -> tuple[list[dict], list[str]]:
    """Return row dicts with path/target/category and sorted class names."""
    root = download_esc50_files(root)
    csv_path = root / "meta" / "esc50.csv"
    audio_dir = root / "audio"
    rows: list[dict] = []
    categories: dict[int, str] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            fname = rec["filename"]
            target = int(rec["target"])
            categories[target] = rec.get("category", str(target))
            wav_path = audio_dir / fname
            if not wav_path.is_file():
                continue
            rows.append(
                {
                    "path": str(wav_path),
                    "target": target,
                    "category": categories[target],
                    "fold": int(rec.get("fold", 0)),
                }
            )

    if not rows:
        raise RuntimeError(f"No ESC-50 wav files found under {audio_dir}")

    class_names = [categories[i] for i in sorted(categories.keys())]
    return rows, class_names


def load_esc50_file_training(root: Path | None = None):
    """Load ESC-50 from local wav files (low memory)."""
    from sklearn.model_selection import train_test_split

    rows, class_names = load_esc50_file_rows(root)
    num_classes = len(class_names)
    label_ids = [r["target"] for r in rows]
    n = len(rows)
    idx_all = list(range(n))
    strat = label_ids if len(set(label_ids)) > 1 else None
    tr_idx, va_idx = train_test_split(
        idx_all, test_size=0.2, random_state=42, stratify=strat
    )
    tr_rows = [rows[i] for i in tr_idx]
    va_rows = [rows[i] for i in va_idx]
    return tr_rows, va_rows, class_names, num_classes
