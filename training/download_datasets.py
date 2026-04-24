"""
Download datasets for ALM-Lite CNN training.

- Sound (AudioSet subset): ESC-50 from Hugging Face Datasets is used as a **manageable
  AudioSet-style environmental subset** (50 classes, ~2k clips). Full AudioSet is ~2M clips
  and is not auto-downloaded here; you can swap the loader in `train_sed.py` for a custom
  AudioSet CSV + clips if you have them.

- Emotion: RAVDESS Speech (Zenodo) — `Audio_Speech_Actors_01-24.zip`.

Usage (from project root):

  python -m training.download_datasets --sed
  python -m training.download_datasets --ravdess
  python -m training.download_datasets --all
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = BASE / "data" / "datasets"


def download_esc50_subset_hf(root: Path) -> Path:
    """Load ESC-50 via Hugging Face `datasets` into cache; write manifest under root."""
    from datasets import Audio, load_dataset

    root.mkdir(parents=True, exist_ok=True)
    # Triggers download / cache; training reads the same split from HF in train_sed.py
    ds = load_dataset("ashraq/esc50", split="train")
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=False))
    manifest = root / "esc50_manifest.txt"
    manifest.write_text(
        f"ESC-50 (AudioSet-style subset for CNN). Hugging Face cached. num_rows={len(ds)}\n",
        encoding="utf-8",
    )
    print(f"ESC-50 ready (HF cache). Manifest: {manifest}")
    return root


def download_ravdess_zip(root: Path, url: str | None = None) -> Path:
    """Download and extract RAVDESS Speech zip from Zenodo."""
    import requests
    from tqdm import tqdm

    root.mkdir(parents=True, exist_ok=True)
    extract_dir = root / "ravdess"
    marker = extract_dir / ".extracted_ok"
    if marker.exists():
        print(f"RAVDESS already present: {extract_dir}")
        return extract_dir

    zenodo = url or (
        "https://zenodo.org/records/1188976/files/"
        "Audio_Speech_Actors_01-24.zip?download=1"
    )
    zip_path = root / "Audio_Speech_Actors_01-24.zip"

    print(f"Downloading RAVDESS from Zenodo (~1.2 GB)...")
    with requests.get(zenodo, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="ravdess.zip"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print("Extracting...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    marker.write_text("ok", encoding="utf-8")
    print(f"RAVDESS extracted to: {extract_dir}")
    return extract_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download training datasets")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Dataset root")
    parser.add_argument("--sed", action="store_true", help="Download/load ESC-50 (AudioSet subset)")
    parser.add_argument("--ravdess", action="store_true", help="Download RAVDESS Speech zip")
    parser.add_argument("--all", action="store_true", help="Both")
    args = parser.parse_args()
    root = args.root.resolve()

    if args.all:
        args.sed = True
        args.ravdess = True

    if not args.sed and not args.ravdess:
        parser.print_help()
        print("\nSpecify --sed and/or --ravdess, or --all.", file=sys.stderr)
        sys.exit(1)

    if args.sed:
        download_esc50_subset_hf(root)
    if args.ravdess:
        try:
            download_ravdess_zip(root)
        except Exception as e:
            print(
                f"RAVDESS download failed: {e}\n"
                "Manually download Audio_Speech_Actors_01-24.zip from\n"
                "https://zenodo.org/records/1188976 and extract to "
                f"{root / 'ravdess'}",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
