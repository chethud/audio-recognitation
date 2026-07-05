"""Low-memory ESC-50 loading (streaming) for CNN training on constrained RAM."""
from __future__ import annotations

from typing import Any

from training.train_sed import ESC50MelDataset, _build_label_map, _label_key


def load_esc50_hf_streaming() -> tuple[Any, list[int], int]:
    """
    Load ESC-50 row-by-row via HF streaming to avoid parquet OOM.
    Returns (row_list, label_ids, num_classes) compatible with ESC50MelDataset.
    """
    from datasets import load_dataset

    print("Loading ESC-50 via streaming (low memory)...")
    stream = load_dataset("ashraq/esc50", split="train", streaming=True)
    rows: list[dict] = []
    for row in stream:
        rows.append(dict(row))

    class _Column:
        def __init__(self, items: list[dict], key: str):
            self._items = items
            self._key = key

        def __iter__(self):
            for row in self._items:
                yield row[self._key]

        def __getitem__(self, idx: int):
            return self._items[idx][self._key]

        def __len__(self) -> int:
            return len(self._items)

    class _RowList:
        """Minimal dataset-like wrapper for ESC50MelDataset._load_wav."""

        def __init__(self, items: list[dict], label_col: str):
            self._items = items
            self.column_names = list(items[0].keys()) if items else []
            self._label_col = label_col
            self.features = {label_col: type("F", (), {"names": None})()}

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx):
            if isinstance(idx, str):
                return _Column(self._items, idx)
            return self._items[idx]

        def __iter__(self):
            return iter(self._items)

    if not rows:
        raise RuntimeError("ESC-50 streaming returned no rows")

    label_col = _label_key(_RowList(rows, "label"))
    ds = _RowList(rows, label_col)
    # Patch features.names if available from first row labels
    label_ids, num_classes = _build_label_map(ds)
    return ds, label_ids, num_classes


def load_esc50_for_training():
    """Prefer local wav files; fall back to Hugging Face."""
    from training.esc50_files import DEFAULT_ESC50_ROOT, _find_esc50_root, load_esc50_file_training

    found = _find_esc50_root(DEFAULT_ESC50_ROOT)
    if found:
        print(f"Using local ESC-50 wav files: {found}")
        tr_rows, va_rows, class_names, num_classes = load_esc50_file_training()
        return (tr_rows, va_rows, class_names, num_classes), None, num_classes, "files"

    from datasets import Audio, load_dataset

    try:
        print("Loading ESC-50 (AudioSet-style subset) from Hugging Face...")
        ds = load_dataset("ashraq/esc50", split="train")
        if "audio" in ds.column_names:
            ds = ds.cast_column("audio", Audio(decode=False))
        label_ids, num_classes = _build_label_map(ds)
        return ds, label_ids, num_classes, "hf"
    except Exception as exc:
        print(f"HF ESC-50 unavailable ({exc}); downloading local wav files.")
        tr_rows, va_rows, class_names, num_classes = load_esc50_file_training()
        return (tr_rows, va_rows, class_names, num_classes), None, num_classes, "files"


def make_esc50_mel_datasets(
    ds,
    label_ids: list[int],
    tr_idx: list[int],
    va_idx: list[int],
    n_mels: int,
    time_frames: int,
    tr_labels: list[int],
    va_labels: list[int],
):
    train_set = ESC50MelDataset(ds, tr_idx, n_mels, time_frames, tr_labels, center_crop=False)
    val_set = ESC50MelDataset(ds, va_idx, n_mels, time_frames, va_labels, center_crop=True)
    return train_set, val_set
