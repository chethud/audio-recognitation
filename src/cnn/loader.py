"""Load ALM-Lite mel-spectrogram CNN checkpoints (SED + emotion)."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from training.data_utils import MelSpecCNN

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE / "config.yaml"

_lock = threading.Lock()
_sed_bundle: Optional[dict[str, Any]] = None
_emo_bundle: Optional[dict[str, Any]] = None


def _alm_cnn_cfg(cfg: Optional[dict] = None) -> dict[str, Any]:
    if cfg is None:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    return cfg.get("alm_cnn", {})


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else BASE / p


def _class_names_from_ckpt(ckpt: dict[str, Any]) -> list[str]:
    if ckpt.get("class_names"):
        return list(ckpt["class_names"])
    if ckpt.get("emotion_names"):
        return list(ckpt["emotion_names"])
    metrics_path = ckpt.get("_metrics_json")
    if metrics_path and Path(metrics_path).is_file():
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
        names = data.get("summary", {}).get("class_names")
        if names:
            return list(names)
    n = int(ckpt.get("num_classes", 0))
    return [str(i) for i in range(n)]


def _load_ckpt_file(path: Path, metrics_json: Path | None = None) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if metrics_json and metrics_json.is_file():
        ckpt = dict(ckpt)
        ckpt["_metrics_json"] = str(metrics_json)
    return ckpt


def _bundle_from_ckpt(ckpt: dict[str, Any], device: torch.device) -> dict[str, Any]:
    class_names = _class_names_from_ckpt(ckpt)
    n_mels = int(ckpt.get("n_mels", 64))
    time_frames = int(ckpt.get("time_frames", 128))
    model = MelSpecCNN(
        num_classes=len(class_names),
        n_mels=n_mels,
        time_frames=time_frames,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return {
        "model": model,
        "class_names": class_names,
        "n_mels": n_mels,
        "time_frames": time_frames,
        "device": device,
    }


def cnn_checkpoints_exist(cfg: Optional[dict] = None) -> bool:
    cnn = _alm_cnn_cfg(cfg)
    merged = _resolve(cnn.get("merged_checkpoint", "outputs/alm_cnn_merged.pt"))
    if merged.is_file():
        return True
    sed = _resolve(cnn.get("sed_checkpoint", "outputs/sed_cnn.pt"))
    emo = _resolve(cnn.get("emotion_checkpoint", "outputs/emotion_cnn.pt"))
    return sed.is_file() and emo.is_file()


def should_use_cnn(cfg: Optional[dict] = None) -> bool:
    cnn = _alm_cnn_cfg(cfg)
    mode = str(cnn.get("use_in_inference", "auto")).lower()
    if mode == "false" or mode == "0":
        return False
    if mode == "true" or mode == "1":
        return cnn_checkpoints_exist(cfg)
    return cnn_checkpoints_exist(cfg)


def _load_sed_bundle(cfg: Optional[dict] = None) -> Optional[dict[str, Any]]:
    global _sed_bundle
    if _sed_bundle is not None:
        return _sed_bundle
    with _lock:
        if _sed_bundle is not None:
            return _sed_bundle
        cnn = _alm_cnn_cfg(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        merged = _resolve(cnn.get("merged_checkpoint", "outputs/alm_cnn_merged.pt"))
        if merged.is_file():
            data = torch.load(merged, map_location="cpu", weights_only=False)
            ckpt = data.get("sed")
        else:
            sed_path = _resolve(cnn.get("sed_checkpoint", "outputs/sed_cnn.pt"))
            if not sed_path.is_file():
                return None
            metrics = _resolve(cnn.get("sed_metrics_json", "outputs/sed_metrics.json"))
            ckpt = _load_ckpt_file(sed_path, metrics)
        if not ckpt or "model" not in ckpt:
            return None
        _sed_bundle = _bundle_from_ckpt(ckpt, device)
        logger.info("Loaded SED CNN (%d classes)", len(_sed_bundle["class_names"]))
        return _sed_bundle


def _load_emo_bundle(cfg: Optional[dict] = None) -> Optional[dict[str, Any]]:
    global _emo_bundle
    if _emo_bundle is not None:
        return _emo_bundle
    with _lock:
        if _emo_bundle is not None:
            return _emo_bundle
        cnn = _alm_cnn_cfg(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        merged = _resolve(cnn.get("merged_checkpoint", "outputs/alm_cnn_merged.pt"))
        if merged.is_file():
            data = torch.load(merged, map_location="cpu", weights_only=False)
            ckpt = data.get("emotion")
        else:
            emo_path = _resolve(cnn.get("emotion_checkpoint", "outputs/emotion_cnn.pt"))
            if not emo_path.is_file():
                return None
            metrics = _resolve(cnn.get("emotion_metrics_json", "outputs/emotion_metrics.json"))
            ckpt = _load_ckpt_file(emo_path, metrics)
        if not ckpt or "model" not in ckpt:
            return None
        _emo_bundle = _bundle_from_ckpt(ckpt, device)
        logger.info("Loaded emotion CNN (%d classes)", len(_emo_bundle["class_names"]))
        return _emo_bundle


def warmup_cnn(cfg: Optional[dict] = None) -> bool:
    """Load CNN weights at startup. Returns True if SED CNN is ready."""
    if not should_use_cnn(cfg):
        return False
    return _load_sed_bundle(cfg) is not None
