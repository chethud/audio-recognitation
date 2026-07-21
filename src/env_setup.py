"""Set ML/HuggingFace env vars before any transformers import."""
from __future__ import annotations

import logging
import os
from pathlib import Path


def _ensure_ffmpeg_on_path() -> None:
    """Make bundled imageio-ffmpeg visible to subprocesses and shutil.which."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if not exe or not Path(exe).is_file():
            return
        bin_dir = str(Path(exe).resolve().parent)
        path = os.environ.get("PATH", "")
        if bin_dir.lower() not in path.lower():
            os.environ["PATH"] = bin_dir + os.pathsep + path
        # Parent may pass IMAGEIO_FFMPEG_EXE="" — override empty values.
        if not os.environ.get("IMAGEIO_FFMPEG_EXE", "").strip():
            os.environ["IMAGEIO_FFMPEG_EXE"] = exe
    except Exception:
        pass


def configure_ml_env() -> None:
    # Prefer project-local HF cache. On Render, allow downloads (ephemeral disk / lean models).
    src_dir = Path(__file__).resolve().parent
    workspace_root = src_dir.parent
    local_hf_cache = workspace_root / ".cache" / "huggingface"
    on_render = (os.environ.get("RENDER") or "").strip().lower() in ("1", "true", "yes")
    allow_download = on_render or (os.environ.get("ALM_ALLOW_HF_DOWNLOAD") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    os.environ.setdefault("HF_HOME", str(local_hf_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    if allow_download:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
    else:
        # Local: prefer offline when models are already cached.
        os.environ["HF_HOME"] = str(local_hf_cache)
        os.environ["HF_HUB_CACHE"] = str(local_hf_cache / "hub")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Single-thread BLAS/OpenMP — lower RAM + fewer native crashes on Windows/Render.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    _ensure_ffmpeg_on_path()

    for name in (
        "transformers",
        "huggingface_hub",
        "huggingface_hub.utils._http",
        "filelock",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
