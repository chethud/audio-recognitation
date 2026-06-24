"""Set ML/HuggingFace env vars before any transformers import."""
from __future__ import annotations

import logging
import os


def configure_ml_env() -> None:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    for name in ("transformers", "huggingface_hub", "filelock"):
        logging.getLogger(name).setLevel(logging.ERROR)
