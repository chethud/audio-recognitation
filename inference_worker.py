"""
Legacy inference worker entry point.

The original end-to-end HuBERT+LLM path was removed; all inference now runs through
the ALM-Lite modular pipeline (ASR + SED + emotion + LLM).
"""
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent
MODULAR = BASE / "inference_worker_modular.py"


def main() -> None:
    if not MODULAR.is_file():
        sys.stderr.write(f"Modular worker not found: {MODULAR}\n")
        sys.exit(1)
    raise SystemExit(
        subprocess.call([sys.executable, str(MODULAR), *sys.argv[1:]])
    )


if __name__ == "__main__":
    main()
