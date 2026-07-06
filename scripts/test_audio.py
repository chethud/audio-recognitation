"""Quick local test for a single audio file."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.pipeline import run_alm_lite
from src.utils import load_audio_from_file


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not path or not Path(path).is_file():
        print(f"File not found: {path}")
        sys.exit(1)

    with open(BASE / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = cfg.get("data", {})
    alm = cfg.get("alm_lite", {})
    asr = alm.get("asr", {})
    sed = alm.get("sed", {})
    emo = alm.get("emotion", {})
    dia = alm.get("diarization", {})
    max_sec = data.get("max_audio_length_sec", 0)

    audio = load_audio_from_file(
        path, sr=data.get("sample_rate", 16000), max_sec=max_sec
    )
    if audio.dim() == 2:
        audio = audio.squeeze(0)

    sr = data.get("sample_rate", 16000)
    print(f"Audio seconds: {len(audio) / sr:.1f}")

    result = run_alm_lite(
        audio.numpy(),
        "What can be inferred from the audio?",
        sample_rate=sr,
        asr_model_id=asr.get("model_id", "openai/whisper-tiny"),
        asr_segment_sec=asr.get("segment_sec", 30),
        asr_max_segments=asr.get("max_segments", 1),
        diarization_enabled=bool(dia.get("enabled", False)),
        diarization_cfg=dia,
        sed_enabled=sed.get("enabled", True),
        sed_model_id=sed.get("model_id"),
        sed_top_k=sed.get("top_k", 3),
        sed_threshold=sed.get("threshold", 0.08),
        sed_segment_sec=sed.get("segment_sec", 3),
        sed_max_windows=sed.get("max_windows", 2),
        sed_backend=sed.get("backend", "auto"),
        emotion_enabled=emo.get("enabled", True),
        emotion_model_id=emo.get("model_id"),
        emotion_backend=emo.get("backend", "auto"),
        fast_mode=bool(alm.get("fast_mode", True)),
        max_duration_sec=max_sec if max_sec and max_sec > 0 else None,
    )

    print(f"NUM_SPEAKERS: {result.get('num_speakers', 0)}")
    print("--- SPEAKER TURNS ---")
    for t in result.get("speaker_turns", []):
        print(f"{t.get('speaker')}: {t.get('text')}")
    print("--- TRANSCRIPT ---")
    print(result.get("transcript", ""))
    print("--- ANSWER ---")
    print(result.get("answer", ""))


if __name__ == "__main__":
    main()
