"""
ALM-Lite modular inference worker: ASR → SED → Context + LLM.
Runs in subprocess; writes JSON with transcript, sound_events, context, answer.
"""
import json
import os
import sys
from pathlib import Path

from src.env_setup import configure_ml_env

configure_ml_env()

# PyTorch-only: avoid loading TF/Keras Whisper (Keras 3 breaks TF models in Transformers).
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))


def _stage(msg: str) -> None:
    """Print progress so parent can report last stage after a native crash."""
    line = f"[alm-worker] {msg}"
    print(line, file=sys.stderr, flush=True)


def main():
    if len(sys.argv) < 3:
        sys.stderr.write(
            "Usage: inference_worker_modular.py <audio_path> <output_json_path> [--question-file <path>]\n"
        )
        sys.exit(1)

    args = sys.argv[1:]
    if "--question-file" in args:
        i = args.index("--question-file")
        question = Path(args[i + 1]).read_text(encoding="utf-8").strip() if i + 1 < len(args) else "What can be inferred from the audio?"
        args = [a for j, a in enumerate(args) if j != i and j != i + 1]
    else:
        question = args[2] if len(args) > 2 else "What can be inferred from the audio?"
        args = args[:2]

    audio_path = args[0]
    output_path = args[1] if len(args) > 1 else None
    if not output_path:
        sys.stderr.write("Missing output path\n")
        sys.exit(1)

    def write_out(obj):
        Path(output_path).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

    if not Path(audio_path).exists():
        write_out({"ok": False, "error": f"File not found: {audio_path}"})
        sys.exit(1)

    try:
        _stage("importing")
        import torch
        import yaml

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        from src.pipeline import run_alm_lite
        from src.utils import load_audio_from_file

        config_path = BASE / "config.yaml"
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        data_cfg = cfg.get("data", {})
        alm = cfg.get("alm_lite", {})
        fast = bool(alm.get("fast_mode", False))
        asr_cfg = alm.get("asr", {})
        sed_cfg = alm.get("sed", {})
        llm_cfg = alm.get("llm", {})
        emo_cfg = alm.get("emotion", {})
        dia_cfg = alm.get("diarization", {})

        _stage("loading_audio")
        from src.asr.audio_trace import log_audio_identity, sha256_file, sha256_waveform

        file_sha = (os.environ.get("ALM_AUDIO_SHA256") or "").strip() or sha256_file(audio_path)
        upload_name = (os.environ.get("ALM_UPLOAD_NAME") or "").strip() or Path(audio_path).name
        audio = load_audio_from_file(
            audio_path,
            sr=data_cfg.get("sample_rate", 16000),
            max_sec=data_cfg.get("max_audio_length_sec", 12),
        )
        # (1, L) -> (L,) for pipeline
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        wav_np = audio.numpy()
        wav_sha = sha256_waveform(wav_np)
        log_audio_identity(
            stage="worker_loaded",
            upload_name=upload_name,
            temp_path=audio_path,
            file_bytes=Path(audio_path).stat().st_size,
            file_sha256=file_sha,
            language=os.environ.get("ALM_ASR_LANGUAGE", "") or "auto",
            wav=wav_np,
            sample_rate=int(data_cfg.get("sample_rate", 16000)),
        )

        max_sec = data_cfg.get("max_audio_length_sec", 12)
        asr_language = (
            os.environ.get("ALM_ASR_LANGUAGE", "").strip()
            or asr_cfg.get("language")
            or None
        )
        _stage(
            f"run_pipeline audio_sec≈{audio.numel() / float(data_cfg.get('sample_rate', 16000)):.1f} "
            f"lang={asr_language or 'auto'} fast={fast} sha={file_sha[:12]}"
        )
        result = run_alm_lite(
            wav_np,
            question,
            sample_rate=data_cfg.get("sample_rate", 16000),
            asr_model_id=asr_cfg.get("model_id", "openai/whisper-tiny"),
            asr_language=asr_language,
            sed_model_id=sed_cfg.get("model_id", "MIT/ast-finetuned-audioset-10-10-0.4593"),
            sed_top_k=sed_cfg.get("top_k", 10),
            sed_threshold=sed_cfg.get("threshold", 0.04),
            sed_segment_sec=sed_cfg.get("segment_sec", 2.5),
            sed_max_windows=sed_cfg.get("max_windows", 12),
            sed_max_results=sed_cfg.get("max_results", 12),
            asr_segment_sec=asr_cfg.get("segment_sec", 4.0),
            asr_max_segments=asr_cfg.get("max_segments", 2),
            diarization_enabled=bool(dia_cfg.get("enabled", False)),
            diarization_cfg=dia_cfg,
            llm_model_id=llm_cfg.get("model_id", "Qwen/Qwen2-0.5B-Instruct"),
            max_new_tokens=llm_cfg.get("max_new_tokens", 32),
            repetition_penalty=llm_cfg.get("repetition_penalty", 1.1),
            no_repeat_ngram_size=llm_cfg.get("no_repeat_ngram_size", 2),
            emotion_model_id=emo_cfg.get("model_id"),
            emotion_enabled=emo_cfg.get("enabled", True),
            sed_enabled=sed_cfg.get("enabled", True),
            llm_enabled=llm_cfg.get("enabled", True),
            fast_mode=fast,
            max_duration_sec=max_sec if max_sec and max_sec > 0 else None,
            sed_backend=sed_cfg.get("backend", "cnn"),
            emotion_backend=emo_cfg.get("backend", "auto"),
            parallel=bool(alm.get("parallel_inference", False)),
        )

        transcript_preview = result.get("transcript_original") or result.get("transcript") or ""
        log_audio_identity(
            stage="worker_asr_done",
            upload_name=upload_name,
            temp_path=audio_path,
            file_bytes=Path(audio_path).stat().st_size,
            file_sha256=file_sha,
            language=str(result.get("language") or asr_language or "auto"),
            wav=wav_np,
            sample_rate=int(data_cfg.get("sample_rate", 16000)),
            whisper_out=transcript_preview,
        )

        _stage("writing_result")
        write_out({
            "ok": True,
            "answer": result["answer"],
            "summary": result.get("summary", result["answer"]),
            "transcript": result["transcript"],
            "transcript_original": result.get("transcript_original", ""),
            "formatted_transcript": result.get("formatted_transcript", ""),
            "language": result.get("language", "en"),
            "language_name": result.get("language_name", "English"),
            "languages": result.get("languages", []),
            "language_names": result.get("language_names", []),
            "speaker_turns": result.get("speaker_turns", []),
            "num_speakers": result.get("num_speakers", 0),
            "detected_speakers": result.get("detected_speakers", []),
            "sound_events": result["sound_events"],
            "emotion": result.get("emotion", "neutral"),
            "context": result["context"],
            "audio_sha256": file_sha,
            "wav_sha256": wav_sha,
            "temp_filename": Path(audio_path).name,
            "upload_name": upload_name,
        })
        _stage("done")
    except Exception as e:
        write_out({"ok": False, "error": f"{type(e).__name__}: {str(e)}"})
        sys.exit(1)


if __name__ == "__main__":
    main()
