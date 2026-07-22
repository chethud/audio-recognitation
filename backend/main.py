"""
ALM Backend API - Robust, crash-resistant design.
Runs inference in-process with cached models (fast). Persistence: SQLite at data/alm.db.

POST /analyze always runs fresh inference on the uploaded bytes. SQLite is
write-only for history — it is never read to build the /analyze response.
"""
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import asyncio
import uuid
from pathlib import Path

from src.env_setup import configure_ml_env

configure_ml_env()

import yaml
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend import database
from backend import inference_service
from backend.models import (
    AnalyzeLogItem,
    AnalyzeResponse,
    AuthResponse,
    HealthResponse,
    HistoryItem,
    InferenceResponse,
    InferenceResponseModular,
    LoginRequest,
    SignupRequest,
    UserPublic,
)
from src.config_path import low_memory_mode, resolve_config_path

_auth_bearer = HTTPBearer(auto_error=False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_auth_bearer),
) -> dict:
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not credentials.credentials
    ):
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = database.get_user_by_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


class UTF8JSONResponse(JSONResponse):
    """JSON response that keeps non-ASCII characters (e.g. Hindi, Tamil) as-is for correct display."""

    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False).encode("utf-8")


app = FastAPI(
    title="ALM API",
    description="Audio Language Model - Listen, Think, Understand",
    default_response_class=UTF8JSONResponse,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE = Path(__file__).parent.parent
CONFIG_PATH = resolve_config_path(BASE)
WORKER_SCRIPT_MODULAR = BASE / "inference_worker_modular.py"

# Limits (audio still capped by config data.max_audio_length_sec on Render)
MAX_FILE_SIZE_MB = 10 if low_memory_mode() else 500
MAX_QUESTION_LEN = 1000
INFERENCE_TIMEOUT_SEC = 600 if low_memory_mode() else 3600
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4", ".mkv", ".webm", ".avi", ".mov"}


def _alm_settings() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f).get("alm_lite", {})
    except Exception:
        return {}


def _save_analyze_log_async(
    *,
    audio_filename: str,
    question: str,
    transcript: str,
    sounds: list[str],
    emotion: str,
    answer: str,
    audio_data: bytes | None,
    audio_mime: str | None,
) -> None:
    try:
        database.save_analyze_log(
            audio_filename=audio_filename,
            question=question,
            transcript=transcript,
            sounds=sounds,
            emotion=emotion,
            answer=answer,
            audio_data=audio_data,
            audio_mime=audio_mime,
        )
    except Exception as e:
        logger.warning("SQLite analyze_logs save failed: %s", e)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    logger.exception("Unhandled exception")
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {str(exc)}"})


def _worker_env(
    language: str | None = None,
    *,
    upload_name: str = "",
    temp_name: str = "",
    audio_sha256: str = "",
    audio_bytes: int | None = None,
) -> dict:
    """Env for the inference subprocess (ffmpeg + thread limits + audio identity)."""
    env = {
        **os.environ,
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "TRANSFORMERS_NO_TF": "1",
        "USE_TF": "0",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "ALM_ASR_LANGUAGE": (language or "").strip(),
        "ALM_UPLOAD_NAME": (upload_name or "").strip(),
        "ALM_TEMP_NAME": (temp_name or "").strip(),
        "ALM_AUDIO_SHA256": (audio_sha256 or "").strip(),
        "ALM_AUDIO_BYTES": str(int(audio_bytes) if audio_bytes is not None else ""),
        # No transcript/result memoization. Model weights may stay in memory.
        "ALM_DISABLE_TRANSCRIPT_CACHE": "1",
        # CT2 off on Render low-memory by default (extra model copy).
        "ALM_ENABLE_CT2": os.environ.get(
            "ALM_ENABLE_CT2",
            "0" if low_memory_mode() else "1",
        ),
    }
    ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE", "").strip()
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe() or ""
        except Exception:
            ffmpeg = ""
    if ffmpeg:
        env["IMAGEIO_FFMPEG_EXE"] = ffmpeg
        bin_dir = str(Path(ffmpeg).resolve().parent)
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    return env


def _clean_worker_stderr(stderr: str) -> str:
    """Drop HF rate-limit noise; keep last [alm-worker] stage for crash reports."""
    lines = []
    last_stage = ""
    for line in (stderr or "").splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if "unauthenticated requests to the hf hub" in low:
            continue
        if "hf_token" in low and "rate limit" in low:
            continue
        if s.startswith("[alm-worker]"):
            last_stage = s
            continue
        lines.append(s)
    tip = ""
    if last_stage:
        tip = f" Last stage: {last_stage.replace('[alm-worker] ', '')}."
    body = " ".join(lines)[:280]
    return f"{tip} {body}".strip()


def _run_worker(
    script_path: Path,
    audio_path: str,
    question: str,
    output_path: str,
    question_file: str,
    language: str | None = None,
    *,
    upload_name: str = "",
    audio_sha256: str = "",
    audio_bytes: int | None = None,
) -> dict:
    """Run inference worker subprocess. Passes question via file to avoid Windows arg issues."""
    import json
    if not script_path.exists():
        return {"ok": False, "error": f"Worker script not found: {script_path}"}
    try:
        import threading
        import time

        process = subprocess.Popen(
            [sys.executable, str(script_path), audio_path, output_path, "--question-file", question_file],
            cwd=str(BASE),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_worker_env(
                language,
                upload_name=upload_name,
                temp_name=Path(audio_path).name,
                audio_sha256=audio_sha256,
                audio_bytes=audio_bytes,
            ),
        )

        stdout_lines = []
        stderr_lines = []

        def read_stream(stream, lines, stream_type):
            try:
                for line in stream:
                    lines.append(line)
                    text = f"[worker-{stream_type}] {line}"
                    if stream_type == "stderr":
                        try:
                            sys.stderr.write(text)
                            sys.stderr.flush()
                        except UnicodeEncodeError:
                            sys.stderr.write(text.encode("ascii", "replace").decode("ascii"))
                            sys.stderr.flush()
                    else:
                        try:
                            sys.stdout.write(text)
                            sys.stdout.flush()
                        except UnicodeEncodeError:
                            sys.stdout.write(text.encode("ascii", "replace").decode("ascii"))
                            sys.stdout.flush()
            except Exception:
                pass

        t_out = threading.Thread(target=read_stream, args=(process.stdout, stdout_lines, "stdout"))
        t_err = threading.Thread(target=read_stream, args=(process.stderr, stderr_lines, "stderr"))
        t_out.daemon = True
        t_err.daemon = True
        t_out.start()
        t_err.start()

        start_time = time.time()
        while process.poll() is None:
            if time.time() - start_time > INFERENCE_TIMEOUT_SEC:
                process.kill()
                t_out.join(timeout=1.0)
                t_err.join(timeout=1.0)
                raise subprocess.TimeoutExpired(process.args, INFERENCE_TIMEOUT_SEC)
            time.sleep(0.2)

        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        returncode = process.returncode

        # Native crash (access violation) => no output file / non-zero return
        if returncode != 0 and not Path(output_path).exists():
            detail = _clean_worker_stderr(stderr or "")
            return {
                "ok": False,
                "error": (
                    f"Inference worker crashed (exit {returncode}). "
                    f"Try again, or pick English/Kannada explicitly."
                    + (f" {detail}" if detail else "")
                ).strip(),
            }
        if Path(output_path).exists():
            data = json.loads(Path(output_path).read_text(encoding="utf-8"))
        else:
            stderr_cleaned = (stderr or "").strip()
            stdout_cleaned = (stdout or "").strip()
            err = stderr_cleaned or stdout_cleaned or "Worker produced no output file (possible crash during startup)"
            if stderr_cleaned and stdout_cleaned:
                err = f"{err}\n[stdout: {stdout_cleaned[:200]}]"
            return {"ok": False, "error": err[:800]}
        if not data.get("ok"):
            return {"ok": False, "error": data.get("error", "Unknown worker error")}
        return data
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Inference timed out"}
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"Invalid worker output: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)}"}


def _use_subprocess_inference() -> bool:
    """Isolate ML in a child process so native crashes cannot kill the API."""
    env = os.getenv("ALM_SUBPROCESS_INFERENCE", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    # Windows: subprocess avoids native segfaults taking down uvicorn.
    if sys.platform == "win32":
        return True
    # Render low-RAM: in-process (one copy of torch). Parent+child OOMs on Standard.
    return False


def _run_modular_inference(
    audio_path: str,
    question: str,
    output_path: str,
    question_file: str,
    language: str | None = None,
    *,
    upload_name: str = "",
    audio_sha256: str = "",
    audio_bytes: int | None = None,
) -> dict:
    """Run ALM-Lite; subprocess on Windows by default (crash-isolated)."""
    if _use_subprocess_inference():
        return _run_worker(
            WORKER_SCRIPT_MODULAR,
            audio_path,
            question,
            output_path,
            question_file,
            language=language,
            upload_name=upload_name,
            audio_sha256=audio_sha256,
            audio_bytes=audio_bytes,
        )
    if not inference_service.is_ready():
        if low_memory_mode():
            try:
                logger.info("Low-memory: warming models for this request…")
                inference_service.warmup()
            except Exception as exc:
                return {
                    "ok": False,
                    "error": f"Model load failed (low memory): {type(exc).__name__}: {exc}",
                }
        else:
            raise HTTPException(
                503,
                "Models are still loading. Wait until /health returns model_ready=true, then retry.",
            )
    # Propagate identity into in-process path (same env keys the worker uses).
    if upload_name:
        os.environ["ALM_UPLOAD_NAME"] = upload_name
    os.environ["ALM_TEMP_NAME"] = Path(audio_path).name
    if audio_sha256:
        os.environ["ALM_AUDIO_SHA256"] = audio_sha256
    if audio_bytes is not None:
        os.environ["ALM_AUDIO_BYTES"] = str(int(audio_bytes))
    if language:
        os.environ["ALM_ASR_LANGUAGE"] = language
    return inference_service.analyze_file(audio_path, question, language=language)


def _run_inference_worker_modular(
    audio_path: str,
    question: str,
    output_path: str,
    question_file: str,
    language: str | None = None,
    *,
    upload_name: str = "",
    audio_sha256: str = "",
    audio_bytes: int | None = None,
) -> dict:
    return _run_modular_inference(
        audio_path,
        question,
        output_path,
        question_file,
        language,
        upload_name=upload_name,
        audio_sha256=audio_sha256,
        audio_bytes=audio_bytes,
    )


def _safe_upload_stem(filename: str) -> str:
    stem = Path(filename or "audio").stem
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")
    return (stem[:40] or "audio")


def _sound_events_to_labels(sound_events: list) -> list[str]:
    """SED output -> simple label list for API /analyze."""
    if not sound_events:
        return []
    out: list[str] = []
    for e in sound_events:
        if isinstance(e, dict) and e.get("label"):
            out.append(str(e["label"]))
        elif isinstance(e, str):
            out.append(e)
    return out


def _mime_for_suffix(suffix: str) -> str:
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".webm": "audio/webm",
    }.get(suffix.lower(), "application/octet-stream")


@app.on_event("startup")
def startup():
    database.init_db()
    if _use_subprocess_inference():
        logger.info("Subprocess inference enabled — models load per request in a worker.")
        return

    if low_memory_mode():
        # Do not preload: keep idle RSS low. First /analyze calls warmup().
        logger.info(
            "Low-memory mode (%s): defer model load until first /analyze "
            "(whisper-tiny / ASR-only).",
            resolve_config_path(BASE).name,
        )
        return

    def _warmup_background() -> None:
        logger.info(
            "Loading AI models in background (Whisper + SED + emotion). "
            "Port is open; /health will show model_ready=true when done."
        )
        try:
            inference_service.warmup()
            logger.info("All models loaded — ready for analysis.")
        except Exception:
            logger.exception("Model warmup failed")

    threading.Thread(target=_warmup_background, daemon=True).start()


@app.get("/")
def root():
    return {
        "service": "ALM-Lite API",
        "health": "/health",
        "docs": "/docs",
        "model_ready": _health_ready(),
        "config": resolve_config_path(BASE).name if low_memory_mode() else "config.yaml",
    }


def _health_ready() -> bool:
    if _use_subprocess_inference():
        return True
    # Low-memory: ready to accept traffic; models load on first analyze.
    if low_memory_mode():
        return True
    return inference_service.is_ready()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_ready=_health_ready())


@app.head("/")
def root_head():
    """Render port probes sometimes use HEAD /."""
    return Response(status_code=200)


@app.post("/auth/signup", response_model=AuthResponse)
def auth_signup(body: SignupRequest):
    try:
        uid = database.create_user(body.email, body.password)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered") from None
    token = database.create_session(uid)
    row = database.get_user_by_id(uid)
    if not row:
        raise HTTPException(status_code=500, detail="User create failed")
    return AuthResponse(
        access_token=token,
        user=UserPublic(id=row["id"], email=row["email"]),
    )


@app.post("/auth/login", response_model=AuthResponse)
def auth_login(body: LoginRequest):
    row = database.get_user_by_email(body.email)
    if not row:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not database.verify_password(
        body.password, row["password_salt"], row["password_hash"]
    ):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = database.create_session(int(row["id"]))
    return AuthResponse(
        access_token=token,
        user=UserPublic(id=int(row["id"]), email=row["email"]),
    )


@app.get("/auth/me", response_model=UserPublic)
def auth_me(user: dict = Depends(_get_current_user)):
    return UserPublic(id=int(user["id"]), email=user["email"])


@app.post("/auth/logout")
def auth_logout(credentials: HTTPAuthorizationCredentials | None = Depends(_auth_bearer)):
    if credentials and credentials.scheme.lower() == "bearer" and credentials.credentials:
        database.delete_session(credentials.credentials)
    return {"ok": True}


@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
    question: str = Form("What can be inferred from the audio?"),
    use_modular: bool = Form(
        False,
        description="When true, include transcript, sound_events, emotion, and context in the response",
    ),
):
    # Validate
    if not file.filename or not file.filename.strip():
        raise HTTPException(400, "No file provided")
    if not question or not question.strip():
        raise HTTPException(400, "Question cannot be empty")
    if len(question) > MAX_QUESTION_LEN:
        raise HTTPException(400, f"Question too long (max {MAX_QUESTION_LEN} chars)")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"File too large (max {MAX_FILE_SIZE_MB}MB)")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        unique_name = f"{uuid.uuid4().hex}_{_safe_upload_stem(file.filename)}{suffix}"
        tmp_path = tmpdir / unique_name
        tmp_path.write_bytes(content)
        output_path = tmpdir / "result.json"
        qfile = tmpdir / "question.txt"
        qfile.write_text(question.strip(), encoding="utf-8")

        result = _run_inference_worker_modular(
            str(tmp_path),
            question.strip(),
            str(output_path),
            str(qfile),
            upload_name=file.filename or unique_name,
            audio_sha256=__import__("hashlib").sha256(content).hexdigest(),
            audio_bytes=len(content),
        )

    if not result.get("ok"):
        logger.warning("Inference failed: %s", result.get("error"))
        raise HTTPException(status_code=500, detail=result.get("error", "Inference failed"))

    answer = result.get("answer", "")
    if not answer:
        raise HTTPException(500, "Empty answer from model")

    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        sr = cfg.get("data", {}).get("sample_rate", 16000)
        max_sec = cfg.get("data", {}).get("max_audio_length_sec", 30)
        raw_dur = len(content) / max(sr, 1)
        duration_sec = min(raw_dur, max_sec) if max_sec and max_sec > 0 else raw_dur
        row_id = database.save_inference(
            audio_filename=file.filename,
            question=question.strip(),
            answer=answer,
            audio_duration_sec=duration_sec,
            audio_data=content,
            audio_mime=_mime_for_suffix(suffix),
        )
    except Exception as e:
        logger.warning("DB save failed (inference still returned): %s", e)
        row_id = 0

    if use_modular:
        return InferenceResponseModular(
            id=row_id,
            answer=answer,
            question=question.strip(),
            audio_filename=file.filename,
            transcript=result.get("transcript", ""),
            sound_events=result.get("sound_events", []),
            emotion=result.get("emotion", ""),
            context=result.get("context", ""),
        )
    return InferenceResponse(
        id=row_id,
        answer=answer,
        question=question.strip(),
        audio_filename=file.filename,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    question: str = Form("What can be inferred from the audio?"),
    language: str = Form(
        "",
        description="ASR language code (e.g. 'en' or 'kn'). Empty = auto-detect.",
    ),
):
    """
    ALM-Lite production endpoint: ASR + SED + emotion + LLM.
    Results and uploaded audio are stored in SQLite (`analyze_logs`).
    """
    if not file.filename or not file.filename.strip():
        raise HTTPException(400, "No file provided")
    if not question or not question.strip():
        raise HTTPException(400, "Question cannot be empty")
    if len(question) > MAX_QUESTION_LEN:
        raise HTTPException(400, f"Question too long (max {MAX_QUESTION_LEN} chars)")

    language = (language or "").strip().lower()
    if language and language not in {"en", "kn"}:
        # Ignore unsupported codes rather than failing — fall back to auto-detect.
        language = ""

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"File too large (max {MAX_FILE_SIZE_MB}MB)")

    from src.asr.audio_trace import log_audio_identity, sha256_bytes

    audio_sha256 = sha256_bytes(content)
    unique_name = f"{uuid.uuid4().hex}_{_safe_upload_stem(file.filename)}{suffix}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_path = tmpdir / unique_name
        tmp_path.write_bytes(content)
        # Verify bytes on disk match the upload (guards against wrong-file reuse).
        on_disk = tmp_path.read_bytes()
        if len(on_disk) != len(content) or sha256_bytes(on_disk) != audio_sha256:
            raise HTTPException(500, "Temp file write verification failed")
        output_path = tmpdir / f"result_{uuid.uuid4().hex}.json"
        qfile = tmpdir / f"question_{uuid.uuid4().hex}.txt"
        qfile.write_text(question.strip(), encoding="utf-8")

        log_audio_identity(
            stage="upload",
            upload_name=file.filename or "",
            temp_path=tmp_path,
            file_bytes=len(content),
            file_sha256=audio_sha256,
            language=language or "auto",
        )

        result = await asyncio.to_thread(
            _run_inference_worker_modular,
            str(tmp_path),
            question.strip(),
            str(output_path),
            str(qfile),
            language or None,
            upload_name=file.filename or unique_name,
            audio_sha256=audio_sha256,
            audio_bytes=len(content),
        )

    if not result.get("ok"):
        logger.warning("Analyze failed: %s", result.get("error"))
        raise HTTPException(status_code=500, detail=result.get("error", "Inference failed"))

    answer = result.get("answer", "")
    if not answer:
        raise HTTPException(500, "Empty answer from model")

    transcript = result.get("transcript", "") or ""
    transcript_original = result.get("transcript_original", "") or ""
    language = result.get("language", "en") or "en"
    language_name = result.get("language_name", "English") or "English"
    languages = result.get("languages") or []
    language_names = result.get("language_names") or []
    sound_events = result.get("sound_events") or []
    sounds = _sound_events_to_labels(sound_events)
    sound_details = [
        {"label": e.get("label", ""), "score": e.get("score", 0)}
        for e in sound_events
        if isinstance(e, dict) and e.get("label")
    ]
    emotion = result.get("emotion", "") or "neutral"
    speaker_emotions = result.get("speaker_emotions") or {}
    speaker_turns = result.get("speaker_turns") or []
    num_speakers = int(result.get("num_speakers") or 0)
    detected_speakers = result.get("detected_speakers") or []
    if not detected_speakers and speaker_turns:
        seen = []
        for t in speaker_turns:
            sp = (t.get("speaker") or "").strip()
            if sp and sp not in seen:
                seen.append(sp)
        detected_speakers = seen
    if not speaker_emotions and emotion:
        for sp in (detected_speakers or ["Speaker 1"]):
            speaker_emotions[sp] = emotion
    formatted_transcript = result.get("formatted_transcript") or ""
    summary = result.get("summary") or answer

    alm = _alm_settings()
    store_audio = bool(alm.get("store_uploaded_audio", False))
    background_tasks.add_task(
        _save_analyze_log_async,
        audio_filename=file.filename,
        question=question.strip(),
        transcript=transcript,
        sounds=sounds,
        emotion=emotion,
        answer=answer,
        audio_data=content if store_audio else None,
        audio_mime=_mime_for_suffix(suffix) if store_audio else None,
    )

    return AnalyzeResponse(
        transcript=transcript,
        transcript_original=transcript_original,
        language=language,
        language_name=language_name,
        languages=languages,
        language_names=language_names,
        sounds=sounds,
        sound_details=sound_details,
        emotion=emotion,
        speaker_emotions=speaker_emotions,
        answer=answer,
        question=question.strip(),
        audio_filename=file.filename,
        log_id=None,
        speaker_turns=speaker_turns,
        num_speakers=num_speakers,
        detected_speakers=detected_speakers,
        formatted_transcript=formatted_transcript,
        summary=summary,
        audio_sha256=audio_sha256,
        audio_bytes=len(content),
        temp_filename=unique_name,
        wav_sha256=str(result.get("wav_sha256") or ""),
    )


@app.get("/analyze/history", response_model=list[AnalyzeLogItem])
def analyze_history(limit: int = 50, offset: int = 0):
    """List recent ``POST /analyze`` runs stored in SQLite."""
    limit = min(max(1, limit), 100)
    offset = max(0, offset)
    rows = database.get_analyze_logs(limit=limit, offset=offset)
    return [AnalyzeLogItem(**r) for r in rows]


@app.get("/analyze/history/{log_id}", response_model=AnalyzeLogItem)
def analyze_history_item(log_id: int):
    row = database.get_analyze_by_id(log_id)
    if not row:
        raise HTTPException(404, "Not found")
    return AnalyzeLogItem(**row)


@app.get("/analyze/history/{log_id}/audio")
def analyze_history_audio(log_id: int):
    """Download the uploaded audio file stored in SQLite for an analyze log."""
    stored = database.get_analyze_audio(log_id)
    if not stored:
        raise HTTPException(404, "Audio not found for this log")
    data, mime, filename = stored
    return Response(
        content=data,
        media_type=mime,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.get("/history", response_model=list[HistoryItem])
def history(limit: int = 50, offset: int = 0):
    limit = min(max(1, limit), 100)
    offset = max(0, offset)
    rows = database.get_inferences(limit=limit, offset=offset)
    return [HistoryItem(**dict(r)) for r in rows]


@app.get("/history/{inference_id}")
def get_inference(inference_id: int):
    row = database.get_inference_by_id(inference_id)
    if not row:
        raise HTTPException(404, "Not found")
    return row
