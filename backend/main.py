"""
ALM Backend API - Robust, crash-resistant design.
Runs inference in-process with cached models (fast). Persistence: SQLite at data/alm.db.
"""
import json
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from src.env_setup import configure_ml_env

configure_ml_env()

import yaml
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
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
CONFIG_PATH = BASE / "config.yaml"
WORKER_SCRIPT_MODULAR = BASE / "inference_worker_modular.py"

# Limits (no audio length limit; full file is analyzed)
MAX_FILE_SIZE_MB = 500
MAX_QUESTION_LEN = 1000
INFERENCE_TIMEOUT_SEC = 3600  # 1 hour for long files (full-file analysis)
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4", ".mkv", ".webm", ".avi", ".mov"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    logger.exception("Unhandled exception")
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {str(exc)}"})


def _run_worker(script_path: Path, audio_path: str, question: str, output_path: str, question_file: str) -> dict:
    """Run inference worker subprocess. Passes question via file to avoid Windows arg issues."""
    import json
    if not script_path.exists():
        return {"ok": False, "error": f"Worker script not found: {script_path}"}
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), audio_path, output_path, "--question-file", question_file],
            cwd=str(BASE),
            capture_output=True,
            text=True,
            timeout=INFERENCE_TIMEOUT_SEC,
            env={
                **os.environ,
                "TF_CPP_MIN_LOG_LEVEL": "3",
                "TRANSFORMERS_NO_TF": "1",
                "USE_TF": "0",
                "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "TRANSFORMERS_VERBOSITY": "error",
            },
        )
        if Path(output_path).exists():
            data = json.loads(Path(output_path).read_text(encoding="utf-8"))
        else:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            err = stderr or stdout or "Worker produced no output file (possible crash during startup)"
            if stderr and stdout:
                err = f"{err}\n[stdout: {stdout[:200]}]"
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
    return os.getenv("ALM_SUBPROCESS_INFERENCE", "").strip().lower() in ("1", "true", "yes")


def _run_modular_inference(audio_path: str, question: str, output_path: str, question_file: str) -> dict:
    """Run ALM-Lite; in-process with cached models by default (fast)."""
    if _use_subprocess_inference():
        return _run_worker(WORKER_SCRIPT_MODULAR, audio_path, question, output_path, question_file)
    if not inference_service.is_ready():
        raise HTTPException(
            503,
            "Models are still loading. Wait until /health returns model_ready=true, then retry.",
        )
    return inference_service.analyze_file(audio_path, question)


def _run_inference_worker_modular(audio_path: str, question: str, output_path: str, question_file: str) -> dict:
    return _run_modular_inference(audio_path, question, output_path, question_file)


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
    if not _use_subprocess_inference():
        logger.info(
            "Loading AI models into memory (fast_mode loads Whisper only; "
            "wait ~30–90s on first start)…"
        )
        try:
            inference_service.warmup()
            logger.info("All models loaded — ready for fast inference.")
        except Exception:
            logger.exception("Model warmup failed")


@app.get("/health", response_model=HealthResponse)
def health():
    ready = inference_service.is_ready() or _use_subprocess_inference()
    return HealthResponse(status="ok", model_ready=ready)


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
        tmp_path = tmpdir / f"audio{suffix}"
        tmp_path.write_bytes(content)
        output_path = tmpdir / "result.json"
        qfile = tmpdir / "question.txt"
        qfile.write_text(question.strip(), encoding="utf-8")

        result = _run_inference_worker_modular(
            str(tmp_path), question.strip(), str(output_path), str(qfile)
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
    file: UploadFile = File(...),
    question: str = Form("What can be inferred from the audio?"),
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
        tmp_path = tmpdir / f"audio{suffix}"
        tmp_path.write_bytes(content)
        output_path = tmpdir / "result.json"
        qfile = tmpdir / "question.txt"
        qfile.write_text(question.strip(), encoding="utf-8")

        result = _run_inference_worker_modular(
            str(tmp_path), question.strip(), str(output_path), str(qfile)
        )

    if not result.get("ok"):
        logger.warning("Analyze failed: %s", result.get("error"))
        raise HTTPException(status_code=500, detail=result.get("error", "Inference failed"))

    answer = result.get("answer", "")
    if not answer:
        raise HTTPException(500, "Empty answer from model")

    transcript = result.get("transcript", "") or ""
    sounds = _sound_events_to_labels(result.get("sound_events") or [])
    emotion = result.get("emotion", "") or "neutral"

    sqlite_id = 0
    try:
        sqlite_id = database.save_analyze_log(
            audio_filename=file.filename,
            question=question.strip(),
            transcript=transcript,
            sounds=sounds,
            emotion=emotion,
            answer=answer,
            audio_data=content,
            audio_mime=_mime_for_suffix(suffix),
        )
    except Exception as e:
        logger.warning("SQLite analyze_logs save failed: %s", e)

    log_id_out = str(sqlite_id) if sqlite_id else None

    return AnalyzeResponse(
        transcript=transcript,
        sounds=sounds,
        emotion=emotion,
        answer=answer,
        question=question.strip(),
        audio_filename=file.filename,
        log_id=log_id_out,
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
