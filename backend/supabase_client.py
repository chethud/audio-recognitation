"""
Optional Supabase: PostgreSQL (audio_logs) + Storage (audio-files bucket).
Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) to enable.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Optional

_BUCKET = "audio-files"


def is_configured() -> bool:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.environ.get("SUPABASE_ANON_KEY", "").strip()
    )
    return bool(url and key)


def _client():
    if not is_configured():
        return None
    from supabase import create_client

    url = os.environ["SUPABASE_URL"].strip()
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.environ["SUPABASE_ANON_KEY"].strip()
    )
    return create_client(url, key)


def upload_audio_bytes(
    filename: str,
    data: bytes,
    content_type: Optional[str] = None,
) -> Optional[str]:
    """
    Upload bytes to storage bucket `audio-files`. Returns public URL or storage path if available.
    """
    cli = _client()
    if not cli or not data:
        return None
    ext = Path(filename).suffix or ".wav"
    safe_name = f"{uuid.uuid4().hex}{ext}"
    path = f"uploads/{safe_name}"
    mime = content_type or "application/octet-stream"
    try:
        cli.storage.from_(_BUCKET).upload(path, data, file_options={"content-type": mime})
        pub = cli.storage.from_(_BUCKET).get_public_url(path)
        if pub:
            return pub
        return path
    except Exception:
        return None


def insert_audio_log(
    *,
    audio_url: Optional[str],
    transcript: str,
    sounds: list[str],
    emotion: str,
    answer: str,
    question: str,
) -> Optional[str]:
    """Insert one row into audio_logs. Returns row UUID as string if successful."""
    cli = _client()
    if not cli:
        return None
    row: dict[str, Any] = {
        "audio_url": audio_url or "",
        "transcript": transcript or "",
        "sounds": sounds,
        "emotion": emotion or "",
        "answer": answer or "",
        "question": question or "",
    }
    try:
        res = cli.table("audio_logs").insert(row).select("id").execute()
        if res.data and len(res.data) > 0 and res.data[0].get("id"):
            return str(res.data[0]["id"])
    except Exception:
        pass
    return None
