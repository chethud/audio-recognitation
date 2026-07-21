# ALM-Lite API — Render / Docker
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# PyAnnote diarization reads HF_TOKEN at runtime (set in Render/Vercel dashboard or docker run -e).
# Accept https://huggingface.co/pyannote/speaker-diarization-3.1 before first deploy.
ENV PYTHONUNBUFFERED=1 \
    TORCHDYNAMO_DISABLE=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_TOKEN="" \
    ALM_SUBPROCESS_INFERENCE=1 \
    ALM_ENABLE_CT2=0

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
