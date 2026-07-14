# ALM-Lite — WhisperX + PyAnnote diarization setup (Windows)
# Requires Python 3.9–3.13 (WhisperX does not support 3.14 yet).
#
# Usage:
#   $env:HF_TOKEN = "hf_..."
#   .\setup-diarization.ps1
#
# Or pass token:
#   .\setup-diarization.ps1 -HfToken "hf_..."

param(
    [string]$HfToken = "",
    [switch]$SkipPip
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Test-PythonVersion {
    $version = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found. Install Python 3.11 from https://www.python.org/downloads/"
    }
    if ($version -match "Python (\d+)\.(\d+)") {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -eq 3 -and $minor -ge 9 -and $minor -le 13) {
            Write-Host "Python OK: $version"
            return
        }
        if ($major -eq 3 -and $minor -ge 14) {
            throw @"
Python $version detected. WhisperX requires Python 3.9–3.13.

Options:
  1. Install Python 3.11 and create a venv:
       py -3.11 -m venv .venv
       .\.venv\Scripts\Activate.ps1
       .\setup-diarization.ps1

  2. Use Docker (see Dockerfile — already Python 3.11):
       docker build -t alm-lite .
"@
        }
    }
    Write-Warning "Could not parse Python version ($version). Continuing anyway."
}

Write-Step "Checking Python version"
Test-PythonVersion

Write-Step "Hugging Face token (PyAnnote)"
if ($HfToken) {
    $env:HF_TOKEN = $HfToken
}
if (-not $env:HF_TOKEN -and -not $env:HUGGINGFACE_TOKEN) {
    Write-Host @"
HF_TOKEN is not set. PyAnnote diarization requires a Hugging Face token.

1. Create a token: https://huggingface.co/settings/tokens
2. Accept model terms: https://huggingface.co/pyannote/speaker-diarization-3.1
3. Set:  `$env:HF_TOKEN = 'hf_...'

Diarization will fall back to legacy Wav2Vec2 without a token.
"@ -ForegroundColor Yellow
} else {
    if (-not $env:HF_TOKEN) { $env:HF_TOKEN = $env:HUGGINGFACE_TOKEN }
    Write-Host "HF_TOKEN is set (length $($env:HF_TOKEN.Length))"
}

if (-not $SkipPip) {
    Write-Step "Installing dependencies (whisperx, pyannote.audio, nltk)"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
}

Write-Step "Downloading NLTK punkt (WhisperX alignment)"
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); print('NLTK punkt ready')"

Write-Step "Verifying diarization stack"
python -c @"
from src.diarization.whisperx_pipeline import is_whisperx_available
from src.diarization.speaker_diarization import warmup_diarization

wx = is_whisperx_available()
print('WhisperX+PyAnnote available:', wx)
if wx:
    ok = warmup_diarization()
    print('WhisperX warmup:', ok)
else:
    print('Legacy Wav2Vec2 fallback will be used until WhisperX is installed + HF_TOKEN is set.')
"@

Write-Step "Done"
Write-Host @"

Next steps:
  1. Ensure config.yaml has:  alm_lite.diarization.backend: whisperx
  2. Start backend:  python run.py --host 127.0.0.1 --port 8002
  3. Upload audio at http://localhost:5173/ with English or Kannada selected

Render production: add HF_TOKEN as a secret env var (see render.yaml).
"@ -ForegroundColor Green
