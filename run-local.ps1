# Start ALM-Lite locally (backend + frontend)
# Usage:  .\run-local.ps1
#         .\run-local.ps1 -Port 8002

param(
    [int]$Port = 8002
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

# Keep caches on D: (C: is often low on space)
$env:HF_HOME = Join-Path $Root ".cache\huggingface"
$env:PIP_CACHE_DIR = Join-Path $Root ".pip-cache"
$env:TMP = Join-Path $Root ".tmp"
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:PIP_CACHE_DIR, $env:TMP | Out-Null

# Prefer Python 3.11 venv (stable). Avoid system Python 3.14 — it crashes during Whisper.
$Python = $null
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $VenvPy) {
    & $VenvPy -c "import fastapi, torch, transformers" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $Python = $VenvPy
        Write-Host "Using .venv (Python 3.11)" -ForegroundColor Green
    }
}
if (-not $Python) {
    & py -3.11 -c "import fastapi" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $Python = "py"
        $PythonArgsPrefix = @("-3.11")
        Write-Host "Using system Python 3.11" -ForegroundColor Green
    }
}
if (-not $Python) {
    Write-Host @"
Python 3.11 with packages not ready yet.

If install is still running, wait for it to finish, then re-run:
  .\run-local.ps1

Or install now:
  `$env:PIP_CACHE_DIR = '$Root\.pip-cache'
  .\.venv\Scripts\pip.exe install -r requirements.txt
"@ -ForegroundColor Red
    exit 1
}

# Stop anything already on these ports
foreach ($p in @($Port, 5173)) {
    Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
}
Start-Sleep -Seconds 1

Write-Host "`nStarting backend on http://127.0.0.1:$Port ..." -ForegroundColor Cyan
# Isolate ML in a child process so native crashes cannot kill the API server.
$env:ALM_SUBPROCESS_INFERENCE = "1"
if ($Python -eq "py") {
    Start-Process -FilePath "py" -ArgumentList (@("-3.11", "run.py", "--host", "127.0.0.1", "--port", "$Port")) -WorkingDirectory $Root -WindowStyle Normal
} else {
    Start-Process -FilePath $Python -ArgumentList @("run.py", "--host", "127.0.0.1", "--port", "$Port") -WorkingDirectory $Root -WindowStyle Normal
}

Start-Sleep -Seconds 2

Write-Host "Starting frontend on http://localhost:5173/ ..." -ForegroundColor Cyan
Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "npm run dev -- --host" -WorkingDirectory (Join-Path $Root "frontend") -WindowStyle Normal

Write-Host @"

Local URLs:
  App:     http://localhost:5173/
  API:     http://127.0.0.1:$Port
  Health:  http://127.0.0.1:$Port/health

Refresh the browser after models finish loading (~1-2 min first start).
"@ -ForegroundColor Green
