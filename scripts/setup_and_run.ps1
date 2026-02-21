# GenAI PCB Platform - Setup and run everything
# Run from repo root: .\scripts\setup_and_run.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path "$Root\src\main.py")) { $Root = (Get-Location).Path }
Set-Location $Root

Write-Host "=== 1. Ensure .env ===" -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example" -ForegroundColor Green
} else {
    Write-Host ".env exists" -ForegroundColor Green
}

Write-Host "`n=== 2. Install Python dependencies ===" -ForegroundColor Cyan
python -m pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host "Done" -ForegroundColor Green

Write-Host "`n=== 3. Run auth + key unit tests ===" -ForegroundColor Cyan
python -m pytest tests/unit/test_auth.py tests/unit/test_nlp_service.py -v -q --tb=line 2>&1 | Select-Object -Last 15
if ($LASTEXITCODE -ne 0) {
    Write-Host "Some tests failed (see above). Continuing to start server." -ForegroundColor Yellow
}

Write-Host "`n=== 4. Start backend (API) ===" -ForegroundColor Cyan
Write-Host "API: http://localhost:8000" -ForegroundColor Green
Write-Host "Docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop.`n" -ForegroundColor Gray
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
