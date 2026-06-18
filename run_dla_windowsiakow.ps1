# Run with: powershell -ExecutionPolicy Bypass -File run_dla_windowsiakow.ps1

Start-Process "https://www.linux.org/pages/download/"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv      = "$ScriptDir\.venv_suml"
$ModelsDir = "$ScriptDir\models"

$Python    = "$Venv\Scripts\python.exe"
$Uvicorn   = "$Venv\Scripts\uvicorn.exe"
$Streamlit = "$Venv\Scripts\streamlit.exe"

New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

# --- TRAINER ---
Write-Host ">>> Running trainer..."
$env:PYTHONPATH   = $ScriptDir
$env:MODEL_DIR    = $ModelsDir
$env:DATASET_DIR  = "$ScriptDir\data\raw"
& $Python "$ScriptDir\backend\train\train.py"
Write-Host ">>> Trainer done."

# --- BACKEND ---
Write-Host ">>> Starting backend..."
$env:PYTHONPATH  = "$ScriptDir\backend"
$env:MODEL_DIR   = $ModelsDir
$BackendProc = Start-Process -FilePath $Uvicorn `
    -ArgumentList "app.main:app", "--host", "0.0.0.0", "--port", "8000" `
    -PassThru -NoNewWindow

Write-Host ">>> Waiting for backend to become healthy..."
$healthy = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        Invoke-WebRequest -Uri "http://localhost:8000/health" `
            -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop | Out-Null
        $healthy = $true
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $healthy) {
    Write-Host "ERROR: Backend did not become healthy in time." -ForegroundColor Red
    Stop-Process -Id $BackendProc.Id -Force
    exit 1
}
Write-Host ">>> Backend healthy."

# --- FRONTEND ---
Write-Host ">>> Starting frontend..."
$env:BACKEND_URL = "http://localhost:8000"
$FrontendProc = Start-Process -FilePath $Streamlit `
    -ArgumentList "run", "$ScriptDir\frontend\app.py", "--server.headless=true" `
    -PassThru -NoNewWindow

Write-Host ""
Write-Host "========================================"
Write-Host "  Frontend : http://localhost:8501"
Write-Host "  API docs : http://localhost:8000/docs"
Write-Host "  Health   : http://localhost:8000/health"
Write-Host "========================================"
Write-Host "Press Enter to stop all services..."
Read-Host | Out-Null

Write-Host "Stopping services..."
Stop-Process -Id $BackendProc.Id  -Force -ErrorAction SilentlyContinue
Stop-Process -Id $FrontendProc.Id -Force -ErrorAction SilentlyContinue
Write-Host "Stopped."