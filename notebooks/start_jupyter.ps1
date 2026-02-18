$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runtimeDir = Join-Path $projectRoot ".jupyter_runtime"

if (-not (Test-Path $python)) {
    throw "Python not found: $python"
}

New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
$env:JUPYTER_RUNTIME_DIR = $runtimeDir

Write-Host "JUPYTER_RUNTIME_DIR=$env:JUPYTER_RUNTIME_DIR"
Write-Host "Starting JupyterLab from: $python"

& $python -m jupyterlab --no-browser --ServerApp.use_redirect_file=False @args
