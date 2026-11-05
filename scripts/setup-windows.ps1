# Run from the extracted project: powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1
# Changes only this project's virtual environment. Does not change global execution policy.
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    if (-not (Test-Path '.venv\Scripts\python.exe')) {
        & py -3 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Creating the virtual environment failed. Install Python 3.11+.' }
    }
    & .\.venv\Scripts\python.exe -m pip install -e '.[dev,app]'
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & .\.venv\Scripts\python.exe -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed; review the output above.' }
    & .\.venv\Scripts\doppler-geo.exe demo --output outputs\demo
    if ($LASTEXITCODE -ne 0) { throw 'The synthetic experiment failed.' }
    Write-Host 'Setup and validation finished.'
    Write-Host 'Run: .\.venv\Scripts\streamlit.exe run app\research.py'
}
finally {
    Pop-Location
}
