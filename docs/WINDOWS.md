# Save and run under C:\Projects

This repository is intended to live at `C:\Projects\aircraft-doppler-geolocation`.
The development session runs on Linux and cannot directly write to a user's Windows drive.

## Download archive

Download the supplied `aircraft-doppler-geolocation.zip` and run in PowerShell:

```powershell
New-Item -ItemType Directory -Force C:\Projects | Out-Null
Expand-Archive "$env:USERPROFILE\Downloads\aircraft-doppler-geolocation.zip" C:\Projects
Set-Location C:\Projects\aircraft-doppler-geolocation
powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1
```

The archive contains the project directory, not a machine-specific virtual environment.
Expand it into a new destination; do not overwrite an existing working checkout. The setup
script creates a local virtual environment, installs dependencies, runs tests and executes
the synthetic experiment. Install Python 3.11+ with the `py` launcher first. The process
execution-policy flag applies only to that PowerShell process, not your system policy.

Alternatively, with Git and access to this private repository:

```powershell
New-Item -ItemType Directory -Force C:\Projects | Out-Null
git clone https://github.com/advaithsarva/aircraft-doppler-geolocation.git C:\Projects\aircraft-doppler-geolocation
Set-Location C:\Projects\aircraft-doppler-geolocation
powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1
```

Launch the dashboard:

```powershell
.\.venv\Scripts\streamlit.exe run app\research.py
```

No activation script is needed. FFmpeg is an optional separate executable for video files;
WAV/FLAC and synthetic experiments work without it. Install FFmpeg and put it on PATH to
use video input. A missing executable or extraction failure now produces a clear error;
overlong recordings are rejected consistently, rather than silently truncated for video.

The Python pipeline uses platform-aware peak-memory reporting (Windows working-set peak,
Linux ru_maxrss, macOS byte conversion). Windows CI is configured separately from Linux.
Check the actual workflow result before assuming a platform was successfully validated.
