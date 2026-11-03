@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" main.py --select-video
) else if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" main.py --select-video
) else if exist "..\CBS_VISION_PROJECT\cbs-venv\Scripts\python.exe" (
  "..\CBS_VISION_PROJECT\cbs-venv\Scripts\python.exe" main.py --select-video
) else (
  python main.py --select-video
)
if errorlevel 1 pause
