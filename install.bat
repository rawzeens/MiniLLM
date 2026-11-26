@echo off
rem MiniLLM installer for Windows: finds Python and runs install.py,
rem which detects your hardware and picks the right PyTorch.
rem   install.bat            (add --dry-run, --cpu, --tier small, --yes ...)
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 install.py %*
  goto :end
)
where python >nul 2>nul
if %errorlevel%==0 (
  python install.py %*
  goto :end
)
echo Python wasn't found. Install Python 3.12 from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during setup, then run install.bat again.
:end
pause
