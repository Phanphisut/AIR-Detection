@echo off
REM AIR Detection - build the Windows installer (double-click or run in cmd).
REM Result: dist_installer\AIR Detection Setup v<VERSION>.exe  (Inno Setup installer)
REM
REM Needs (setup_windows.bat installs them): 64-bit Python 3.13, Visual Studio 2022
REM Build Tools ("Desktop development with C++") for Nuitka, Inno Setup 6.
REM Python environment: .venv here if present, otherwise the one setup.bat made
REM (%USERPROFILE%\.venvs\air-detection) - created here if missing.
REM
REM PyTorch: the default is the small CPU-only build (torch 2.14.1). Users with
REM an NVIDIA card add CUDA from inside the app (GPU Pack).
REM For one big installer with CUDA built in: set TORCH_CUDA=cu126 first
REM (cu126 = CUDA 12.6: NVIDIA driver 528+, GTX 900 series and newer).
setlocal EnableExtensions
cd /d "%~dp0"
if "%TORCH_CUDA%"=="" set TORCH_CUDA=cpu
set TORCH_INDEX=https://download.pytorch.org/whl/%TORCH_CUDA%

set VPY=
if exist ".venv\Scripts\python.exe" set "VPY=%~dp0.venv\Scripts\python.exe"
if not defined VPY set "VPY=%USERPROFILE%\.venvs\air-detection\Scripts\python.exe"

if not exist "%VPY%" (
    echo Python environment not found - creating it with setup.bat ...
    call "%~dp0setup.bat" < nul || goto :error
)
if not exist "%VPY%" goto :error

REM build tools (PyInstaller, Nuitka) on top of the run environment
"%VPY%" -m pip install --quiet -r requirements-windows.txt || goto :error

REM TORCH_CUDA=cu126 with a CPU-only environment: swap in the CUDA build.
if not "%TORCH_CUDA%"=="cpu" (
    "%VPY%" -c "import sys, torch; sys.exit(0 if torch.version.cuda else 1)"
    if errorlevel 1 (
        echo Replacing CPU-only PyTorch with the %TORCH_CUDA% build ...
        "%VPY%" -m pip install --force-reinstall --no-deps torch==2.14.1 torchvision==0.29.1 --index-url %TORCH_INDEX% || goto :error
    )
)

"%VPY%" build_windows.py %* || goto :error
echo.
echo Build finished. Installer: "dist_installer\"  (app folder: %LOCALAPPDATA%\AIR Detection build\dist\)
pause
exit /b 0

:error
echo.
echo BUILD FAILED - see the messages above.
pause
exit /b 1
