@echo off
REM AIR Detection - Windows build (double-click or run in cmd).
REM Needs: 64-bit Python 3.13 (py launcher), Visual Studio 2022 Build Tools
REM ("Desktop development with C++") for Nuitka, Inno Setup 6 for the installer.
REM setup_windows.bat installs all three (run it once first).
REM
REM PyTorch: the default is the small CPU-only build (torch 2.14.1). Users with
REM an NVIDIA card add CUDA from inside the app: Train page -> "GPU Pack", which
REM downloads AIDL_gpupack-cu126.zip from Hugging Face and installs it.
REM For one big installer with CUDA built in: set TORCH_CUDA=cu126 first
REM (cu126 = CUDA 12.6: NVIDIA driver 528+, GTX 900 series and newer).
setlocal
cd /d "%~dp0"
if "%TORCH_CUDA%"=="" set TORCH_CUDA=cpu
set TORCH_INDEX=https://download.pytorch.org/whl/%TORCH_CUDA%

py -3.13 -c "pass" >nul 2>&1
if errorlevel 1 (
    echo Python 3.13 64-bit was not found. Double-click setup_windows.bat first,
    echo then close this window and run build_windows.bat again.
    goto :error
)

if not exist .venv\Scripts\python.exe (
    echo Creating build environment .venv ...
    py -3.13 -m venv .venv || goto :error
    .venv\Scripts\python.exe -m pip install --upgrade pip || goto :error
    .venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url %TORCH_INDEX% || goto :error
    .venv\Scripts\python.exe -m pip install -r requirements-windows.txt || goto :error
)

REM TORCH_CUDA=cu126 with an existing CPU-only .venv: swap in the CUDA build.
if not "%TORCH_CUDA%"=="cpu" (
    .venv\Scripts\python.exe -c "import sys, torch; sys.exit(0 if torch.version.cuda else 1)"
    if errorlevel 1 (
        echo Replacing CPU-only PyTorch with the %TORCH_CUDA% build ...
        .venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.14.1 torchvision==0.29.1 --index-url %TORCH_INDEX% || goto :error
    )
)

.venv\Scripts\python.exe build_windows.py %* || goto :error
echo.
echo Build finished. App: "dist\AIR Detection\"  Installer: "dist_installer\"
pause
exit /b 0

:error
echo.
echo BUILD FAILED - see the messages above.
pause
exit /b 1
