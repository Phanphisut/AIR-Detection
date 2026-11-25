@echo off
REM ============================================================
REM  AIR Detection - one-click setup on a new Windows PC (run from source)
REM   1. Python 3.13 64-bit (winget, current user - no admin needed)
REM   2. venv at %USERPROFILE%\.venvs\air-detection  (outside OneDrive / the repo)
REM   3. PyTorch: CUDA 12.6 build if an NVIDIA GPU is found, otherwise CPU
REM   4. the rest of requirements-run.txt
REM  Safe to run again (repairs / updates). Afterwards: Run.bat or F5 in VS Code.
REM  Force CPU torch:  set TORCH_CUDA=cpu  before running.
REM ============================================================
setlocal EnableExtensions
cd /d "%~dp0"
set VENV=%USERPROFILE%\.venvs\air-detection
set VPY=%VENV%\Scripts\python.exe

echo [1/4] Python 3.13
call :find_python
if not defined BASEPY (
    where winget >nul 2>&1 || (
        echo [X] winget not found. Install Python 3.13 64-bit from https://www.python.org/downloads/windows/
        echo     then run setup.bat again.
        goto :fail
    )
    winget install --exact --id Python.Python.3.13 --scope user --silent --accept-source-agreements --accept-package-agreements
    call :find_python
)
if not defined BASEPY (
    echo [X] Python 3.13 did not install. Install it by hand from https://www.python.org/downloads/windows/
    goto :fail
)
echo     %BASEPY%

echo.
echo [2/4] Virtual environment: %VENV%
if not exist "%VPY%" (
    "%BASEPY%" -m venv "%VENV%" || goto :fail
)
"%VPY%" -m pip install --upgrade pip --quiet || goto :fail

echo.
echo [3/4] PyTorch (large download, several minutes)
if not defined TORCH_CUDA (
    set TORCH_CUDA=cpu
    where nvidia-smi >nul 2>&1 && set TORCH_CUDA=cu126
)
echo     build: %TORCH_CUDA%
"%VPY%" -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/%TORCH_CUDA% || goto :fail

echo.
echo [4/4] Other packages
"%VPY%" -m pip install -r requirements-run.txt || goto :fail

"%VPY%" -c "import PySide6, torch, ultralytics, cv2; print('    torch', torch.__version__, '| GPU:', torch.cuda.is_available())" || goto :fail

echo.
echo ============================================================
echo  Done. Start the app: double-click Run.bat
echo  (or open this folder in VS Code and press F5)
echo ============================================================
pause
exit /b 0

:fail
echo.
echo SETUP NOT FINISHED - see the messages above.
pause
exit /b 1

:find_python
set BASEPY=
for %%P in ("%LOCALAPPDATA%\Programs\Python\Python313\python.exe" "%ProgramFiles%\Python313\python.exe") do (
    if not defined BASEPY if exist %%P set "BASEPY=%%~P"
)
if not defined BASEPY (
    for /f "delims=" %%P in ('py -3.13 -c "import sys; print(sys.executable)" 2^>nul') do set "BASEPY=%%P"
)
exit /b 0
