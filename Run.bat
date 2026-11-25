@echo off
REM AIR Detection - run from source (no build needed).
REM Uses the Python environment of the AI Detection Lab v9 build kit unless
REM AIR_DETECTION_PYEXE points at another python.exe (or .venv\Scripts\python.exe exists here).
setlocal
cd /d "%~dp0"
if "%AIR_DETECTION_PYEXE%"=="" if exist ".venv\Scripts\python.exe" set AIR_DETECTION_PYEXE=%~dp0.venv\Scripts\python.exe
if "%AIR_DETECTION_PYEXE%"=="" if exist "%USERPROFILE%\.venvs\air-detection\Scripts\python.exe" set AIR_DETECTION_PYEXE=%USERPROFILE%\.venvs\air-detection\Scripts\python.exe
if "%AIR_DETECTION_PYEXE%"=="" set AIR_DETECTION_PYEXE=D:\AI_Detection_Lab_v9.0.0_Windows_Build_Kit\.venv\Scripts\python.exe
if not exist "%AIR_DETECTION_PYEXE%" (
    echo python.exe not found: %AIR_DETECTION_PYEXE%
    echo Run build_windows.bat once to create .venv, or set AIR_DETECTION_PYEXE.
    pause
    exit /b 1
)
"%AIR_DETECTION_PYEXE%" entry_point.py
if errorlevel 1 pause
