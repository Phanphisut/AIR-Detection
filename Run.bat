@echo off
REM AIR Detection - run from source (no build needed).
REM First time on this PC: double-click setup.bat.
REM Uses .venv here if present, else %USERPROFILE%\.venvs\air-detection (made by setup.bat),
REM or AIR_DETECTION_PYEXE if you set it to another python.exe.
setlocal
cd /d "%~dp0"
if "%AIR_DETECTION_PYEXE%"=="" if exist ".venv\Scripts\python.exe" set AIR_DETECTION_PYEXE=%~dp0.venv\Scripts\python.exe
if "%AIR_DETECTION_PYEXE%"=="" if exist "%USERPROFILE%\.venvs\air-detection\Scripts\python.exe" set AIR_DETECTION_PYEXE=%USERPROFILE%\.venvs\air-detection\Scripts\python.exe
if "%AIR_DETECTION_PYEXE%"=="" (
    echo Python environment not found - running setup.bat first ...
    call "%~dp0setup.bat" || exit /b 1
    set AIR_DETECTION_PYEXE=%USERPROFILE%\.venvs\air-detection\Scripts\python.exe
)
"%AIR_DETECTION_PYEXE%" entry_point.py
if errorlevel 1 pause
