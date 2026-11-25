@echo off
REM AIR Detection - one-time setup of the build tools (double-click).
REM Installs with winget (built into Windows 10 21H1+ / Windows 11):
REM   1. Python 3.13 64-bit (all users, py launcher, on PATH)
REM   2. Visual Studio 2022 Build Tools + "Desktop development with C++" (Nuitka)
REM   3. Inno Setup 6 (makes the Setup .exe)
REM and turns on Windows long paths (PyTorch has very long file names).
REM Anything already installed is skipped, so it is safe to run again.
REM Afterwards: double-click build_windows.bat.
setlocal EnableExtensions
cd /d "%~dp0"

REM --- run as administrator (one UAC prompt for everything) ---
net session >nul 2>&1
if errorlevel 1 (
    echo Asking for administrator rights ...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b 0
)

echo ============================================================
echo  AIR Detection - installing build tools
echo ============================================================
echo.

where winget >nul 2>&1
if errorlevel 1 (
    echo [X] winget was not found on this PC.
    echo     Install "App Installer" from the Microsoft Store, then run this again:
    echo     https://apps.microsoft.com/detail/9NBLGGH4NNS1
    start "" "ms-windows-store://pdp/?ProductId=9NBLGGH4NNS1"
    goto :fail
)
set WINGET=winget install --exact --accept-source-agreements --accept-package-agreements

REM --- free disk space (build needs ~15 GB, Build Tools ~6 GB more) ---
for /f %%G in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive '%~d0'.TrimEnd(':')).Free / 1GB)"') do set FREE_GB=%%G
echo Free space on %~d0 : %FREE_GB% GB
if defined FREE_GB if %FREE_GB% LSS 25 (
    echo [!] Less than 25 GB free - the install and the build may run out of space.
    echo.
)

REM --- 1. Python 3.13 64-bit ---
echo.
echo [1/4] Python 3.13 64-bit
call :has_python
if errorlevel 1 (
    %WINGET% --id Python.Python.3.13 --architecture x64 --override "/quiet InstallAllUsers=1 PrependPath=1 Include_launcher=1 InstallLauncherAllUsers=1 Include_test=0"
    call :has_python
    if errorlevel 1 (
        echo [X] Python 3.13 did not install. Install it by hand from
        echo     https://www.python.org/downloads/windows/  ^(64-bit, tick "Add python.exe to PATH"^)
        goto :fail
    )
)
echo     OK

REM --- 2. Visual Studio 2022 Build Tools with C++ ---
echo.
echo [2/4] Visual Studio 2022 Build Tools (C++) - large download, 10-30 minutes
set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
call :has_msvc
if errorlevel 1 (
    set VS_PATH=
    if exist "%VSWHERE%" for /f "usebackq delims=" %%P in (`"%VSWHERE%" -products Microsoft.VisualStudio.Product.BuildTools -latest -property installationPath`) do set "VS_PATH=%%P"
    call :install_vs
    call :has_msvc
    if errorlevel 1 (
        echo [X] The C++ build tools did not install. Open "Visual Studio Installer",
        echo     choose Build Tools 2022 - Modify - tick "Desktop development with C++".
        goto :fail
    )
)
echo     OK

REM --- 3. Inno Setup 6 ---
echo.
echo [3/4] Inno Setup 6
call :has_inno
if errorlevel 1 (
    %WINGET% --id JRSoftware.InnoSetup --silent
    call :has_inno
    if errorlevel 1 (
        echo [!] Inno Setup did not install. The build still works but makes no Setup .exe.
        echo     Install it from https://jrsoftware.org/isdl.php
    ) else (
        echo     OK
    )
) else (
    echo     OK
)

REM --- 4. long paths ---
echo.
echo [4/4] Windows long paths
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f >nul
if errorlevel 1 (echo [!] Could not turn on long paths - keep the kit near the drive root, e.g. C:\build) else (echo     OK)

echo.
echo ============================================================
echo  Done. Close this window, then double-click build_windows.bat
echo ============================================================
pause
exit /b 0

:fail
echo.
echo SETUP NOT FINISHED - see the messages above.
pause
exit /b 1

REM ---------------------------------------------------------------- helpers
:has_python
set PYEXE=py
if exist "%WINDIR%\py.exe" set "PYEXE=%WINDIR%\py.exe"
"%PYEXE%" -3.13 -c "import struct, sys; sys.exit(0 if struct.calcsize('P') == 8 else 1)" >nul 2>&1
exit /b %errorlevel%

:has_msvc
if not exist "%VSWHERE%" exit /b 1
set VC_FOUND=
for /f "usebackq delims=" %%P in (`"%VSWHERE%" -products * -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VC_FOUND=%%P"
if defined VC_FOUND exit /b 0
exit /b 1

:install_vs
set VS_ADD=--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended
if defined VS_PATH (
    echo     Build Tools found without C++ - adding it ...
    "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\setup.exe" modify --installPath "%VS_PATH%" %VS_ADD% --passive --norestart --wait
) else (
    %WINGET% --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --norestart %VS_ADD%"
)
exit /b 0

:has_inno
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" exit /b 0
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" exit /b 0
if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" exit /b 0
exit /b 1
