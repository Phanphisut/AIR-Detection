"""Standalone Windows compatibility checker for AIR Detection.

Compiled by PyInstaller into its own small "Check Compatibility.exe" -
deliberately kept to stdlib + ctypes only (no torch/cv2/PySide6) so it
builds fast and can run even before the main app's heavier dependencies
are known to work on this machine.

Two ways this runs:
  1. Double-clicked from the Start Menu shortcut the installer creates -
     shows a native MessageBox with the full pass/fail report.
  2. Called by the Inno Setup installer itself (installer.iss,
     InitializeSetup) with --silent, right before it copies any files -
     still shows the same MessageBox (that IS the "alert saying why" the
     installer needs), but the installer additionally reads this
     process's exit code (0 = all hard checks passed, 1 = at least one
     hard check failed) to decide whether to ask the user to continue
     anyway.

Checks are split into hard failures (the app will very likely fail to
even start - wrong OS bitness, Windows too old, no VC++ runtime DLLs)
and warnings (the app will start, but a specific feature may not work -
no webcam detected). Only hard failures flip the exit code to 1.
"""
import ctypes
import platform
import shutil
import subprocess
import sys

MB_ICONERROR = 0x10
MB_ICONWARNING = 0x30
MB_ICONINFORMATION = 0x40

MIN_RAM_GB = 4
MIN_FREE_DISK_GB = 3
# Prebuilt PyTorch/OpenCV wheels bundled in this build are 64-bit only.
REQUIRED_ARCH = ("amd64", "x86_64")


def _is_64bit_windows() -> bool:
    return platform.machine().lower() in REQUIRED_ARCH


def _windows_version_ok() -> bool:
    try:
        return sys.getwindowsversion().major >= 10
    except AttributeError:
        # Not running on Windows at all.
        return False


def _ram_gb() -> float:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullTotalPhys / (1024 ** 3)


def _free_disk_gb(drive: str = "C:\\") -> float:
    return shutil.disk_usage(drive).free / (1024 ** 3)


def _vcredist_present() -> bool:
    """The frozen build's bundled PyTorch/OpenCV DLLs link against the
    Visual C++ 2015-2022 runtime. Without it, the app doesn't show a
    Python traceback at all - Windows just refuses to launch the .exe
    with a generic "This application was unable to start correctly"
    dialog, which is very hard for a student to diagnose on their own."""
    for dll in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
        try:
            ctypes.WinDLL(dll)
        except OSError:
            return False
    return True


def _webcam_present() -> bool:
    """Informational only (never a hard failure): most of the app (Label
    & Train AI on existing video, Code Editor, post-flight reports) works
    with no camera at all. Only the live camera / Gesture features need
    one."""
    try:
        proc = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                # @(...) forces array context - without it, PowerShell
                # unwraps a single matching device into a bare object,
                # which has no .Count property at all (blank output, not
                # "1"). Confirmed live: a real machine with exactly one
                # "USB Camera" (Class=Camera, Status=OK) printed nothing
                # here, which the strip()-into-("", "0") check below then
                # read as "no webcam" and warned the student their camera
                # was missing even though it was plugged in and working -
                # the single-device case is the COMMON case (most
                # machines have exactly one webcam), not an edge case.
                "@(Get-PnpDevice -Class Camera,Image -Status OK "
                "-ErrorAction SilentlyContinue).Count",
            ],
            capture_output=True, text=True, timeout=10,
        )
        return proc.stdout.strip() not in ("", "0")
    except Exception:
        # Can't tell either way - don't punish the user for a check that
        # itself failed to run.
        return True


def run_checks():
    failures = []
    warnings = []

    if not _windows_version_ok():
        failures.append(
            "Needs Windows 10 or newer (found an older version, or not Windows)"
        )
    if not _is_64bit_windows():
        failures.append(
            f"This is a 32-bit system ({platform.machine()}), but the program needs 64-bit Windows "
            "(the bundled PyTorch/OpenCV are 64-bit only)"
        )
    if not _vcredist_present():
        failures.append(
            "Microsoft Visual C++ Redistributable 2015-2022 (x64) is missing — "
            "the program will not start. Install it first from "
            "https://aka.ms/vs/17/release/vc_redist.x64.exe"
        )

    ram = _ram_gb()
    if ram < MIN_RAM_GB:
        failures.append(
            f"Only {ram:.1f} GB of RAM (needs at least {MIN_RAM_GB} GB — "
            "PyTorch/YOLO use a lot of memory while processing pictures)"
        )

    free_disk = _free_disk_gb()
    if free_disk < MIN_FREE_DISK_GB:
        failures.append(
            f"Only {free_disk:.1f} GB free on drive C: (needs at least "
            f"{MIN_FREE_DISK_GB} GB to install)"
        )

    if not _webcam_present():
        warnings.append(
            "No working webcam found — the program still opens, but features that need a camera "
            "(like Webcam only) won't work until one is connected"
        )

    return failures, warnings


def main():
    # --silent is used by the Inno Setup installer (InitializeSetup) to
    # get a pass/fail exit code WITHOUT a blocking MessageBox - the
    # installer decides itself whether/how to alert the user based on
    # the exit code, so this process must never sit waiting for a click
    # during an otherwise-unattended install step.
    silent = "--silent" in sys.argv
    failures, warnings = run_checks()

    if failures:
        if not silent:
            lines = ["This computer may not be able to run AIR Detection because:\n"]
            lines += [f"❌ {f}" for f in failures]
            if warnings:
                lines.append("")
                lines += [f"⚠️ {w}" for w in warnings]
            ctypes.windll.user32.MessageBoxW(
                0, "\n".join(lines), "AIR Detection — compatibility check (failed)",
                MB_ICONERROR,
            )
        return 1

    if not silent:
        lines = ["This computer is ready to run AIR Detection ✅"]
        if warnings:
            lines.append("")
            lines += [f"⚠️ {w}" for w in warnings]
        icon = MB_ICONWARNING if warnings else MB_ICONINFORMATION
        ctypes.windll.user32.MessageBoxW(
            0, "\n".join(lines), "AIR Detection — compatibility check", icon,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
