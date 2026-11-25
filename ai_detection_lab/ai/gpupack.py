"""v9.0.0: GPU Pack for the Windows app - lets Train use an NVIDIA card (cuda).

The Windows installer ships the small CPU-only PyTorch. The GPU Pack
(Hugging Face: rapeepat-jrs/ai-detection-lab-gpupack, AIDL_gpupack-cu126.zip,
2.4 GB) holds the CUDA 12.6 build of the same torch 2.14.1 / torchvision
0.29.1 files, laid out as `_internal/torch/...` to be copied over the app's
own `_internal` folder.

Why it used to say "no GPU" even after the pack was downloaded:
  1. The zip has to land exactly in `<app folder>\\_internal`. Windows'
     "Extract All" puts it in a NEW folder named after the zip
     (`...\\AIDL_gpupack-cu126\\_internal\\...`), and the app folder is in
     Program Files, which needs administrator rights - either way the app
     keeps loading its CPU files and torch.cuda.is_available() is False.
  2. PyInstaller had compiled torch/version.py into the app's archive, so
     a copied pack still left torch.version.cuda = None: the v6 app could
     not tell an installed pack from a missing one and always blamed "the
     CPU build". The v9 spec ships torch/version.py as a loose file
     (module_collection_mode) so the pack's copy is the one that is read
     (checked with PyInstaller 6.22.3).
  Which of the two hit the user's RTX 3060 Ti is not confirmed - diagnose()
  now says which.

This module installs the pack itself (download -> check -> copy into the
right folder while the app is closed, as administrator only if needed) and
explains in Thai why cuda is not available when it is not.

Nothing here imports torch at module level: entry_point.py runs the
`--apply-gpupack` step before torch may load (its DLLs would be locked).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = "rapeepat-jrs/ai-detection-lab-gpupack"
FILE = "AIDL_gpupack-cu126.zip"
URL = f"https://huggingface.co/{REPO}/resolve/main/{FILE}"
# As uploaded 2026-10-14. download() asks Hugging Face for the current
# values first, so a re-uploaded pack still verifies.
SIZE = 2574505904
SHA256 = "44a434ece6c6836a1f81f3a88c02d3c75d9a1602491f5c75ffe6e0721b7eba0f"
TORCH_VERSION = "2.14.1"        # the pack is built for exactly this torch (cp313)
CUDA_VERSION = "12.6"
MIN_DRIVER = (528, 33)          # oldest Windows NVIDIA driver for CUDA 12.6
MARKER = Path("torch") / "lib" / "torch_cuda.dll"
PREFIX = "_internal/"
ALLOWED = ("_internal/torch/", "_internal/torchvision/")
STAMP = "gpupack.json"
# Copying 2.4 GB needs room for the zip AND the copied files.
FREE_SPACE_NEEDED = 3 * 1024 ** 3


def supported() -> bool:
    """Only the Windows app can use the pack (Mac uses mps)."""
    return sys.platform == "win32"


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def internal_dir() -> Path | None:
    """The app's `_internal` folder (PyInstaller onedir), None from source."""
    if not frozen():
        return None
    return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent / "_internal"))


def installed(internal: Path | None) -> bool:
    return internal is not None and (internal / MARKER).is_file()


def misplaced_copies(app_dir: Path, internal: Path | None, depth: int = 5) -> list[Path]:
    """torch_cuda.dll anywhere under the app folder except the right place -
    the usual result of "Extract All" into the app folder."""
    right = (internal / MARKER).resolve() if internal else None
    found = []
    base_depth = len(app_dir.parts)
    for root, dirs, files in os.walk(app_dir):
        if len(Path(root).parts) - base_depth >= depth:
            dirs[:] = []
        if "torch_cuda.dll" in files:
            p = (Path(root) / "torch_cuda.dll").resolve()
            if p != right:
                found.append(p)
    return found


def downloads_copy() -> Path | None:
    """A pack zip the user already downloaded with the browser."""
    for folder in (Path.home() / "Downloads", Path.home() / "Desktop"):
        p = folder / FILE
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def nvidia_info() -> dict | None:
    """{'name', 'driver'} of the first NVIDIA card via nvidia-smi (installed
    with every NVIDIA driver), or None when there is no card / no driver."""
    exe = shutil.which("nvidia-smi")
    if exe is None and sys.platform == "win32":
        cand = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        exe = str(cand) if cand.is_file() else None
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, creationflags=_no_window())
    except Exception:
        return None
    line = (out.stdout or "").strip().splitlines()[:1]
    if out.returncode != 0 or not line:
        return None
    name, _, driver = line[0].partition(",")
    return {"name": name.strip(), "driver": driver.strip()}


def driver_ok(driver: str) -> bool:
    try:
        parts = tuple(int(x) for x in driver.split(".")[:2])
    except ValueError:
        return True     # unknown format: let torch decide
    return parts >= MIN_DRIVER


def probe_torch() -> dict:
    """What the PyTorch inside this app can do. Never raises."""
    info = {"version": None, "cuda_build": False, "cuda_available": False, "error": ""}
    try:
        import torch
    except Exception as exc:
        info["error"] = f"import torch: {type(exc).__name__}: {exc}"
        return info
    info["version"] = str(getattr(torch, "__version__", ""))
    lib = Path(torch.__file__).parent / "lib"
    # torch.version.cuda alone is not enough: v6 apps froze a CPU version.py.
    info["cuda_build"] = bool(getattr(torch.version, "cuda", None)) or (lib / "torch_cuda.dll").is_file()
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if not info["cuda_available"] and hasattr(torch._C, "_cuda_getDeviceCount"):
            torch.cuda.init()   # raises with the real reason (driver too old, ...)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:300] if str(exc).strip() else ''}"
    return info


def base_version(v) -> str:
    return str(v or "").split("+", 1)[0]


def diagnose(probe: dict | None = None, gpu: dict | None = ..., internal: Path | None = ...,
             app_dir: Path | None = None) -> dict:
    """{'ok': bool, 'reason': short code, 'lines': [Thai text], 'can_install': bool}"""
    if probe is None:
        probe = probe_torch()
    if gpu is ...:
        gpu = nvidia_info()
    if internal is ...:
        internal = internal_dir()
    if app_dir is None and internal is not None:
        app_dir = internal.parent
    lines = []
    if gpu:
        lines.append(f"การ์ดจอ: {gpu['name']} (ไดรเวอร์ {gpu['driver']})")
    if probe.get("version"):
        build = f"CUDA {CUDA_VERSION}" if probe.get("cuda_build") else "CPU"
        lines.append(f"PyTorch ในแอป: {probe['version']} รุ่น {build}")

    def result(ok, reason, msg, can_install=False):
        return {"ok": ok, "reason": reason, "lines": lines + ([msg] if msg else []), "can_install": can_install}

    if probe.get("cuda_available"):
        return result(True, "ok", "✅ ใช้การ์ดจอ NVIDIA (cuda) เทรนได้")
    if not supported():
        return result(False, "not_windows", "GPU Pack ใช้กับแอปบน Windows เท่านั้น (Mac ใช้ GPU แบบ mps)")
    if not gpu:
        return result(False, "no_nvidia",
                      "❌ ไม่พบการ์ดจอ NVIDIA หรือยังไม่ได้ลงไดรเวอร์ NVIDIA — "
                      "ลงไดรเวอร์จาก nvidia.com/drivers ก่อน (การ์ด Intel/AMD ใช้ cuda ไม่ได้)")
    if not driver_ok(gpu["driver"]):
        return result(False, "old_driver",
                      f"❌ ไดรเวอร์ NVIDIA เก่าเกินไป ({gpu['driver']}) — CUDA {CUDA_VERSION} ต้องใช้ "
                      f"{MIN_DRIVER[0]}.{MIN_DRIVER[1]} ขึ้นไป อัปเดตจาก nvidia.com/drivers")
    if not frozen():
        return result(False, "source",
                      "รันจากซอร์ส: ติดตั้ง PyTorch รุ่น CUDA เอง — pip install --force-reinstall "
                      f"torch=={TORCH_VERSION} torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126")
    if probe.get("version") and base_version(probe["version"]) != TORCH_VERSION:
        return result(False, "torch_mismatch",
                      f"❌ แอปนี้ใช้ PyTorch {probe['version']} แต่ GPU Pack ทำไว้สำหรับ {TORCH_VERSION} — "
                      "ใช้ด้วยกันไม่ได้ ต้อง build แอปใหม่ด้วย torch เวอร์ชันเดียวกัน")
    if not installed(internal):
        wrong = misplaced_copies(app_dir, internal) if app_dir and app_dir.is_dir() else []
        if wrong:
            return result(False, "misplaced",
                          "❌ เจอไฟล์ GPU Pack แต่แตกไว้ผิดที่: " + str(wrong[0].parent.parent.parent) +
                          f"\nต้องอยู่ใน {internal} — กด \"ติดตั้ง GPU Pack\" ให้แอปติดตั้งให้", True)
        return result(False, "not_installed",
                      "ยังไม่ได้ติดตั้ง GPU Pack (ไฟล์ CUDA 2.4 GB) — กด \"ติดตั้ง GPU Pack\"", True)
    if not probe.get("cuda_build"):
        return result(False, "stale_version",
                      "❌ ไฟล์ CUDA อยู่ครบ แต่แอปยังอ่านเวอร์ชันเป็น CPU (แอป v6 อ่าน version.py จากในแอปเอง) "
                      "— ใช้แอป v9.0.0 ขึ้นไป", True)
    return result(False, "cuda_error",
                  "❌ ติดตั้ง GPU Pack แล้วแต่ PyTorch ยังใช้การ์ดไม่ได้"
                  + (f": {probe['error']}" if probe.get("error") else "") +
                  "\nลองอัปเดตไดรเวอร์ NVIDIA แล้วรีสตาร์ทเครื่อง หรือกดติดตั้ง GPU Pack ใหม่", True)


# --- download --------------------------------------------------------------------

def remote_info(timeout: float = 15.0) -> tuple[int, str]:
    """(size, sha256) Hugging Face reports now; the built-in values if offline."""
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    try:
        opener = urllib.request.build_opener(_NoRedirect)
        req = urllib.request.Request(URL, method="HEAD")
        try:
            resp = opener.open(req, timeout=timeout)
            headers = resp.headers
        except urllib.error.HTTPError as exc:     # the 302 to the CDN carries the headers
            headers = exc.headers
        size = int(headers.get("x-linked-size") or 0) or SIZE
        sha = (headers.get("x-linked-etag") or "").strip('"') or SHA256
        return size, sha
    except Exception:
        return SIZE, SHA256


def download(dest: Path, progress=None, cancelled=None, url: str = URL,
             size: int | None = None, chunk: int = 1 << 20) -> Path:
    """Download (resuming a .part file) and return dest. progress(done, total)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    total = size or SIZE
    have = part.stat().st_size if part.exists() else 0
    if have > total:
        part.unlink()
        have = 0
    if have < total:
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        with urllib.request.urlopen(req, timeout=60) as resp:
            if have and resp.status != 206:      # server ignored Range: start over
                have = 0
            with open(part, "ab" if have else "wb") as f:
                while True:
                    if cancelled and cancelled():
                        raise InterruptedError("ยกเลิกการดาวน์โหลด")
                    block = resp.read(chunk)
                    if not block:
                        break
                    f.write(block)
                    have += len(block)
                    if progress:
                        progress(have, total)
    if part.stat().st_size != total:
        raise IOError(f"ดาวน์โหลดไม่ครบ ({part.stat().st_size:,} / {total:,} ไบต์) — กดดาวน์โหลดใหม่เพื่อโหลดต่อ")
    os.replace(part, dest)
    return dest


def sha256_of(path: Path, progress=None, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
            done += len(block)
            if progress:
                progress(done, total)
    return h.hexdigest()


# --- install -------------------------------------------------------------------

def check_zip(zip_path: Path) -> list[zipfile.ZipInfo]:
    """The files to copy. Raises ValueError if this is not a GPU Pack."""
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"ไฟล์ zip เสีย หรือดาวน์โหลดไม่ครบ ({exc})") from exc
    with zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        names = [m.filename.replace("\\", "/") for m in members]
        if not names or any(not n.startswith(ALLOWED) or ".." in n.split("/") for n in names):
            raise ValueError("ไม่ใช่ไฟล์ GPU Pack ของ AI Detection Lab (ต้องมีแค่ _internal/torch และ _internal/torchvision)")
        if PREFIX + MARKER.as_posix() not in names:
            raise ValueError("ไฟล์ GPU Pack ไม่ครบ (ไม่มี torch_cuda.dll)")
        if PREFIX + "torch/version.py" in names:
            text = zf.read(PREFIX + "torch/version.py").decode("utf-8", "replace")
            if f"'{TORCH_VERSION}+" not in text and f'"{TORCH_VERSION}+' not in text:
                raise ValueError(f"GPU Pack นี้ไม่ใช่ PyTorch {TORCH_VERSION}")
        return members


def writable(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / f".aidl_write_test_{os.getpid()}"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def install_zip(zip_path: Path, internal: Path, progress=None, retry_sec: float = 60.0) -> int:
    """Copy the pack into `internal` (each file via a temp name + os.replace).
    Retries locked files for up to retry_sec (the app may still be closing).
    Returns the number of files copied."""
    members = check_zip(zip_path)
    total = sum(m.file_size for m in members)
    done = 0
    with zipfile.ZipFile(zip_path) as zf:
        for m in members:
            rel = m.filename.replace("\\", "/")[len(PREFIX):]
            target = internal / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".gpupack-new")
            with zf.open(m) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            deadline = time.monotonic() + retry_sec
            while True:
                try:
                    os.replace(tmp, target)
                    break
                except PermissionError:
                    if time.monotonic() > deadline:
                        tmp.unlink(missing_ok=True)
                        raise
                    time.sleep(0.5)
            done += m.file_size
            if progress:
                progress(done, total)
    (internal / STAMP).write_text(json.dumps({
        "pack": FILE, "torch": TORCH_VERSION, "cuda": CUDA_VERSION, "files": len(members),
        "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")
    return len(members)


def result_path() -> Path:
    from .. import config
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home())
    return base / f"AI Detection Lab v{config.APP_VERSION}" / "gpupack_result.json"


def launch_apply(zip_path: Path) -> bool:
    """Start `<app>.exe --apply-gpupack <zip>` (as administrator if the app
    folder needs it). The caller then quits the app so its DLLs are free."""
    internal = internal_dir()
    if not supported() or internal is None:
        return False
    args = ["--apply-gpupack", str(zip_path), "--wait-pid", str(os.getpid())]
    if writable(internal):
        subprocess.Popen([sys.executable, *args], close_fds=True)
        return True
    import ctypes
    params = subprocess.list2cmdline(args)
    # > 32 = started; the user may answer "No" to the UAC prompt (5 = denied)
    return int(ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, params, None, 1)) > 32


def _wait_for_pid(pid: int, timeout: float = 30.0) -> None:
    if sys.platform != "win32" or pid <= 0:
        return
    import ctypes
    SYNCHRONIZE = 0x00100000
    handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if handle:
        ctypes.windll.kernel32.WaitForSingleObject(handle, int(timeout * 1000))
        ctypes.windll.kernel32.CloseHandle(handle)


def apply_main(argv: list[str]) -> int:
    """entry_point.py: `--apply-gpupack <zip> [--wait-pid N]`. Shows a small
    progress window, copies the pack, then reopens the app (or says why not)."""
    zip_path = Path(argv[argv.index("--apply-gpupack") + 1])
    pid = int(argv[argv.index("--wait-pid") + 1]) if "--wait-pid" in argv else 0
    from PySide6 import QtCore, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    dlg = QtWidgets.QProgressDialog("กำลังรอให้แอปปิด…", None, 0, 1000)
    dlg.setWindowTitle("ติดตั้ง GPU Pack")
    dlg.setMinimumWidth(460)
    dlg.setMinimumDuration(0)
    dlg.show()
    app.processEvents()
    _wait_for_pid(pid)
    internal = internal_dir()
    outcome = {"ok": False, "zip": str(zip_path), "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        if internal is None:
            raise RuntimeError("ต้องรันจากแอปที่ติดตั้งแล้ว")
        dlg.setLabelText(f"กำลังคัดลอกไฟล์ CUDA ไปที่\n{internal}")

        def step(done, total):
            dlg.setValue(int(done * 1000 / max(1, total)))
            app.processEvents()

        outcome["files"] = install_zip(zip_path, internal, step)
        outcome["ok"] = True
    except Exception as exc:
        outcome["error"] = f"{type(exc).__name__}: {exc}"
    dlg.close()
    try:
        result_path().parent.mkdir(parents=True, exist_ok=True)
        result_path().write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    if outcome["ok"]:
        QtWidgets.QMessageBox.information(None, "ติดตั้ง GPU Pack",
                                          f"ติดตั้งเสร็จแล้ว ({outcome['files']} ไฟล์)\nกำลังเปิดแอปใหม่…")
        _relaunch()
        return 0
    QtWidgets.QMessageBox.critical(None, "ติดตั้ง GPU Pack ไม่สำเร็จ",
                                   outcome["error"] + "\n\nปิดแอปทุกหน้าต่างแล้วลองใหม่ หรือรันแอปแบบ Run as administrator")
    return 1


def _relaunch() -> None:
    """Open the app normally - not as administrator, even if this step was."""
    try:
        if sys.platform == "win32":
            # explorer.exe starts it with the signed-in user's normal rights
            subprocess.Popen(["explorer.exe", sys.executable], close_fds=True)
        else:
            subprocess.Popen([sys.executable], close_fds=True)
    except OSError:
        pass


def last_result() -> dict | None:
    """The outcome of the last --apply-gpupack run, read once then removed."""
    p = result_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        p.unlink()
        return data
    except (OSError, ValueError):
        return None


def disk_free(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0
