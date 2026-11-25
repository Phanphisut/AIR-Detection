"""Build AIR Detection 1.1.0 for Windows (run ON Windows, 64-bit Python 3.13).

    python build_windows.py            # Nuitka -> PyInstaller -> checker -> installer
    python build_windows.py --no-installer

Steps:
  1. Nuitka compiles ai_detection_lab/ into ONE native module
     (build/nuitka/ai_detection_lab.cp313-win_amd64.pyd) - no .py/.pyc of the
     app ships (same as the macOS v6.0.0 build). Needs a C compiler: Visual
     Studio 2022 Build Tools with "Desktop development with C++".
  2. PyInstaller builds dist/AIR Detection/ from AIR_Detection_Windows.spec.
  3. A one-file "Check Compatibility.exe" for the installer's pre-install check.
  4. Inno Setup (ISCC.exe) packs dist_installer/AIR Detection Setup v1.1.0.exe.
  5. Checks the result: no ai_detection_lab .py/.pyc in the build.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = "AIR Detection"
PY = sys.executable
# Everything PyInstaller/Nuitka write (thousands of files) goes OUTSIDE the project:
# the project lives in OneDrive, whose sync client locks files so old builds cannot be
# deleted ("Access is denied"). Only the finished installer lands in dist_installer/.
BUILD = Path(os.environ.get("LOCALAPPDATA", str(ROOT))) / "AIR Detection build"
DIST = BUILD / "dist"


def run(cmd, **kw):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True, **kw)


def source_digest() -> str:
    digest = hashlib.sha256()
    for p in sorted((ROOT / "ai_detection_lab").rglob("*.py")):
        if "__pycache__" not in p.parts:
            digest.update(str(p.relative_to(ROOT)).encode() + b"\0" + p.read_bytes())
    return digest.hexdigest()


def compile_with_nuitka():
    out = ROOT / "build" / "nuitka"
    stamp = out / "sources.sha256"
    digest = source_digest()
    built = sorted(out.glob("ai_detection_lab.*.pyd"))
    if built and stamp.is_file() and stamp.read_text().strip() == digest:
        print(f"Nuitka: {built[0].name} matches the source")
        return
    for old in built:
        old.unlink()
    out.mkdir(parents=True, exist_ok=True)
    # Nuitka's temporary C build folder holds thousands of files: inside OneDrive the
    # sync client locks them and Nuitka cannot delete the folder afterwards. Build in
    # %LOCALAPPDATA% instead and copy only the finished module back.
    work = BUILD / "nuitka"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    run([PY, "-m", "nuitka", "--module", "ai_detection_lab",
         "--include-package=ai_detection_lab", f"--output-dir={work}",
         "--python-flag=no_docstrings", "--remove-output",
         "--assume-yes-for-downloads", "--msvc=latest"])
    for f in list(work.glob("ai_detection_lab.*.pyd")) + list(work.glob("ai_detection_lab.pyi")):
        shutil.copy2(f, out / f.name)
    stamp.write_text(digest + "\n")


def build_app():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "vendor"), PYTHONDONTWRITEBYTECODE="1",
               AIR_BUILD_DIR=str(BUILD))
    run([PY, "-m", "PyInstaller", "--noconfirm",
         "--distpath", DIST, "--workpath", BUILD / "work",
         ROOT / "AIR_Detection_Windows.spec"], env=env)


def build_checker():
    run([PY, "-m", "PyInstaller", "--noconfirm", "--onefile", "--windowed",
         "--name", "Check Compatibility", "--icon", ROOT / "build_assets" / "app_icon.ico",
         "--distpath", BUILD / "checker", "--workpath", BUILD / "checker-work",
         "--specpath", BUILD, ROOT / "build_assets" / "check_compatibility.py"])


def build_installer():
    iscc = shutil.which("ISCC") or next((str(p) for p in (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",   # per-user install
    ) if p.is_file()), None)
    if iscc is None:
        print("\n⚠️ Inno Setup 6 not found - skipping the installer. Install it from "
              "https://jrsoftware.org/isdl.php and run: ISCC installer.iss")
        return
    iss_paths = [f"/DDistDir={DIST}", f"/DCheckerDir={BUILD / 'checker'}"]
    try:
        run([iscc, *iss_paths, ROOT / "installer.iss"])
    except subprocess.CalledProcessError:
        # The CUDA build of PyTorch makes the app large; one setup .exe tops
        # out at ~2.1 GB, so Inno Setup needs disk spanning (setup .exe + .bin).
        print("\nInstaller too large for one file - retrying with disk spanning (.exe + .bin)")
        run([iscc, "/DBigSetup", *iss_paths, ROOT / "installer.iss"])
        print("⚠️ Hand out EVERYTHING in dist_installer\\ together (Setup .exe + its .bin files).")


def verify():
    app = DIST / APP
    leaked = [p for p in app.rglob("*") if p.suffix in (".py", ".pyc") and "ai_detection_lab" in p.parts]
    compiled = list(app.rglob("ai_detection_lab.*.pyd"))
    params = list(app.rglob("app_params/config.py"))
    print("\n=== check ===")
    print(f"no ai_detection_lab .py/.pyc in the build: {'OK' if not leaked else leaked[:5]}")
    print(f"Nuitka module present: {'OK' if len(compiled) == 1 else compiled}")
    print(f"config.py for Quick Parameters: {'OK' if params else 'MISSING'}")
    cuda_dll = list(app.rglob("torch_cuda.dll"))
    print(f"PyTorch CUDA in the installer: {'yes (big installer)' if cuda_dll else 'no - users add it with ⚡ GPU Pack in the Train page'}")
    # v9.0.0: the GPU Pack overwrites these loose files; frozen into the PYZ
    # they would keep reporting the CPU build (torch.version.cuda = None).
    loose_version = [app / "_internal" / "torch" / "version.py", app / "_internal" / "torchvision" / "version.py"]
    missing = [str(p) for p in loose_version if not p.is_file()]
    print(f"torch/torchvision version.py as loose files (GPU Pack): {'OK' if not missing else 'MISSING ' + ', '.join(missing)}")
    if leaked or len(compiled) != 1 or not params or missing:
        raise SystemExit("build check failed")


def main():
    if sys.platform != "win32":
        raise SystemExit("Run this on Windows. (Windows only.)")
    if sys.version_info[:2] != (3, 13) or sys.maxsize <= 2 ** 32:
        raise SystemExit(f"Needs 64-bit Python 3.13 (this is {sys.version.split()[0]}, "
                         f"{'64' if sys.maxsize > 2 ** 32 else '32'}-bit)")
    import torch
    import torchvision
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda or 'none - CPU-only build'})")
    # The Hugging Face GPU Pack holds torch 2.14.1+cu126 / torchvision 0.29.1+cu126
    # files: it only fits an app built with exactly these versions.
    if torch.__version__.split("+")[0] != "2.14.1" or torchvision.__version__.split("+")[0] != "0.29.1":
        raise SystemExit(f"Needs torch 2.14.1 + torchvision 0.29.1 (found {torch.__version__} / "
                         f"{torchvision.__version__}) - the GPU Pack is built for exactly those.")
    if torch.version.cuda is None:
        print("ℹ️ CPU-only PyTorch: small installer. NVIDIA users install the GPU Pack from\n"
              "   the app (Train page -> ⚡ GPU Pack). For a full CUDA installer instead:\n"
              "   set TORCH_CUDA=cu126 and run build_windows.bat again.")
    compile_with_nuitka()
    build_app()
    verify()
    build_checker()
    if "--no-installer" not in sys.argv:
        build_installer()
    print(f"\n✅ Done: {DIST / APP / (APP + '.exe')}  ->  installer in dist_installer\\")


if __name__ == "__main__":
    main()
