# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for the Windows build of AIR Detection.

Must be built ON Windows (PyInstaller does not cross-compile). Use
build_windows.bat (sets up the environment) or build_windows.py, which
compiles ai_detection_lab with Nuitka first - this spec refuses to run
without that compiled module (v6.0.0 ships no .py/.pyc of the app).

Produces dist/AI Detection Lab/ - a self-contained --onedir Windows build
(PySide6 GUI + bundled Python/OpenCV/PyTorch/Ultralytics runtime), with
"AI Detection Lab.exe" as the entry point next to an _internal/ folder of
dependencies - same shape as this project's existing (pre-fix) Windows
build at the project root (AI Detection Lab.exe + _internal/).

This mirrors AI_Detection_Lab.spec (the macOS build), minus the
macOS-only BUNDLE()/codesign_identity/icns steps, and using a .ico icon
instead of .icns.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata
import sys

# pyhulax 0.2.0 ships in vendor/ (the copy tested with the real drone);
# collect_all("pyhulax") below must be able to import it.
sys.path.insert(0, str(Path(SPECPATH) / "vendor"))

block_cipher = None

datas = []
binaries = []
hiddenimports = []

# Export is PyTorch .pt only: no onnx in the app.
for pkg in ("ultralytics", "torch", "torchvision", "cv2", "pyhulax"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# ultralytics reads its own package metadata (version, etc.) at import time.
datas += copy_metadata("ultralytics")

# torch's own .dist-info ships a vendored copy of every third-party
# dependency's license text (kineto, tensorpipe, protobuf, flash-attention,
# ...), each nested many folders deep (e.g.
# torch-2.13.0.dist-info/licenses/third_party/kineto/libkineto/third_party/
# dynolog/third_party/prometheus-cpp/3rdparty/civetweb/src/third_party/
# duktape-1.5.2). None of it is read at runtime - it's just legal text -
# but once COLLECT recreates that depth under
# dist/AI Detection Lab/_internal/..., the combined path blows past
# Windows' 260-char MAX_PATH and PyInstaller's COLLECT step crashes with
# "FileNotFoundError: [WinError 206] The filename or extension is too
# long" (confirmed by a real build here). collect_all() hands back one
# directory-level (src, dest) entry per *.dist-info folder (copied
# wholesale, not per-file), so the fix re-copies each offending
# *.dist-info into a trimmed staging copy under build/ with just the
# licenses/third_party subtree dropped, and points datas at that copy
# instead - rather than editing the real site-packages install (shared
# with every other tool on this machine) or relying on every build
# machine having Windows long-path support enabled (a machine-wide
# registry/group-policy setting that can't be assumed here).
import shutil


def _trim_distinfo_license_bloat(datas):
    trimmed_root = _project_root / "build" / "_distinfo_trimmed"
    result = []
    for src, dest in datas:
        src_path = Path(src)
        if dest.endswith(".dist-info") and (src_path / "licenses" / "third_party").is_dir():
            trimmed_src = trimmed_root / src_path.name
            if trimmed_src.exists():
                shutil.rmtree(trimmed_src)
            shutil.copytree(src, trimmed_src, ignore=shutil.ignore_patterns("third_party"))
            result.append((str(trimmed_src), dest))
        else:
            result.append((src, dest))
    return result


_project_root = Path(SPECPATH)
datas = _trim_distinfo_license_bloat(datas)

# Seed a couple of stock YOLO models into the bundle's Resources/models so
# config.ensure_default_models_seeded() has something to copy into the
# user's writable per-user data folder on first run.
datas += [
    ("yolov8n.pt", "models"),
    ("app_icon.png", "."),
    ("logo.png", "."),
    ("assets/fonts", "assets/fonts"),  # pixel UI fonts (ui/pixel_theme.py)
]

# v6.0.0: no loose copy of the ai_detection_lab/ source ships any more. Only a
# text copy of config.py goes in, under app_params/, for Quick Parameters.
datas.append((str(_project_root / "ai_detection_lab" / "config.py"), "app_params"))

a = Analysis(
    ["entry_point.py"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    pathex=[str(_project_root / "vendor")],   # pyhulax 0.2.0 (tested copy)
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["build_assets/rthook_fix_logging.py"],
    # Export is .pt only (2026-10-13): keep the ONNX / CoreML converters out
    # even though ultralytics' exporter mentions them.
    excludes=["onnx", "onnxruntime", "onnxslim", "coremltools"],
    noarchive=False,
    cipher=block_cipher,
    # torch._numpy._ufuncs (and friends under torch._numpy) attach ufuncs to
    # themselves at import time via bare `vars()[name] = ...` at module
    # scope. PyInstaller's own frozen-module importer (pyimod02_importers.py)
    # executes collected modules in a way that breaks that `vars()` trick,
    # raising "NameError: name 'name' is not defined" - confirmed via
    # AI Data & Training/<session>/training_error.log, reproduced only in
    # the frozen build (not from source) and only once training reaches
    # ultralytics' init_seeds() -> torch.use_deterministic_algorithms() ->
    # lazy `import torch._inductor` -> ... -> `torch._numpy._ufuncs`.
    # Forcing this submodule to be collected as loose source (imported via
    # Python's normal filesystem loader instead of PyInstaller's PYZ
    # loader) sidesteps the incompatibility; verified with a minimal
    # torch-only frozen build before applying here.
    #
    # v9.0.0: torch/version.py and torchvision/version.py ship as loose files
    # too. The GPU Pack (ai/gpupack.py) replaces them with the CUDA build's
    # copies; frozen into the PYZ, the CPU copy always won and the app said
    # "PyTorch in this app is the CPU build" even with the pack installed.
    module_collection_mode={"torch._numpy": "py", "torch.version": "py", "torchvision.version": "py"},
)

# v6.0.0: the app's own code ships as ONE Nuitka-compiled module
# (build/nuitka/ai_detection_lab.*.pyd) instead of .pyc files in the PYZ.
# Analysis above still reads the .py source to find every library it imports.
_compiled = sorted((_project_root / "build" / "nuitka").glob("ai_detection_lab.*.pyd"))
if not _compiled:
    raise SystemExit("build/nuitka/ai_detection_lab.*.pyd missing - run build_windows.py")
a.pure = [entry for entry in a.pure if entry[0] != "ai_detection_lab" and not entry[0].startswith("ai_detection_lab.")]
a.binaries += [(_compiled[0].name, str(_compiled[0]), "EXTENSION")]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AIR Detection",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    icon="build_assets/app_icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AIR Detection",
)
