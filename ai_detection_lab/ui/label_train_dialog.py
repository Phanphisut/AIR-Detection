"""
AI Object Detection trainer popup for AI Detection Lab.

Opened from a button on the main window ("🏷️ Label & Train AI"). Lets a
student go from a raw .mp4 recording all the way to a working YOLO model
without leaving the app, in three steps (one tab each):

    1. Import Files     : bring in photos to label - either existing image
       files/folders, or an .mp4/.avi/.mov cut into stills at a
       configurable interval. A model trained elsewhere can also be
       imported straight into the models folder here, skipping steps 2-3
       entirely.
    2. Label Objects    : pick a photo out of the gallery, draw bounding
       boxes on it by hand (drag to draw, drag to move/resize) and assign
       them to a class - the classic YOLO labeling workflow.
    3. Train & Export   : turn the labeled frames into a YOLO dataset,
       fine-tune a YOLOv8 model on it (runs in the background), then copy
       the resulting best.pt straight into this project's models/ folder
       so it shows up next time the AI camera or the webcam test scans
       for models - no manual file copying needed.

Everything needed for this (PySide6, OpenCV, ultralytics, torch) is
already part of this project's embedded Python - no extra installs.

Each click of "Open Video" starts a fresh labeling session under
<project>/AI Data & Training/training_sessions/<video-name>_<timestamp>/ -
frames, label .txt files (standard YOLO format: "class_id cx cy w h",
normalized 0-1) and a classes.txt all live there so nothing is lost if the
dialog is closed and reopened, although resuming a previous session from
the UI is not supported yet (a fresh "Open Video" always starts a new one).
"""
from __future__ import annotations

import datetime
import inspect
import json
import random
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from .. import config
from ..i18n import tr
from ..image_io import imread_any, imwrite_any
from .code_editor import DIALOG_STYLESHEET, EN_US_LOCALE
from .video_clip_test_dialog import VideoClipTestDialog

# ---------------------------------------------------------------------------
# Constants / defaults ("fast & light" per the lab's default training preset)
# ---------------------------------------------------------------------------

TRAINING_ROOT = config.DATA_ROOT / "training_sessions"
DEFAULT_FRAME_INTERVAL_SEC = 1.0
DEFAULT_EPOCHS = 40
TRAIN_IMGSZ = 640
VAL_RATIO = 0.15
MIN_LABELED_FRAMES = 4
MAX_BACKGROUND_RATIO = 0.5  # no-object photos used per labeled photo (at most)
BASE_MODEL_NAME = "yolov8n.pt"  # smallest/fastest YOLOv8 checkpoint

# --- Auto confidence calibration -------------------------------------------
# One fixed confidence threshold (config.YOLO_CONF_THRES) applied to every
# model regardless of how it was trained is a bad fit in practice: measured
# directly on a real trained model in this project, its own F1 curve showed
# recall collapsing from 100% to 55% just by using conf=0.8 instead of its
# own optimum (~0.64), and further collapsing to 0% by conf=0.9. Instead,
# each freshly trained model gets its OWN calibrated threshold, computed from
# that model's real post-training validation results (see
# _calibrate_confidence() below) and stored in MODEL_CONF_REGISTRY_FILENAME
# next to it. yolo_detector.py reads that registry and uses the per-model
# value when present, falling back to config.YOLO_CONF_THRES for any model
# that has no calibration on record (e.g. one dropped in manually, or
# trained before this feature existed).
MODEL_CONF_REGISTRY_FILENAME = "model_confidence.json"
# Sanity clamp on the auto-picked value. Calibration is measured from very
# small validation sets (often 1-4 images) so the raw curve can be noisy;
# these bounds keep an unlucky calibration from landing somewhere pathological
# (e.g. near-0 = accepts almost anything as a detection, or near-1 = the
# recall-collapse zone observed above) while still letting each model differ
# from every other one.
MIN_AUTO_CONF = 0.15
MAX_AUTO_CONF = 0.75

# --- Training-data diversity guard ------------------------------------------
# Root-caused from a real field failure: a model trained on 27 frames that
# were all extracted from ONE continuous close-up indoor selfie clip (same
# person, same background, same lighting, drone always filling the frame at
# arm's length) scored well in validation (mAP50 ~0.75-0.99) because its 4
# held-out "val" images were just more near-duplicate frames from that same
# clip - but during a real outdoor farmland flight it fired scattered
# false-positive boxes everywhere. The model never learned what "drone"
# looks like in general; it learned to associate the label with whatever
# incidental background/lighting cues were in that one video, and - because
# every one of its training frames contained the object - it never saw a
# single example of "no object here" to anchor against.
#
# DIVERSITY_THUMB_SIZE / _estimate_frame_diversity() below measure this
# directly (mean pairwise pixel difference across small grayscale
# thumbnails of the labeled frames) instead of guessing. Calibrated against
# two real datasets, not synthetic noise: this exact 27-frame homogeneous
# session scored 0.068, while a set of unrelated real photographs (distinct
# scenes, subjects, and lighting) scored 0.234 - a 3.5x gap. MIN_FRAME_DIVERSITY
# sits well inside that gap so a dataset like this one trips the warning
# while a genuinely varied one does not.
DIVERSITY_THUMB_SIZE = 48
MIN_FRAME_DIVERSITY = 0.10

CLASS_COLOR_HEX = [
    "#33d17a", "#ffd166", "#5aa9ff", "#ff6b6b", "#c77dff",
    "#f4a261", "#4cc9f0", "#f72585", "#90be6d", "#e0e0e0",
]

# Step-2 gallery thumbnails: GALLERY_THUMB_PX is the long edge each photo
# is decoded to, GALLERY_THUMB_H the height of the cell it is drawn in.
# Sizing the cell 4:3 rather than square keeps ordinary landscape video
# frames from sitting in a tall box padded with dead space, which is what
# decides how many photos fit on screen without scrolling.
GALLERY_THUMB_PX = 168
GALLERY_THUMB_H = 126

# Photo formats accepted by "Import Photos" / "Import Folder" on step 1.
# Everything is normalized to frame_NNNNN.jpg inside the session, because
# every step after this one (the gallery, build_dataset_split, training)
# looks for *.jpg in the session's frames folder.
IMAGE_IMPORT_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def class_colors() -> list[QtGui.QColor]:
    return [QtGui.QColor(h) for h in CLASS_COLOR_HEX]


# ---------------------------------------------------------------------------
# Pure helpers - YOLO label file I/O. Kept dependency-free and side-effect
# free (besides the actual read/write) so they're easy to unit test.
# ---------------------------------------------------------------------------

def label_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def no_object_path_for(image_path: Path) -> Path:
    """Marker file saying "this photo was checked and has NO object in it"."""
    return image_path.with_suffix(".noobject")


def is_no_object(image_path: Path) -> bool:
    return no_object_path_for(image_path).exists()


def set_no_object(image_path: Path, flag: bool) -> None:
    marker = no_object_path_for(image_path)
    if flag:
        marker.write_text("checked: no object in this photo\n", encoding="utf-8")
    elif marker.exists():
        marker.unlink()


def _is_labeled(p: Path) -> bool:
    return label_path_for(p).exists() and label_path_for(p).stat().st_size > 0


def read_yolo_labels(txt_path: Path) -> list[dict]:
    if not txt_path.exists():
        return []
    boxes = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            continue
        boxes.append(dict(class_id=cid, cx=cx, cy=cy, w=w, h=h))
    return boxes


def write_yolo_labels(txt_path: Path, boxes: list[dict]) -> None:
    lines = [
        f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
        for b in boxes
    ]
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_classes(classes_path: Path) -> list[str]:
    if not classes_path.exists():
        return []
    return [
        line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_classes(classes_path: Path, classes: list[str]) -> None:
    classes_path.write_text("\n".join(classes) + ("\n" if classes else ""), encoding="utf-8")


def new_session_dir(source_path: Path) -> Path:
    """A labeling session folder named after whatever started it - the
    video that was cut into frames, or the folder/first photo that was
    imported."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in source_path.stem)[:40]
    return TRAINING_ROOT / f"{safe_stem}_{stamp}"


# ---------------------------------------------------------------------------
# Projects. A "project" is just one of these session folders - photos, their
# .txt labels and classes.txt all live inside it - so every session ever
# created (including the ones made before projects had a UI) shows up in the
# picker and can be reopened and added to.
# ---------------------------------------------------------------------------

def project_dir_for_name(name: str) -> Path:
    safe = "".join(c if (c.isalnum() or c in "-_ ") else "_" for c in name.strip())[:60]
    safe = "_".join(safe.split()) or "project"
    return TRAINING_ROOT / safe


def list_projects() -> list[Path]:
    """Every project folder, most recently touched first."""
    if not TRAINING_ROOT.is_dir():
        return []
    found = [
        p for p in TRAINING_ROOT.iterdir()
        if p.is_dir() and ((p / "frames").is_dir() or (p / "classes.txt").exists())
    ]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def project_summary(project_dir: Path) -> dict:
    frames = sorted((project_dir / "frames").glob("*.jpg")) if (project_dir / "frames").is_dir() else []
    labeled = sum(
        1 for p in frames
        if label_path_for(p).exists() and label_path_for(p).stat().st_size > 0
    )
    return dict(
        path=project_dir,
        name=project_dir.name,
        photos=len(frames),
        labeled=labeled,
        classes=read_classes(project_dir / "classes.txt"),
        modified=datetime.datetime.fromtimestamp(project_dir.stat().st_mtime),
    )


def next_frame_index(frames_dir: Path) -> int:
    """One past the highest frame number already in the folder, so newly
    added photos never overwrite ones that may already be labeled."""
    if not frames_dir.is_dir():
        return 0
    highest = -1
    for p in frames_dir.glob("frame_*.jpg"):
        try:
            highest = max(highest, int(p.stem.split("_")[-1]))
        except ValueError:
            continue
    return highest + 1


class ProjectPickerDialog(QtWidgets.QDialog):
    """Pick which existing project to carry on with."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Open a Project"))
        self.setStyleSheet(DIALOG_STYLESHEET)
        self.resize(620, 440)
        self.selected: Path | None = None

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel(tr("YOUR PROJECTS"))
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        hint = QtWidgets.QLabel(tr(
            "Pick a project to carry on labeling where you left off - all its "
            "photos, boxes and classes come back exactly as you left them."
        ))
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.list = QtWidgets.QListWidget()
        layout.addWidget(self.list, 1)

        self.projects = list_projects()
        for project in self.projects:
            info = project_summary(project)
            classes = ", ".join(info["classes"]) if info["classes"] else tr("no classes yet")
            item = QtWidgets.QListWidgetItem(
                f"{info['name']}\n"
                f"    {info['photos']} {tr('photos')}  •  {info['labeled']} {tr('labeled')}  •  {classes}"
                f"  •  {tr('last used')} {info['modified']:%Y-%m-%d %H:%M}"
            )
            self.list.addItem(item)
        if self.projects:
            self.list.setCurrentRow(0)
        else:
            empty = QtWidgets.QLabel(tr(
                "No projects yet. Close this and press “➕ New Project” to make your first one."
            ))
            empty.setObjectName("muted")
            empty.setWordWrap(True)
            layout.addWidget(empty)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Open | QtWidgets.QDialogButtonBox.Cancel
        )
        self.open_button = buttons.button(QtWidgets.QDialogButtonBox.Open)
        self.open_button.setEnabled(bool(self.projects))
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.list.itemDoubleClicked.connect(lambda _item: self._accept_selection())

    def _accept_selection(self):
        row = self.list.currentRow()
        if 0 <= row < len(self.projects):
            self.selected = self.projects[row]
            self.accept()


SPLIT_CONFIG_FILENAME = "split.json"
MAX_TEST_RATIO = 0.3


def read_split_config(project_dir: Path | None) -> dict:
    """Validation/test ratios saved per project (Datasets → Splits).

    Projects without a split.json keep the original behavior: 15 % val and
    no held-out test split.
    """
    ratios = {"val": VAL_RATIO, "test": 0.0}
    if project_dir is None:
        return ratios
    try:
        data = json.loads((project_dir / SPLIT_CONFIG_FILENAME).read_text(encoding="utf-8"))
        ratios["val"] = min(0.5, max(0.05, float(data.get("val", VAL_RATIO))))
        ratios["test"] = min(MAX_TEST_RATIO, max(0.0, float(data.get("test", 0.0))))
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return ratios


def write_split_config(project_dir: Path, val_ratio: float, test_ratio: float) -> None:
    (project_dir / SPLIT_CONFIG_FILENAME).write_text(
        json.dumps({"val": round(val_ratio, 4), "test": round(test_ratio, 4)}) + "\n",
        encoding="utf-8",
    )


def plan_dataset_split(images: list[Path], val_ratio: float = VAL_RATIO,
                       test_ratio: float = 0.0, seed: int = 42) -> dict:
    """Decide which photo lands in train / val / test, without copying.

    The single source of truth for both training (build_dataset_split) and
    the Studio's split view, so what the gallery shows is exactly what the
    trainer uses. With test_ratio == 0 this reproduces the original
    two-way split image for image.
    """
    labeled = [p for p in images if _is_labeled(p)]
    backgrounds = [p for p in images if is_no_object(p) and not _is_labeled(p)]
    shuffled = list(labeled)
    random.Random(seed).shuffle(shuffled)
    n_val = test_n = 0
    if len(shuffled) >= 2:
        n_val = max(1, round(len(shuffled) * val_ratio))
        n_val = min(n_val, len(shuffled) - 1)  # always leave >=1 for train
        if test_ratio > 0:
            test_n = max(1, round(len(shuffled) * test_ratio))
            test_n = max(0, min(test_n, len(shuffled) - n_val - 1))
    val_set = shuffled[:n_val]
    test_set = shuffled[n_val:n_val + test_n]
    train_set = shuffled[n_val + test_n:]

    bg_shuffled = list(backgrounds)
    random.Random(seed + 1).shuffle(bg_shuffled)
    # Too many no-object photos vs labeled ones makes the model timid (it
    # stops trusting its own detections). Measured on a real project with
    # 9 labeled + 12 no-object photos: false alarms 13/13 -> 0/13, but only
    # 5/10 objects still found. Use at most MAX_BACKGROUND_RATIO x labeled.
    bg_shuffled = bg_shuffled[: max(1, int(len(labeled) * MAX_BACKGROUND_RATIO))]
    n_val_bg = round(len(bg_shuffled) * val_ratio)
    n_test_bg = round(len(bg_shuffled) * test_ratio) if test_ratio > 0 else 0
    return dict(
        train=train_set, val=val_set, test=test_set,
        train_bg=bg_shuffled[n_val_bg + n_test_bg:],
        val_bg=bg_shuffled[:n_val_bg],
        test_bg=bg_shuffled[n_val_bg:n_val_bg + n_test_bg],
        labeled=labeled, backgrounds=backgrounds,
    )


def split_membership(plan: dict) -> dict[Path, str]:
    """Photo -> "train" / "val" / "test" for every photo the trainer uses."""
    member = {}
    for split_name in ("train", "val", "test"):
        for p in plan[split_name] + plan[split_name + "_bg"]:
            member[p] = split_name
    return member


def build_dataset_split(
    frames_dir: Path,
    classes: list[str],
    dataset_dir: Path,
    val_ratio: float | None = None,
    seed: int = 42,
    test_ratio: float | None = None,
) -> dict:
    """Copies every labeled frame (image that has a non-empty matching
    .txt) into <dataset_dir>/images/{train,val[,test]} + labels/..., and
    writes data.yaml. Returns counts. Raises ValueError if there aren't
    enough labeled frames yet. Ratios default to the project's split.json.
    """
    saved = read_split_config(frames_dir.parent)
    val_ratio = saved["val"] if val_ratio is None else val_ratio
    test_ratio = saved["test"] if test_ratio is None else test_ratio
    images = sorted(frames_dir.glob("*.jpg"))
    plan = plan_dataset_split(images, val_ratio, test_ratio, seed)
    labeled = plan["labeled"]
    if len(labeled) < MIN_LABELED_FRAMES:
        raise ValueError(
            f"Only {len(labeled)} labeled frame(s) - need at least "
            f"{MIN_LABELED_FRAMES} before training can start."
        )

    # "🚫 No object" photos go in as background images (empty label file):
    # they teach the model what is NOT the object - without any, a model
    # trained on one clip fires on almost anything (measured on a real
    # project: 23/23 false alarms on photos with no object in them).
    #
    # Start from an EMPTY dataset folder every time. It used to be reused
    # between runs, so photos from an earlier split stayed behind: an image
    # could end up in BOTH train and val (inflating the score), and photos
    # that had since been unlabeled or deleted were still trained on.
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir, ignore_errors=True)

    splits = ["train", "val"] + (["test"] if plan["test"] or plan["test_bg"] else [])
    for split_name in splits:
        items, bgs = plan[split_name], plan[split_name + "_bg"]
        (dataset_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        for img_path in items:
            shutil.copy2(img_path, dataset_dir / "images" / split_name / img_path.name)
            shutil.copy2(
                label_path_for(img_path),
                dataset_dir / "labels" / split_name / label_path_for(img_path).name,
            )
        for img_path in bgs:
            shutil.copy2(img_path, dataset_dir / "images" / split_name / img_path.name)
            (dataset_dir / "labels" / split_name / label_path_for(img_path).name).write_text("", encoding="utf-8")

    names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))
    yaml_text = (
        f"path: {dataset_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        + ("test: images/test\n" if "test" in splits else "")
        + f"names:\n{names_yaml}\n"
    )
    data_yaml_path = dataset_dir / "data.yaml"
    data_yaml_path.write_text(yaml_text, encoding="utf-8")

    used_bg = plan["train_bg"] + plan["val_bg"] + plan["test_bg"]
    return dict(
        train=len(plan["train"]) + len(plan["train_bg"]),
        val=len(plan["val"]) + len(plan["val_bg"]),
        test=len(plan["test"]) + len(plan["test_bg"]),
        backgrounds=len(used_bg),
        backgrounds_available=len(plan["backgrounds"]),
        background_names=[p.name for p in used_bg],
        total_labeled=len(labeled),
        total_frames=len(images),
        data_yaml=data_yaml_path,
    )


def _estimate_frame_diversity(image_paths: list[Path]) -> float | None:
    """Mean pairwise normalized absolute difference (0..1) across small
    grayscale thumbnails of the given images. ~0 means the frames are
    near-duplicates of each other (e.g. consecutive frames pulled from one
    short, mostly-static video clip); a higher score means the frames show
    visually distinct scenes (different backgrounds, distances, lighting).

    This is deliberately cheap and dependency-free (just OpenCV, already a
    hard requirement of this app) rather than a learned similarity model -
    it only needs to catch the specific, real failure mode it was built
    for: an entire training set drawn from one continuous clip. See the
    MIN_FRAME_DIVERSITY comment above for how the threshold was picked.

    Returns None if fewer than 2 images are given (nothing to compare) or
    if none of the images could be read.
    """
    thumbs = []
    for p in image_paths:
        img = imread_any(p)
        if img is None:
            continue
        img = cv2.resize(img, (DIVERSITY_THUMB_SIZE, DIVERSITY_THUMB_SIZE),
                          interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
        thumbs.append(gray)
    if len(thumbs) < 2:
        return None
    diffs = []
    for i in range(len(thumbs)):
        for j in range(i + 1, len(thumbs)):
            diffs.append(float(abs(thumbs[i] - thumbs[j]).mean()) / 255.0)
    return sum(diffs) / len(diffs)


def suggest_model_name(classes: list[str], fallback: str = "custom_model") -> str:
    """Builds a filename hint from what was actually labeled (e.g.
    ["Car", "Sign"] -> "car_sign") instead of an opaque session-folder
    timestamp, so the auto-saved .pt file's name means something at a
    glance without the student having to type one."""
    parts = []
    for name in classes[:3]:
        cleaned = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name.strip().lower())
        cleaned = cleaned.strip("_")
        if cleaned:
            parts.append(cleaned)
    if len(classes) > 3:
        parts.append(f"plus{len(classes) - 3}")
    return "_".join(parts) or fallback


def unique_model_filename(models_dir: Path, base_name: str) -> str:
    """Avoids clobbering an existing .pt in the models folder."""
    stem = Path(base_name).stem or "custom_model"
    stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)
    candidate = f"{stem}.pt"
    n = 1
    while (models_dir / candidate).exists():
        candidate = f"{stem}_{n}.pt"
        n += 1
    return candidate


def _patch_inspect_getsource_for_six_workaround():
    """Works around a real crash confirmed via the full traceback saved to
    this project's training_error.log: right as training finishes,
    ultralytics does a deferred `import matplotlib.pyplot` (to plot the
    PR curve) for the very first time in the process. That import chain
    pulls in python-dateutil, which does `from six.moves import _thread` -
    a virtual module served by `six`'s `_SixMetaPathImporter`, an importer
    that predates Python's modern `find_spec()` protocol.

    Because this app is built on PySide6, PySide6's own "shiboken
    signature loader" installs a hook that calls `inspect.getsource(module)`
    on every module right after it finishes importing, to check whether it
    uses Qt. `six`'s virtual module doesn't play nice with `inspect`'s
    module-repr machinery and raises a raw
    "AttributeError: '_SixMetaPathImporter' object has no attribute
    '_path'" there, instead of the OSError/TypeError that
    `inspect.getsource()` is documented to raise when source isn't
    available. PySide6's hook only expects those two, so the unexpected
    AttributeError crashes the whole import chain (and training) instead
    of being swallowed.

    The fix: for the duration of training, make `inspect.getsource()`
    translate ANY unexpected exception into a plain OSError - exactly what
    its own documentation says callers should expect and handle - so
    PySide6's hook (and anything else calling it) degrades gracefully
    instead of crashing. Returns the original function so it can be
    restored afterwards."""
    original = inspect.getsource

    def _patched(obj, *args, **kwargs):
        try:
            return original(obj, *args, **kwargs)
        except (OSError, TypeError):
            raise
        except Exception as exc:
            # Deliberately does NOT format `obj` (e.g. via repr()) into this
            # message - for the exact broken objects this works around,
            # repr(obj) is itself what raises, so doing that here would
            # just trigger the same crash a second time while building the
            # "safe" error message.
            raise OSError(f"source unavailable for a {type(obj).__name__} object: {exc}") from exc

    inspect.getsource = _patched
    return original


def _restore_inspect_getsource(original) -> None:
    inspect.getsource = original


# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------

class FrameExtractThread(threading.Thread):
    """Pulls still frames out of a video file at a fixed time interval."""

    def __init__(self, video_path: Path, out_dir: Path, interval_sec: float,
                 on_progress, on_status, on_error, on_finished, start_index: int = 0):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.out_dir = out_dir
        self.interval_sec = max(0.1, interval_sec)
        # Where to start numbering, so adding a second video to a project
        # appends to it instead of overwriting frame_00000.jpg onwards.
        self.start_index = start_index
        self.on_progress = on_progress
        self.on_status = on_status
        self.on_error = on_error
        self.on_finished = on_finished
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        saved = 0
        cap = None
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened() and not str(self.video_path).isascii():
                # OpenCV on Windows can't open a video whose path contains
                # non-ASCII characters (e.g. a Thai folder or file name).
                # Work on a temporary ASCII-named copy instead.
                cap.release()
                import tempfile
                tmp_dir = Path(tempfile.mkdtemp(prefix="ailab_video_"))
                tmp_copy = tmp_dir / ("video" + self.video_path.suffix.lower())
                self.on_status("📋 Copying video to a temporary folder (non-English path)...")
                shutil.copy2(self.video_path, tmp_copy)
                self._tmp_dir = tmp_dir
                cap = cv2.VideoCapture(str(tmp_copy))
            if not cap.isOpened():
                self.on_error(f"❌ Could not open video: {self.video_path.name}")
                self.on_finished(0)
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if fps <= 0:
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            step = max(1, round(fps * self.interval_sec))

            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.on_status(
                f"🎬 Extracting frames every {self.interval_sec:.1f}s "
                f"(~every {step} frames at {fps:.1f} fps)..."
            )

            idx = 0
            last_report = 0.0
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    out_path = self.out_dir / f"frame_{self.start_index + saved:05d}.jpg"
                    imwrite_any(out_path, frame)
                    saved += 1
                idx += 1
                # Report at most ~10x/sec: once per decoded frame queued
                # thousands of signals for a long video and froze the UI.
                now = time.monotonic()
                if total_frames and (now - last_report >= 0.1 or idx >= total_frames):
                    last_report = now
                    self.on_progress(idx, total_frames)

            if self._stop_event.is_set():
                self.on_status(f"⏹ Extraction stopped early - {saved} frame(s) saved")
            else:
                self.on_status(f"✅ Extracted {saved} frame(s)")
            self.on_finished(saved)
        except Exception as exc:
            self.on_error(f"❌ Frame extraction failed: {exc}")
            self.on_finished(saved)
        finally:
            if cap is not None:
                cap.release()
            tmp_dir = getattr(self, "_tmp_dir", None)
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)


class ImportImagesThread(threading.Thread):
    """Brings already-taken photos into a labeling session, as the
    alternative to cutting a video up.

    Photos are copied into the session's own frames folder rather than
    labeled where they sit: labeling writes a .txt next to every image, and
    scribbling those into the student's own photo folder (or worse, into a
    read-only one) is not ours to do.

    Files land as frame_NNNNN.jpg so the rest of the app - the gallery,
    build_dataset_split(), training - keeps working unchanged. A JPEG that
    needs no EXIF rotation is copied byte-for-byte (no re-encode, no
    quality loss); anything else is decoded with the EXIF orientation
    applied and written back out as JPEG. Baking the rotation in matters:
    QPixmap (the labeling canvas) ignores EXIF while ultralytics applies
    it, so an un-baked phone photo would be labeled at one orientation and
    trained at another, silently misplacing every box.
    """

    def __init__(self, sources: list[Path], out_dir: Path, on_progress, on_status,
                 on_error, on_finished, start_index: int = 0):
        super().__init__(daemon=True, name="AI-LAB-IMPORT-IMAGES")
        self.sources = sources
        self.out_dir = out_dir
        # Where to start numbering, so photos added to an existing project
        # append to it instead of overwriting the ones already labeled.
        self.start_index = start_index
        self.on_progress = on_progress
        self.on_status = on_status
        self.on_error = on_error
        self.on_finished = on_finished
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        saved = 0
        skipped: list[str] = []
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            total = len(self.sources)
            for i, src in enumerate(self.sources):
                if self._stop_event.is_set():
                    break
                dest = self.out_dir / f"frame_{self.start_index + saved:05d}.jpg"
                try:
                    reader = QtGui.QImageReader(str(src))
                    reader.setAutoTransform(True)
                    needs_rotation = (
                        reader.transformation() != QtGui.QImageIOHandler.TransformationNone
                    )
                    if src.suffix.lower() in (".jpg", ".jpeg") and not needs_rotation:
                        shutil.copy2(src, dest)
                    else:
                        image = reader.read()
                        if image.isNull() or not image.save(str(dest), "JPG", 92):
                            skipped.append(src.name)
                            continue
                    saved += 1
                except Exception:
                    skipped.append(src.name)
                self.on_progress(i + 1, total)

            if skipped:
                self.on_status(
                    f"⚠️ Skipped {len(skipped)} file(s) that could not be read: "
                    + ", ".join(skipped[:5])
                    + (f", +{len(skipped) - 5} more" if len(skipped) > 5 else "")
                )
            self.on_status(f"✅ Imported {saved} photo(s)")
            self.on_finished(saved)
        except Exception as exc:
            self.on_error(f"❌ Photo import failed: {exc}")
            self.on_finished(saved)


class ThumbnailLoadThread(threading.Thread):
    """Decodes the step-2 gallery's photo thumbnails off the GUI thread.

    A one-minute clip at the default 1-frame-per-second easily produces
    hundreds of full-resolution JPEGs; decoding those inline would freeze
    the window for seconds right after extraction. QImageReader.setScaledSize
    also lets each file be decoded straight to thumbnail size instead of
    full resolution first, which is what keeps this fast enough to finish
    while the student is still reading the page.

    Emits QImage (not QPixmap) - QPixmap may only be touched on the GUI
    thread, so the conversion happens in the receiving slot.
    """

    def __init__(self, paths: list[Path], generation: int, on_thumbnail):
        super().__init__(daemon=True, name="AI-LAB-THUMBS")
        self.paths = paths
        self.generation = generation
        self.on_thumbnail = on_thumbnail
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        for index, path in enumerate(self.paths):
            if self._stop_event.is_set():
                return
            try:
                reader = QtGui.QImageReader(str(path))
                reader.setAutoTransform(True)
                size = reader.size()
                if size.isValid():
                    reader.setScaledSize(
                        size.scaled(
                            GALLERY_THUMB_PX, GALLERY_THUMB_PX, QtCore.Qt.KeepAspectRatio
                        )
                    )
                image = reader.read()
            except Exception:
                continue
            if image.isNull() or self._stop_event.is_set():
                continue
            try:
                self.on_thumbnail(self.generation, index, image)
            except Exception:
                # The dialog's C++ side went away between that check and
                # this emit - i.e. the app is shutting down with thumbnails
                # still in flight. There is nothing left to deliver to, so
                # stop quietly instead of dumping a traceback on the way out.
                return


class ImportModelThread(threading.Thread):
    """Validates and copies an externally trained .pt into the project's
    models folder (the "Import Model" button on step 3).

    Loading a checkpoint costs a second or two of torch work, so it runs
    here rather than freezing the dialog. The task check matters: the AI
    camera's detector (ai/yolo_detector.py) loads every .pt in the models
    folder and reads `.boxes` off the results, so letting a classification
    or segmentation checkpoint land there would break detection later with
    a confusing error far away from this button.
    """

    def __init__(self, src: Path, on_finished):
        super().__init__(daemon=True, name="AI-LAB-IMPORT-MODEL")
        self.src = src
        self.on_finished = on_finished

    def run(self):
        try:
            from ultralytics import YOLO

            model = YOLO(str(self.src))
            task = getattr(model, "task", None)
            names = dict(getattr(model, "names", {}) or {})
        except Exception as exc:
            self.on_finished({
                "ok": False,
                "error": (
                    f"{self.src.name} could not be opened as a YOLO model: {exc}\n\n"
                    "Make sure it is a YOLOv8 .pt checkpoint (for example a "
                    "best.pt from an Ultralytics training run)."
                ),
            })
            return

        if task != "detect":
            self.on_finished({
                "ok": False,
                "error": (
                    f"{self.src.name} is a '{task}' model, not an object-detection "
                    "model.\n\nThe AI camera can only use detection models (the "
                    "kind that draws boxes). A classification model cannot be "
                    "imported here."
                ),
            })
            return

        try:
            config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            filename = unique_model_filename(config.MODEL_DIR, self.src.stem)
            dest = config.MODEL_DIR / filename
            shutil.copy2(self.src, dest)
        except Exception as exc:
            self.on_finished({"ok": False, "error": f"Could not copy the model into the models folder: {exc}"})
            return

        self.on_finished({"ok": True, "dest": dest, "classes": list(names.values())})


def _calibrate_confidence(model_cls, best_path: Path, data_yaml: Path, val_images: int,
                           runs_dir: Path | None = None, run_name: str = "train") -> dict | None:
    """Picks a confidence threshold for THIS specific trained model from its
    own post-training validation performance, instead of leaving every model
    stuck with one fixed global number.

    How it works: ultralytics' own validation run (the same `model.val()`
    step ultralytics itself uses to draw the "F1_curve.png" it saves in every
    training run's results folder) computes, per class, precision/recall/F1
    at 1000 confidence thresholds evenly spaced across 0..1. Nothing here is
    invented - `box.px`/`box.f1_curve`/`box.p_curve`/`box.r_curve` are
    ultralytics' real, documented Metric attributes. We average F1 across
    classes at each threshold and take the confidence that maximizes it: the
    threshold that gives the best real balance, for THIS model, measured on
    THIS model's own held-out validation images, of "catches real
    detections" vs "rejects noise".

    Because these validation sets are often only 1-4 images, the curve is
    coarse and can have a wide flat plateau where many thresholds tie for
    the same top F1 score. Taking the very edge of that plateau would be
    fragile (a slightly different lighting/angle next time could fall just
    past it), so we take the MIDDLE of the widest near-maximal plateau
    instead - the most stable choice available from this little data.

    Returns None (caller then falls back to the existing global
    config.YOLO_CONF_THRES, untouched) if there isn't enough validation data
    to measure anything, or if anything about the validation run itself
    fails - this never fabricates a number it can't measure.

    On disk: ultralytics' val() always writes its results somewhere - by
    default that's a bare runs/detect/val folder created relative to
    wherever the process's cwd happens to be, which for this app means
    littering the *project root* with an unrelated, ever-incrementing
    runs/detect/val, val2, val3... on every single training run. We pass
    plots=False (we only need the numeric curves already read below, not
    the PNG plots ultralytics would otherwise render) and point
    project/name at this session's own runs_dir - the same folder training
    already writes into - with exist_ok=True so re-running calibration on
    the same session overwrites in place instead of incrementing. Verified
    empirically: without these three arguments this call creates a stray
    runs/detect/val directory in the project root every time.
    """
    if val_images < 1:
        return None
    try:
        calib_model = model_cls(str(best_path))
        val_kwargs = {"data": str(data_yaml), "split": "val", "verbose": False, "plots": False, "workers": 0,
                      "iou": float(getattr(config, "YOLO_NMS_IOU", 0.5))}
        if runs_dir is not None:
            val_kwargs["project"] = str(runs_dir)
            val_kwargs["name"] = f"{run_name}_calibrate"
            val_kwargs["exist_ok"] = True
        metrics = calib_model.val(**val_kwargs)
        box = metrics.box
        px = box.px  # the 1000 confidence thresholds ultralytics evaluated, 0..1
        f1_by_class = box.f1_curve  # shape (num_classes, 1000)
        if f1_by_class.size == 0:
            return None
        f1 = f1_by_class.mean(axis=0)  # mean F1 across classes at each threshold
        max_f1 = float(f1.max())
        if max_f1 <= 0.0:
            return None

        near_max = f1 >= (max_f1 - 1e-4)
        idxs = [i for i, hit in enumerate(near_max) if hit]
        mid_idx = idxs[len(idxs) // 2]
        raw_conf = float((px[idxs[0]] + px[idxs[-1]]) / 2.0)
        conf_thres = max(MIN_AUTO_CONF, min(MAX_AUTO_CONF, raw_conf))

        return {
            "conf_thres": round(conf_thres, 3),
            "raw_conf_before_clamp": round(raw_conf, 3),
            "f1_at_conf": round(float(f1[mid_idx]), 4),
            "precision_at_conf": round(float(box.p_curve.mean(axis=0)[mid_idx]), 4),
            "recall_at_conf": round(float(box.r_curve.mean(axis=0)[mid_idx]), 4),
            "map50": round(float(box.map50), 4),
            "val_images": val_images,
            "method": "f1_curve_plateau_midpoint",
            "global_default_was": config.YOLO_CONF_THRES,
            "calibrated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        print(f"⚠️ Confidence auto-calibration failed, keeping global default: {exc}")
        return None


class TrainAIThread(threading.Thread):
    """Builds the train/val split and fine-tunes a YOLOv8 model on it."""

    def __init__(self, frames_dir: Path, classes: list[str], dataset_dir: Path,
                 runs_dir: Path, run_name: str, epochs: int,
                 on_progress, on_status, on_error, on_finished, device=None,
                 dataset_builder=None, imgsz=None, batch=None, extra_callbacks=None,
                 train_overrides=None):
        super().__init__(daemon=True)
        # Swarm Detection Lab "สอน AI": the kid wizard splits by video time
        # instead of at random, trains faster at a smaller size, and shows a
        # preview after each epoch. None keeps the Studio's behaviour.
        self.dataset_builder = dataset_builder
        self.imgsz = imgsz or TRAIN_IMGSZ
        self.batch = batch
        self.extra_callbacks = extra_callbacks or {}
        self.train_overrides = train_overrides or {}
        # "auto" / "mps" / "cuda" / "cpu" (ai/devices.py); None = config.TRAIN_DEVICE
        self.requested_device = device if device is not None else getattr(config, "TRAIN_DEVICE", "auto")
        self.device_used = None  # what training actually ran on (after any CPU fallback)
        self.frames_dir = frames_dir
        self.classes = classes
        self.dataset_dir = dataset_dir
        self.runs_dir = runs_dir
        self.run_name = run_name
        self.epochs = max(1, epochs)
        self.on_progress = on_progress
        self.on_status = on_status
        self.on_error = on_error
        self.on_finished = on_finished
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        best_path = None
        # A previous run's held-out score must never be shown for this run.
        (self.runs_dir / self.run_name / "test_metrics.json").unlink(missing_ok=True)
        try:
            self.on_status("📦 Building YOLO dataset (train/val split)...")
            builder = self.dataset_builder or build_dataset_split
            info = builder(self.frames_dir, self.classes, self.dataset_dir)
            self.on_status(
                f"📦 Dataset ready: {info['train']} train / {info['val']} val"
                + (f" / {info['test']} test" if info.get('test') else "") + " "
                f"images (from {info['total_labeled']} labeled frame(s)"
                f" + {info.get('backgrounds', 0)} 🚫 no-object photo(s))"
            )
            if info.get("backgrounds_available", 0) > info.get("backgrounds", 0):
                self.on_status(
                    f"ℹ️ Using {info['backgrounds']} of {info['backgrounds_available']} 🚫 photos "
                    f"(at most {MAX_BACKGROUND_RATIO:.0%} of the labeled photos, so the AI doesn't become "
                    "too timid). Label more photos to use more of them."
                )
            self._background_names = set(info.get("background_names", []))
        except Exception as exc:
            self.on_error(f"❌ Could not build dataset: {exc}")
            self.on_finished(None, None)
            return

        _orig_getsource = _patch_inspect_getsource_for_six_workaround()
        try:
            try:
                from ultralytics import YOLO
            except Exception as exc:
                self.on_error(f"❌ Could not import ultralytics/YOLO: {exc}")
                self.on_finished(None, None)
                return

            if config.BASE_YOLO_WEIGHTS is not None:
                self.on_status(
                    f"🚀 Starting training: {BASE_MODEL_NAME} (bundled, offline), "
                    f"{self.epochs} epochs..."
                )
                model = YOLO(str(config.BASE_YOLO_WEIGHTS))
            else:
                self.on_status(
                    f"🚀 Starting training: {BASE_MODEL_NAME}, {self.epochs} epochs "
                    f"(first run may download the base weights if not cached)..."
                )
                model = YOLO(BASE_MODEL_NAME)

            def _on_epoch_end(trainer):
                try:
                    epoch = int(getattr(trainer, "epoch", -1)) + 1
                    total = int(getattr(trainer, "epochs", self.epochs)) or self.epochs
                    self.on_progress(epoch, total)
                except Exception:
                    pass
                if self._stop_event.is_set():
                    # Best-effort cooperative stop: ultralytics' training
                    # loop checks trainer.stop for early-stopping and exits
                    # cleanly at the end of the current epoch if set. If a
                    # future ultralytics version drops this attribute this
                    # just silently has no effect - training still
                    # finishes normally, it just can't be cut short.
                    try:
                        trainer.stop = True
                    except Exception:
                        pass

            # Ultralytics' default batch=16 together with its default
            # nbs=64 ("nominal batch size") makes it accumulate gradients
            # over accumulate=round(nbs/batch)=4 iterations before every
            # real optimizer step. Our datasets are tiny (often under 16
            # images total), so there is only ONE actual batch per epoch -
            # meaning with the defaults, the optimizer only steps once
            # every ~4 EPOCHS. A "40 epoch" run was therefore really only
            # about 10 real weight updates, nowhere near enough to move the
            # detection head off its untrained starting point. This is the
            # confirmed, reproduced root cause of trained models scoring
            # under 3% confidence on EVERY image, including their own
            # training images (verified by re-running training on a real
            # exported session's data with save_period=1 and inspecting
            # every epoch's checkpoint directly).
            #
            # Fix: scale batch to the dataset size (so a small dataset
            # still gets more than one batch per epoch when possible) and
            # set nbs == batch so every batch immediately triggers a real
            # optimizer step - no accumulation delay. Verified end-to-end
            # on a real 15-image/3-val session: this alone took a trained
            # model from ~1-3% confidence on every image to 80-99%
            # confidence on the correct box, with mAP50 reaching ~0.99.
            #
            # (A stronger-augmentation approach was tried and reverted: on
            # datasets this small it destabilized training instead of
            # helping - mAP50 repeatedly collapsed to 0 mid-run. The real
            # fix was the optimizer step count, not augmentation strength.)
            # ...and never put the WHOLE dataset in one batch. Ultralytics'
            # warm-up always lasts at least 100 optimizer steps; with one
            # batch per epoch, a 100-epoch run on 8 photos was 100 steps -
            # i.e. warm-up only, and the model never really learned.
            # Measured on a real student project (8 train photos, 100
            # epochs): batch=8 -> max confidence 1.5% on its OWN training
            # photos (nothing ever detected); batch=2 -> 98-99.9% on every
            # frame, including frames it was never trained on. Aim for at
            # least ~4 batches per epoch.
            train_batch = self.batch or max(2, min(16, info["train"] // 4))
            # A fraction-of-epochs patience (tried and REVERTED - see below)
            # sounded reasonable but was proven wrong by a real training run
            # and caused a worse regression than the overfitting problem it
            # was meant to prevent, so this app does not use ultralytics'
            # early stopping at all: patience is set equal to the requested
            # epoch count, which ultralytics treats as "never stop early."
            #
            # What actually happened: a real 60-epoch run on a genuinely
            # diverse 36-frame dataset (dataset/labels: 31 train / 5 val)
            # got patience=max(15, round(60*0.3))=18 from the formula this
            # comment used to describe. Its results.csv shows fitness
            # (0.1*mAP50 + 0.9*mAP50-95) spiking to ~0.56 at epoch 2 - a
            # fluke from a barely-trained model getting lucky on only 5 val
            # images - then falling and staying far below that (~0.12-0.20)
            # through epoch 20, at which point 18 epochs with "no
            # improvement" over that fluke triggered early stopping.
            # Training kept the epoch-2 weights as best.pt: a model trained
            # for 2 real epochs, which is why it detected nothing at all
            # once deployed - not enough training had happened yet, early
            # stopping just never let it get there. With single-digit
            # validation-image counts, per-epoch fitness is noisy enough
            # that an early fluke reliably outranks real, later convergence
            # - the opposite of what patience is supposed to protect
            # against. (The original overfitting concern this was meant to
            # address - see the batch/nbs fix above - is real but better
            # left to training-data diversity and the calibrated confidence
            # threshold than to early stopping at this data scale.)
            train_patience = self.epochs
            # Ultralytics' own "automatic" device is cuda-or-cpu - it never
            # picks the Apple GPU, so this used to train on CPU on every Mac.
            # devices.resolve() picks mps / cuda / cpu for real (ai/devices.py).
            from ..ai import devices
            device, note = devices.resolve(self.requested_device)
            if note:
                self.on_status(note)
            self.on_status(f"🖥️ Training on {devices.describe(device)}")
            model.add_callback("on_train_epoch_end", _on_epoch_end)
            for event, callback in self.extra_callbacks.items():
                model.add_callback(event, callback)
            try:
                results = self._train(model, info, train_batch, train_patience, device)
            except Exception as exc:
                if device == "cpu" or self._stop_event.is_set():
                    raise
                # A GPU backend can fail where CPU works (an op mps does not
                # support yet, out of GPU memory...) - the student should
                # still get a model, just more slowly.
                self.on_status(f"⚠️ {devices.describe(device)} failed ({type(exc).__name__}: "
                               f"{str(exc).splitlines()[0][:160] if str(exc) else ''}) — retrying on CPU")
                device = "cpu"
                model = YOLO(str(config.BASE_YOLO_WEIGHTS) if config.BASE_YOLO_WEIGHTS is not None
                             else BASE_MODEL_NAME)
                model.add_callback("on_train_epoch_end", _on_epoch_end)
                for event, callback in self.extra_callbacks.items():
                    model.add_callback(event, callback)
                self.on_status("🖥️ Training on CPU")
                results = self._train(model, info, train_batch, train_patience, device)
            self.device_used = device
            best_path = self._locate_best_weights(results)
            if best_path is None:
                self.on_error("❌ Training finished but best.pt could not be found")
                self.on_finished(None, None)
                return
            self.on_status(f"✅ Training complete on {devices.describe(device)} - best model: {best_path}")
            self._self_check(YOLO, best_path)

            self.on_status("🎯 Calibrating confidence threshold from validation results...")
            calibration = _calibrate_confidence(
                YOLO, best_path, info["data_yaml"], info["val"],
                runs_dir=self.runs_dir, run_name=self.run_name,
            )
            if calibration is not None:
                self.on_status(
                    f"🎯 Auto-calibrated confidence: {calibration['conf_thres']:.2f} "
                    f"(F1={calibration['f1_at_conf']:.2f} on {calibration['val_images']} "
                    f"validation image(s), vs global default {config.YOLO_CONF_THRES:.2f})"
                )
            else:
                self.on_status(
                    "🎯 Not enough validation data to auto-calibrate confidence - "
                    f"will use the global default ({config.YOLO_CONF_THRES:.2f}) for this model."
                )

            if info.get("test"):
                self._evaluate_test_split(YOLO, best_path, info, train_batch)
            self.on_finished(best_path, calibration)
        except Exception as exc:
            self.on_error(f"❌ Training failed: {exc}")
            self._write_error_log(exc)
            self.on_finished(None, None)
        finally:
            _restore_inspect_getsource(_orig_getsource)

    def _train(self, model, info, train_batch, train_patience, device):
        """One ultralytics training run (run() explains batch / patience)."""
        return model.train(
            data=str(info["data_yaml"]),
            epochs=self.epochs,
            patience=train_patience,
            imgsz=self.imgsz,
            project=str(self.runs_dir),
            name=self.run_name,
            exist_ok=True,
            verbose=False,
            batch=train_batch,
            nbs=train_batch,
            device=device,
            # workers=0: load images in this same process. Extra
            # DataLoader worker processes are the #1 source of crashes
            # and hangs in the frozen Windows .exe / macOS .app (each
            # worker re-launches the app binary), and on datasets this
            # small they give no real speed-up.
            workers=0,
            # plots=False: ultralytics' training plots try to download a
            # font from the internet the first time - a stall or error on
            # classroom machines without internet. Nothing in this app
            # reads those PNGs.
            plots=False,
            **self.train_overrides,
        )

    def _evaluate_test_split(self, model_cls, best_path, info, batch) -> None:
        """Score the finished model once on the held-out test split.

        Test images are never seen during training or best.pt selection, so
        this is the honest estimate of real-world accuracy. Saved next to
        results.csv as test_metrics.json for the Studio's Train page.
        """
        try:
            self.on_status(f"🧪 Evaluating on {info['test']} held-out test image(s)...")
            metrics = model_cls(str(best_path)).val(
                data=str(info["data_yaml"]), split="test", imgsz=self.imgsz,
                batch=batch, workers=0, plots=False, verbose=False,
                project=str(self.runs_dir), name=f"{self.run_name}_test", exist_ok=True,
            )
            box = metrics.box
            result = {
                "images": info["test"],
                "metrics/precision(B)": float(box.mp),
                "metrics/recall(B)": float(box.mr),
                "metrics/mAP50(B)": float(box.map50),
                "metrics/mAP50-95(B)": float(box.map),
            }
            (self.runs_dir / self.run_name / "test_metrics.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8")
            self.on_status(
                f"🧪 Test split: mAP50 {result['metrics/mAP50(B)']:.3f} · "
                f"precision {result['metrics/precision(B)']:.3f} · recall {result['metrics/recall(B)']:.3f}"
            )
        except Exception as exc:
            self.on_status(f"⚠️ Test-split evaluation skipped: {exc}")

    def _self_check(self, model_cls, best_path) -> None:
        """Honest post-training check: does the new model find the object in
        the photos it was just taught? If it can't even do that, say so
        plainly (with what to do) instead of a bare "Training complete" -
        otherwise a student only discovers it later as "the camera never
        detects anything"."""
        try:
            bg_names = getattr(self, "_background_names", set())
            all_imgs = sorted((self.dataset_dir / "images" / "train").glob("*.jpg"))
            all_imgs += sorted((self.dataset_dir / "images" / "val").glob("*.jpg"))
            imgs = [p for p in all_imgs if p.name not in bg_names][:30]
            bg_imgs = [p for p in all_imgs if p.name in bg_names][:30]
            if not imgs:
                return
            model = model_cls(str(best_path))
            best_confs = []
            for p in imgs:
                r = model.predict(str(p), conf=0.001, verbose=False)[0]
                confs = [float(c) for c in (r.boxes.conf if r.boxes is not None else [])]
                best_confs.append(max(confs) if confs else 0.0)
            found = sum(1 for c in best_confs if c >= 0.5)
            avg = sum(best_confs) / len(best_confs)
            if bg_imgs:
                false_alarms = 0
                for p in bg_imgs:
                    r = model.predict(str(p), conf=0.5, verbose=False)[0]
                    false_alarms += int(r.boxes is not None and len(r.boxes) > 0)
                self.on_status(
                    f"🧪 Self-check: false alarms on 🚫 no-object photos: {false_alarms}/{len(bg_imgs)}"
                    + (" ✅" if false_alarms <= 0.1 * len(bg_imgs) else
                       " ⚠️ - the model still 'sees' the object where there is none; add more varied photos")
                )
            if found >= 0.8 * len(best_confs):
                self.on_status(
                    f"🧪 Self-check: found the object in {found}/{len(best_confs)} training photos "
                    f"(average confidence {avg:.0%}) ✅"
                )
            else:
                self.on_status(
                    f"⚠️ Self-check: the model found the object in only {found}/{len(best_confs)} of its own "
                    f"training photos (average confidence {avg:.0%}). It will probably detect little or nothing. "
                    "Add more labeled photos (20+ from different angles/backgrounds) and train again."
                )
        except Exception as exc:
            self.on_status(f"⚠️ Self-check skipped: {exc}")

    def _write_error_log(self, exc: Exception) -> None:
        """Best-effort: save the full traceback next to the session's
        dataset so a real cause can be pinned down if this ever fails for
        a reason the app doesn't already know how to work around - the
        session log only ever shows the short one-line message."""
        try:
            log_path = self.dataset_dir.parent / "training_error.log"
            log_path.write_text(
                f"{datetime.datetime.now().isoformat()}\n"
                f"{type(exc).__name__}: {exc}\n\n"
                f"{traceback.format_exc()}",
                encoding="utf-8",
            )
        except Exception:
            pass

    def _locate_best_weights(self, results):
        try:
            save_dir = Path(getattr(results, "save_dir"))
            candidate = save_dir / "weights" / "best.pt"
            if candidate.exists():
                return candidate
        except Exception:
            pass
        try:
            candidates = sorted(
                self.runs_dir.glob(f"{self.run_name}*/weights/best.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return candidates[0]
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# AnnotationCanvas - draws the current frame + its boxes, handles mouse
# drag-to-draw and click-to-select. Geometry/editing logic is split into
# small pure(ish) methods so it can be exercised directly in tests without
# needing to simulate real mouse events.
# ---------------------------------------------------------------------------

class AnnotationCanvas(QtWidgets.QWidget):
    boxes_changed = QtCore.Signal()
    selection_changed = QtCore.Signal()
    zoom_to_point_requested = QtCore.Signal(QtCore.QPointF)

    MIN_BOX_SIZE_PX = 6
    # How close to a corner/edge of the selected box a press counts as
    # grabbing that handle, and how big the handles are drawn. The hit area
    # is deliberately larger than the drawn square so students don't have
    # to land the cursor pixel-perfectly to resize.
    HANDLE_HIT_PX = 10
    HANDLE_DRAW_PX = 7
    # A box may never be dragged smaller than this (normalized) - a
    # zero-area box is not a usable training label.
    MIN_BOX_NORM = 0.005

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._pixmap: QtGui.QPixmap | None = None
        self._boxes: list[dict] = []
        self._class_names: list[str] = []
        self._class_colors: list[QtGui.QColor] = []
        self._selected_index: int | None = None
        self._current_class_id = 0
        # What the mouse is currently doing: None, "new" (rubber-banding a
        # fresh box on empty space), "move" (dragging an existing box) or
        # "resize" (dragging one of its handles).
        self._mode: str | None = None
        self._active_handle: str | None = None
        # The box as it was when the press landed, plus where that press
        # was: every move/resize is computed from these rather than
        # accumulated frame by frame, so a drag can never drift.
        self._orig_box: dict | None = None
        self._edit_origin: QtCore.QPointF | None = None
        self._drag_start_widget: QtCore.QPointF | None = None
        self._drag_current_widget: QtCore.QPointF | None = None

    def set_class_palette(self, names: list[str], colors: list[QtGui.QColor]) -> None:
        self._class_names = list(names)
        self._class_colors = list(colors)
        self.update()

    def set_current_class(self, class_id: int) -> None:
        self._current_class_id = class_id

    def set_frame(self, pixmap: QtGui.QPixmap | None, boxes: list[dict]) -> None:
        self._pixmap = pixmap
        self._boxes = [dict(b) for b in boxes]
        self._selected_index = None
        self._cancel_edit()
        self.update()

    def _cancel_edit(self) -> None:
        self._mode = None
        self._active_handle = None
        self._orig_box = None
        self._edit_origin = None
        self._drag_start_widget = None
        self._drag_current_widget = None

    def boxes(self) -> list[dict]:
        return [dict(b) for b in self._boxes]

    def selected_index(self) -> int | None:
        return self._selected_index

    # ---- geometry helpers (pure, unit-testable) ----
    def _image_rect(self) -> QtCore.QRectF:
        """Where the pixmap is actually drawn inside this widget
        (aspect-fit, centered - same idea as QLabel's KeepAspectRatio)."""
        if self._pixmap is None or self._pixmap.isNull():
            return QtCore.QRectF()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        if pw <= 0 or ph <= 0 or ww <= 0 or wh <= 0:
            return QtCore.QRectF()
        scale = min(ww / pw, wh / ph)
        dw, dh = pw * scale, ph * scale
        return QtCore.QRectF((ww - dw) / 2.0, (wh - dh) / 2.0, dw, dh)

    def _widget_to_norm(self, pos: QtCore.QPointF) -> tuple[float, float] | None:
        rect = self._image_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return None
        x = (pos.x() - rect.x()) / rect.width()
        y = (pos.y() - rect.y()) / rect.height()
        return (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))

    def _norm_box_to_widget_rect(self, box: dict) -> QtCore.QRectF:
        rect = self._image_rect()
        cx, cy, w, h = box["cx"], box["cy"], box["w"], box["h"]
        x1 = rect.x() + (cx - w / 2.0) * rect.width()
        y1 = rect.y() + (cy - h / 2.0) * rect.height()
        return QtCore.QRectF(x1, y1, w * rect.width(), h * rect.height())

    def _hit_test(self, pos: QtCore.QPointF) -> int | None:
        for i in range(len(self._boxes) - 1, -1, -1):
            if self._norm_box_to_widget_rect(self._boxes[i]).contains(pos):
                return i
        return None

    @staticmethod
    def _handle_points(rect: QtCore.QRectF) -> dict[str, QtCore.QPointF]:
        """The eight grab points of a box, keyed by which edges they move:
        't'/'b'/'l'/'r' for the edge midpoints and the two-letter
        combinations for the corners."""
        cx, cy = rect.center().x(), rect.center().y()
        return {
            "tl": rect.topLeft(),
            "t": QtCore.QPointF(cx, rect.top()),
            "tr": rect.topRight(),
            "l": QtCore.QPointF(rect.left(), cy),
            "r": QtCore.QPointF(rect.right(), cy),
            "bl": rect.bottomLeft(),
            "b": QtCore.QPointF(cx, rect.bottom()),
            "br": rect.bottomRight(),
        }

    def _handle_at(self, pos: QtCore.QPointF) -> str | None:
        """Which handle of the SELECTED box (if any) is under the cursor.
        Only the selected box has handles - otherwise overlapping boxes
        would fight over every click near their edges."""
        if self._selected_index is None or not (0 <= self._selected_index < len(self._boxes)):
            return None
        rect = self._norm_box_to_widget_rect(self._boxes[self._selected_index])
        for name, point in self._handle_points(rect).items():
            if (abs(point.x() - pos.x()) <= self.HANDLE_HIT_PX
                    and abs(point.y() - pos.y()) <= self.HANDLE_HIT_PX):
                return name
        return None

    @staticmethod
    def _moved_box(box: dict, dx: float, dy: float) -> dict:
        """Same box, shifted by a normalized delta and clamped so it can
        never be dragged off the edge of the photo."""
        half_w, half_h = box["w"] / 2.0, box["h"] / 2.0
        return dict(
            box,
            cx=min(1.0 - half_w, max(half_w, box["cx"] + dx)),
            cy=min(1.0 - half_h, max(half_h, box["cy"] + dy)),
        )

    @classmethod
    def _resized_box(cls, box: dict, handle: str, dx: float, dy: float) -> dict:
        """Same box with the edges named by `handle` moved by a normalized
        delta. Dragging an edge past its opposite one flips rather than
        inverting the box, which is what every drawing tool does."""
        left, right = box["cx"] - box["w"] / 2.0, box["cx"] + box["w"] / 2.0
        top, bottom = box["cy"] - box["h"] / 2.0, box["cy"] + box["h"] / 2.0
        if "l" in handle:
            left += dx
        if "r" in handle:
            right += dx
        if "t" in handle:
            top += dy
        if "b" in handle:
            bottom += dy
        left, right = sorted((left, right))
        top, bottom = sorted((top, bottom))
        left, right = min(1.0, max(0.0, left)), min(1.0, max(0.0, right))
        top, bottom = min(1.0, max(0.0, top)), min(1.0, max(0.0, bottom))
        w = max(cls.MIN_BOX_NORM, right - left)
        h = max(cls.MIN_BOX_NORM, bottom - top)
        return dict(
            box,
            cx=min(1.0 - w / 2.0, max(w / 2.0, (left + right) / 2.0)),
            cy=min(1.0 - h / 2.0, max(h / 2.0, (top + bottom) / 2.0)),
            w=w,
            h=h,
        )

    # ---- programmatic editing (used by both tests and mouse handlers) ----
    def add_box_from_widget_rect(self, p1: QtCore.QPointF, p2: QtCore.QPointF,
                                  class_id: int | None = None) -> bool:
        rect = QtCore.QRectF(p1, p2).normalized()
        if rect.width() < self.MIN_BOX_SIZE_PX or rect.height() < self.MIN_BOX_SIZE_PX:
            return False  # too small - treat as an accidental click, not a box
        n1 = self._widget_to_norm(rect.topLeft())
        n2 = self._widget_to_norm(rect.bottomRight())
        if n1 is None or n2 is None:
            return False
        x1, y1 = n1
        x2, y2 = n2
        w, h = abs(x2 - x1), abs(y2 - y1)
        if w <= 0 or h <= 0:
            return False
        cid = self._current_class_id if class_id is None else class_id
        self._boxes.append(dict(class_id=cid, cx=(x1 + x2) / 2.0, cy=(y1 + y2) / 2.0, w=w, h=h))
        self._selected_index = len(self._boxes) - 1
        self.update()
        self.boxes_changed.emit()
        return True

    def delete_selected(self) -> bool:
        if self._selected_index is None:
            return False
        del self._boxes[self._selected_index]
        self._selected_index = None
        self.update()
        self.boxes_changed.emit()
        return True

    def select_at(self, pos: QtCore.QPointF) -> int | None:
        idx = self._hit_test(pos)
        self._selected_index = idx
        self.update()
        self.selection_changed.emit()
        return idx

    def select_index(self, index: int | None) -> None:
        """Programmatic selection (used by the "Boxes on this frame" list
        so clicking an item there highlights the same box on the image)."""
        if index is not None and not (0 <= index < len(self._boxes)):
            index = None
        self._selected_index = index
        self.update()
        self.selection_changed.emit()

    def reclassify_selected(self, class_id: int) -> bool:
        """Changes the class of the currently-selected box - lets a
        student fix a mislabeled box just by picking a different class
        from the "New boxes get class" dropdown, no delete+redraw needed."""
        if self._selected_index is None:
            return False
        self._boxes[self._selected_index]["class_id"] = class_id
        self.update()
        self.boxes_changed.emit()
        return True

    # ---- drag lifecycle (pure enough to drive straight from tests) ----
    def begin_edit_at(self, pos: QtCore.QPointF) -> str | None:
        """Decides what a press at `pos` starts: grabbing a handle of the
        selected box ("resize"), grabbing the body of any box ("move"), or
        rubber-banding a brand-new box on empty space ("new")."""
        if self._pixmap is None or self._image_rect().isEmpty():
            return None

        handle = self._handle_at(pos)
        if handle is not None:
            self._mode = "resize"
            self._active_handle = handle
            self._edit_origin = pos
            self._orig_box = dict(self._boxes[self._selected_index])
            return "resize"

        index = self._hit_test(pos)
        if index is not None:
            if index != self._selected_index:
                self._selected_index = index
                self.selection_changed.emit()
            self._mode = "move"
            self._edit_origin = pos
            self._orig_box = dict(self._boxes[index])
            self.update()
            return "move"

        had_selection = self._selected_index is not None
        self._selected_index = None
        self._mode = "new"
        self._drag_start_widget = pos
        self._drag_current_widget = pos
        if had_selection:
            self.selection_changed.emit()
        self.update()
        return "new"

    def update_edit_to(self, pos: QtCore.QPointF) -> None:
        if self._mode is None:
            return
        if self._mode == "new":
            self._drag_current_widget = pos
            self.update()
            return

        rect = self._image_rect()
        if rect.isEmpty() or self._orig_box is None or self._edit_origin is None:
            return
        if self._selected_index is None or not (0 <= self._selected_index < len(self._boxes)):
            return
        dx = (pos.x() - self._edit_origin.x()) / rect.width()
        dy = (pos.y() - self._edit_origin.y()) / rect.height()
        if self._mode == "move":
            self._boxes[self._selected_index] = self._moved_box(self._orig_box, dx, dy)
        else:
            self._boxes[self._selected_index] = self._resized_box(
                self._orig_box, self._active_handle, dx, dy
            )
        self.update()

    def finish_edit(self) -> None:
        """Ends the drag. A move/resize reports boxes_changed only here, on
        release, so the label file is rewritten once per edit instead of on
        every mouse-move along the way."""
        mode = self._mode
        moved_box = None
        if mode in ("move", "resize") and self._selected_index is not None:
            moved_box = self._boxes[self._selected_index]
        start, current = self._drag_start_widget, self._drag_current_widget
        changed = mode in ("move", "resize") and moved_box != self._orig_box
        self._cancel_edit()

        if mode == "new" and start is not None and current is not None:
            self.add_box_from_widget_rect(start, current)
        elif changed:
            self.boxes_changed.emit()
        self.update()

    def _cursor_for(self, pos: QtCore.QPointF) -> QtGui.QCursor:
        """Cursor feedback is what makes "you can drag this" discoverable -
        without it a movable box looks exactly like a fixed one."""
        handle = self._handle_at(pos)
        if handle is not None:
            if handle in ("tl", "br"):
                return QtGui.QCursor(QtCore.Qt.SizeFDiagCursor)
            if handle in ("tr", "bl"):
                return QtGui.QCursor(QtCore.Qt.SizeBDiagCursor)
            if handle in ("t", "b"):
                return QtGui.QCursor(QtCore.Qt.SizeVerCursor)
            return QtGui.QCursor(QtCore.Qt.SizeHorCursor)
        if self._hit_test(pos) is not None:
            return QtGui.QCursor(QtCore.Qt.SizeAllCursor)
        return QtGui.QCursor(QtCore.Qt.CrossCursor)

    # ---- Qt event handlers (thin wrappers around the methods above) ----
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton or self._pixmap is None:
            return
        self.begin_edit_at(event.position())

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        pos = event.position()
        if self._mode is None:
            if self._pixmap is not None:
                self.setCursor(self._cursor_for(pos))
            return
        self.update_edit_to(pos)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._mode is None:
            return
        self.finish_edit()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Double-click a spot on the image to zoom in centered exactly
        there (double-click again to zoom back out) - the direct way to
        "zoom into a specific spot" instead of hunting for it after a
        generic zoom. A double-click's first press/release is still
        governed by the normal too-small-drag rejection above, so this
        never creates a stray box."""
        if event.button() == QtCore.Qt.LeftButton and self.isEnabled() and self._pixmap is not None:
            self._cancel_edit()
            self.zoom_to_point_requested.emit(event.position())
        else:
            super().mouseDoubleClickEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#111014"))
        rect = self._image_rect()
        has_pixmap = self._pixmap is not None and not self._pixmap.isNull() and rect.width() > 0

        if has_pixmap:
            painter.drawPixmap(rect.toRect(), self._pixmap)
        else:
            painter.setPen(QtGui.QColor("#5a6b7a"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "Add images in Datasets, then select an image")

        # Frames are loaded but drawing is blocked because no class exists
        # yet - dim the image and show a big, hard-to-miss instruction
        # instead of leaving the student staring at a frame that just
        # doesn't respond to clicks.
        if has_pixmap and not self.isEnabled():
            painter.fillRect(rect, QtGui.QColor(0, 0, 0, 150))
            painter.setPen(QtGui.QColor("#ffd166"))
            font = painter.font()
            font.setBold(True)
            font.setPointSize(font.pointSize() + 1)
            painter.setFont(font)
            painter.drawText(
                self.rect(), QtCore.Qt.AlignCenter,
                "Add a class in the right sidebar\nto start drawing boxes",
            )
            painter.end()
            return

        for i, box in enumerate(self._boxes):
            r = self._norm_box_to_widget_rect(box)
            color = (
                self._class_colors[box["class_id"] % len(self._class_colors)]
                if self._class_colors else QtGui.QColor("#33d17a")
            )
            pen_width = 3 if i == self._selected_index else 2
            painter.setPen(QtGui.QPen(color, pen_width))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(r)
            name = (
                self._class_names[box["class_id"]]
                if 0 <= box["class_id"] < len(self._class_names) else str(box["class_id"])
            )
            tag_rect = QtCore.QRectF(r.x(), max(0.0, r.y() - 16), max(40, len(name) * 7 + 8), 16)
            painter.fillRect(tag_rect, color)
            painter.setPen(QtGui.QColor("#0b1118"))
            painter.drawText(tag_rect.adjusted(4, 0, 0, 0), QtCore.Qt.AlignVCenter, name)

            # Handles go on the selected box only - they are both the grab
            # targets for resizing and the visual cue that this box can be
            # dragged at all.
            if i == self._selected_index:
                half = self.HANDLE_DRAW_PX / 2.0
                painter.setPen(QtGui.QPen(QtGui.QColor("#0b1118"), 1))
                painter.setBrush(color)
                for point in self._handle_points(r).values():
                    painter.drawRect(
                        QtCore.QRectF(point.x() - half, point.y() - half,
                                      self.HANDLE_DRAW_PX, self.HANDLE_DRAW_PX)
                    )
                painter.setBrush(QtCore.Qt.NoBrush)

        if self._mode == "new" and self._drag_start_widget is not None and self._drag_current_widget is not None:
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffd166"), 1, QtCore.Qt.DashLine))
            painter.drawRect(QtCore.QRectF(self._drag_start_widget, self._drag_current_widget).normalized())
        painter.end()


class _ZoomScrollArea(QtWidgets.QScrollArea):
    """A plain QScrollArea, except Ctrl+wheel over the image is treated as
    a zoom request (anchored at the cursor) instead of scrolling - this is
    the only way to zoom now (no on-screen +/-/slider buttons), so a drone
    photo full of small/aerial objects can be zoomed into right where the
    cursor already is, without extra buttons cluttering the tab.

    Wheel events over the viewport are delivered to the *viewport* child
    widget, not this outer widget, so overriding wheelEvent() here would
    never fire - viewportEvent() is the actual interception point."""

    zoom_at_requested = QtCore.Signal(int, QtCore.QPoint)  # (+1/-1 step, viewport-local pos)

    def viewportEvent(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Wheel and event.modifiers() & QtCore.Qt.ControlModifier:
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            self.zoom_at_requested.emit(1 if event.angleDelta().y() > 0 else -1, pos)
            return True
        return super().viewportEvent(event)


# ---------------------------------------------------------------------------
# The popup dialog itself: three tabs (Video -> Frames, Label Objects,
# Train & Export) sharing one session log at the bottom.
# ---------------------------------------------------------------------------

class _LabelTrainWorkflow(QtWidgets.QDialog):
    """Non-modal popup: turns a raw video into a working YOLO .pt model,
    opened from a button on the main window."""

    extract_progress_signal = QtCore.Signal(int, int)
    extract_finished_signal = QtCore.Signal(int)
    train_progress_signal = QtCore.Signal(int, int)
    train_finished_signal = QtCore.Signal(object)
    status_signal = QtCore.Signal(str)
    # (generation, frame index, QImage) - carries gallery thumbnails from
    # ThumbnailLoadThread onto the GUI thread, same Queued-Connection rule
    # as every other worker callback in this app.
    thumbnail_ready_signal = QtCore.Signal(int, int, object)
    import_finished_signal = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("AI Detection Lab — Label & Train AI"))
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.setWindowFlag(QtCore.Qt.WindowMinMaxButtonsHint, True)
        self._apply_screen_safe_size(preferred=(1360, 860), minimum=(1020, 660))
        self.setStyleSheet(DIALOG_STYLESHEET)
        self._is_maximized_view = False

        # ---- session state ----
        self.video_path: Path | None = None
        self.session_dir: Path | None = None
        self.frames_dir: Path | None = None
        self.classes: list[str] = []
        self.frame_paths: list[Path] = []
        self.current_frame_index: int = -1
        self._boxes_cache: dict[int, list[dict]] = {}
        self._extract_thread: FrameExtractThread | None = None
        self._train_thread: TrainAIThread | None = None
        self._thumb_thread: ThumbnailLoadThread | None = None
        # Bumped on every gallery rebuild so thumbnails still arriving from
        # a previous session's thread are recognised as stale and dropped
        # instead of landing on the new session's photos.
        self._thumb_generation: int = 0
        self._import_thread: ImportModelThread | None = None
        self._placeholder_pixmap: QtGui.QPixmap | None = None
        self._last_best_pt: Path | None = None
        self._zoom_factor: float = 1.0
        self._video_test_dialog: VideoClipTestDialog | None = None

        self._build_ui()
        self._wire_signals()
        self._refresh_class_ui()
        self._refresh_tab2_state()
        self._refresh_tab3_state()
        self._refresh_project_selector()
        self._refresh_project_ui()

    def _apply_screen_safe_size(
        self, preferred: tuple[int, int], minimum: tuple[int, int]
    ) -> None:
        """Sizes this dialog to fit the actual screen instead of a fixed
        desktop-monitor-sized default. A compact field/kiosk touchscreen
        (e.g. a ground-station display, confirmed from a real photo of one
        cutting off the "3 Train & Export" tab) can be smaller than this
        dialog's normal 1020x660 minimum, which pushes part of the window
        off-screen with no way to reach it - there's no OS window manager
        to drag it back into view on a kiosk-style touch screen. When the
        available screen is smaller than `minimum`, the floor is relaxed
        and the dialog opens maximized to exactly fill what's there
        instead; otherwise it behaves as before (fixed preferred size,
        never smaller than `minimum`, capped to the screen either way)."""
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        pref_w, pref_h = preferred
        min_w, min_h = minimum
        if avail is None:
            self.setMinimumSize(min_w, min_h)
            self.resize(pref_w, pref_h)
            return
        if avail.width() < min_w or avail.height() < min_h:
            self.setMinimumSize(min(320, avail.width()), min(240, avail.height()))
            self.resize(avail.width(), avail.height())
            self.setWindowState(self.windowState() | QtCore.Qt.WindowMaximized)
        else:
            self.setMinimumSize(min_w, min_h)
            self.resize(min(pref_w, avail.width()), min(pref_h, avail.height()))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(tr("🏷️ LABEL & TRAIN AI"))
        title.setObjectName("dlgTitle")
        header_row.addWidget(title)
        header_row.addStretch()

        self.fullscreen_btn = QtWidgets.QPushButton(tr("⛶  Full Screen"))
        self.fullscreen_btn.setToolTip(tr("Expand this window to fill the screen"))
        self.fullscreen_btn.clicked.connect(self._toggle_fullscreen)
        header_row.addWidget(self.fullscreen_btn, 0, QtCore.Qt.AlignTop)
        layout.addLayout(header_row)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_import_tab(), tr("1️⃣ Import Files"))
        self.tabs.addTab(self._build_label_tab(), tr("2️⃣ Label Objects"))
        # "&&" because Qt reads a lone "&" in a tab title as a keyboard
        # mnemonic and swallows it, rendering this as "Train _Export".
        self.tabs.addTab(self._build_train_tab(), tr("3️⃣ Train && Export"))
        layout.addWidget(self.tabs, 1)

        log_label = QtWidgets.QLabel(tr("SESSION LOG"))
        log_label.setObjectName("sectionTitle")
        layout.addWidget(log_label)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setObjectName("terminal")
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        layout.addWidget(self.log)

    def _toggle_fullscreen(self):
        """Same single-source-of-truth toggle as the Code Editor popup -
        deliberately does not also try to auto-detect maximize/restore
        done some other way, to avoid state-change events racing with
        this method on platforms where they arrive asynchronously."""
        if self._is_maximized_view:
            self.showNormal()
            self.fullscreen_btn.setText(tr("⛶  Full Screen"))
            self.fullscreen_btn.setToolTip(tr("Expand this window to fill the screen"))
        else:
            self.showMaximized()
            self.fullscreen_btn.setText(tr("🗗  Restore Size"))
            self.fullscreen_btn.setToolTip(tr("Restore this window to its normal size"))
        self._is_maximized_view = not self._is_maximized_view

    def _build_import_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setSpacing(10)

        project_card = QtWidgets.QFrame()
        project_card.setObjectName("card")
        project_layout = QtWidgets.QVBoxLayout(project_card)
        project_section = QtWidgets.QLabel(tr("PROJECT"))
        project_section.setObjectName("sectionTitle")
        project_layout.addWidget(project_section)

        project_row = QtWidgets.QHBoxLayout()
        project_row.addWidget(QtWidgets.QLabel(tr("Working on:")))
        self.project_combo = QtWidgets.QComboBox()
        self.project_combo.setMinimumWidth(260)
        self.project_combo.setToolTip(tr("Switch straight to another project"))
        project_row.addWidget(self.project_combo)
        self.new_project_btn = QtWidgets.QPushButton(tr("➕  New Project"))
        self.open_project_btn = QtWidgets.QPushButton(tr("📂  Browse Projects…"))
        self.open_project_btn.setToolTip(tr(
            "See every project with its photo and labeling counts before opening one"
        ))
        self.delete_project_btn = QtWidgets.QPushButton(tr("🗑  Delete Project"))
        self.delete_project_btn.setToolTip(tr(
            "Permanently delete the project currently open - its photos, labels "
            "and any models saved inside it"
        ))
        project_row.addWidget(self.new_project_btn)
        project_row.addWidget(self.open_project_btn)
        project_row.addWidget(self.delete_project_btn)
        project_row.addStretch()
        project_layout.addLayout(project_row)

        self.project_label = QtWidgets.QLabel("")
        self.project_label.setWordWrap(True)
        project_layout.addWidget(self.project_label)

        project_hint = QtWidgets.QLabel(tr(
            "A project keeps its photos, boxes and classes together. Come "
            "back any time, open it again and carry on - or add more photos "
            "to it below and keep training the same model with more data."
        ))
        project_hint.setObjectName("muted")
        project_hint.setWordWrap(True)
        project_layout.addWidget(project_hint)
        v.addWidget(project_card)

        card = QtWidgets.QFrame()
        card.setObjectName("card")
        card_layout = QtWidgets.QVBoxLayout(card)

        section = QtWidgets.QLabel(tr("STEP 1 — ADD PHOTOS TO THIS PROJECT"))
        section.setObjectName("sectionTitle")
        card_layout.addWidget(section)

        photos_hint = QtWidgets.QLabel(tr(
            "Already have photos of the object? Bring them straight in - no "
            "video needed. Photos are copied into the project, so your own "
            "folder is never written to, and they are added to whatever is "
            "already there rather than replacing it."
        ))
        photos_hint.setObjectName("muted")
        photos_hint.setWordWrap(True)
        card_layout.addWidget(photos_hint)

        photo_row = QtWidgets.QHBoxLayout()
        self.import_photos_btn = QtWidgets.QPushButton(tr("🖼  Import Photos"))
        self.import_photos_btn.setObjectName("primaryButton")
        self.import_folder_btn = QtWidgets.QPushButton(tr("📁  Import a Folder of Photos"))
        photo_row.addWidget(self.import_photos_btn)
        photo_row.addWidget(self.import_folder_btn)
        photo_row.addStretch()
        card_layout.addLayout(photo_row)

        or_label = QtWidgets.QLabel(tr("— or start from a video —"))
        or_label.setObjectName("muted")
        card_layout.addWidget(or_label)

        self.open_video_btn = QtWidgets.QPushButton(tr("📹  Import a Video (.mp4, .avi, .mov, .mkv)"))
        card_layout.addWidget(self.open_video_btn)

        self.video_path_label = QtWidgets.QLabel(tr("No video selected yet"))
        self.video_path_label.setObjectName("muted")
        self.video_path_label.setWordWrap(True)
        card_layout.addWidget(self.video_path_label)

        interval_row = QtWidgets.QHBoxLayout()
        interval_row.addWidget(QtWidgets.QLabel(tr("Grab a photo every:")))
        self.interval_spin = QtWidgets.QDoubleSpinBox()
        self.interval_spin.setLocale(EN_US_LOCALE)
        self.interval_spin.setRange(0.1, 30.0)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setDecimals(1)
        self.interval_spin.setValue(DEFAULT_FRAME_INTERVAL_SEC)
        self.interval_spin.setSuffix(" sec")
        interval_row.addWidget(self.interval_spin)
        interval_row.addWidget(QtWidgets.QLabel(tr("(smaller = more photos, more labeling work)")))
        interval_row.addStretch()
        card_layout.addLayout(interval_row)

        self.extract_btn = QtWidgets.QPushButton(tr("🎬  Cut the Video into Photos"))
        self.extract_btn.setEnabled(False)
        card_layout.addWidget(self.extract_btn)

        self.extract_progress = QtWidgets.QProgressBar()
        self.extract_progress.setRange(0, 100)
        self.extract_progress.setVisible(False)
        card_layout.addWidget(self.extract_progress)

        self.frame_count_label = QtWidgets.QLabel(f"{tr('Photos')}: 0")
        card_layout.addWidget(self.frame_count_label)

        variety_tip = QtWidgets.QLabel(tr(
            "🎥 Tip for a model that actually works later: don't capture the "
            "object from just ONE fixed angle/distance (e.g. only straight "
            "down from far away). Walk around it and get closer/farther "
            "shots too - including an angle similar to how you'll hold it "
            "up to a laptop webcam to test it. A model trained on only one "
            "camera angle will usually fail completely on any other angle."
        ))
        variety_tip.setObjectName("muted")
        variety_tip.setWordWrap(True)
        card_layout.addWidget(variety_tip)
        v.addWidget(card)

        model_card = QtWidgets.QFrame()
        model_card.setObjectName("card")
        model_layout = QtWidgets.QVBoxLayout(model_card)
        model_section = QtWidgets.QLabel(tr("OR IMPORT A MODEL SOMEONE ALREADY TRAINED"))
        model_section.setObjectName("sectionTitle")
        model_layout.addWidget(model_section)

        import_hint = QtWidgets.QLabel(tr(
            "Trained a YOLO model outside this app (Google Colab, Roboflow, "
            "another computer)? Pick its .pt file and it gets copied into "
            "this project's models folder, ready for the AI camera - no "
            "photos, no labeling and no training needed."
        ))
        import_hint.setObjectName("muted")
        import_hint.setWordWrap(True)
        model_layout.addWidget(import_hint)

        self.import_model_btn = QtWidgets.QPushButton(tr("📥  Import Model (.pt)"))
        model_layout.addWidget(self.import_model_btn)

        self.import_status_label = QtWidgets.QLabel("")
        self.import_status_label.setWordWrap(True)
        model_layout.addWidget(self.import_status_label)
        v.addWidget(model_card)

        v.addStretch()

        # Scrolled: this tab carries three cards plus their explanations, so
        # on a short window (or once the project header grows) the bottom
        # ones would otherwise be silently clipped mid-sentence.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        return scroll

    def _build_label_tab(self) -> QtWidgets.QWidget:
        """Two views, one tab: a photo gallery to pick from, and the box
        editor for whichever photo was picked (self.label_stack). The
        classes panel is deliberately outside that stack so the class you
        are about to draw with stays visible in both views."""
        page = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(page)
        h.setSpacing(10)

        h.addWidget(self._build_class_card())

        self.label_stack = QtWidgets.QStackedWidget()
        self.label_stack.addWidget(self._build_gallery_page())
        self.label_stack.addWidget(self._build_editor_page())
        h.addWidget(self.label_stack, 1)

        return page

    def _build_class_card(self) -> QtWidgets.QWidget:
        class_card = QtWidgets.QFrame()
        class_card.setObjectName("card")
        class_card.setMaximumWidth(240)
        cv = QtWidgets.QVBoxLayout(class_card)
        section = QtWidgets.QLabel(tr("CLASSES"))
        section.setObjectName("sectionTitle")
        cv.addWidget(section)

        add_row = QtWidgets.QHBoxLayout()
        self.class_name_edit = QtWidgets.QLineEdit()
        self.class_name_edit.setPlaceholderText(tr("e.g. cone, person..."))
        add_row.addWidget(self.class_name_edit)
        self.add_class_btn = QtWidgets.QPushButton(tr("+ Add"))
        add_row.addWidget(self.add_class_btn)
        cv.addLayout(add_row)

        self.class_list = QtWidgets.QListWidget()
        cv.addWidget(self.class_list, 1)

        self.delete_class_btn = QtWidgets.QPushButton(tr("🗑  Delete Selected Class"))
        self.delete_class_btn.setToolTip(tr(
            "Removes this class and every box already labeled with it, "
            "across every photo in this project"
        ))
        self.delete_class_btn.setEnabled(False)
        cv.addWidget(self.delete_class_btn)

        hint = QtWidgets.QLabel(tr(
            "Name every object you want the AI to find, then click one "
            "here to pick what new boxes get labeled as. Click a class "
            "while a box is selected to relabel that box."
        ))
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        cv.addWidget(hint)
        return class_card

    def _build_gallery_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        section = QtWidgets.QLabel(tr("STEP 2 — CLICK A PHOTO TO LABEL IT"))
        section.setObjectName("sectionTitle")
        v.addWidget(section)

        hint = QtWidgets.QLabel(tr(
            "Every photo pulled out of your video is below. Click one to "
            "open it and drag boxes around the objects.   "
            "✅ = already has boxes   •   🚫 = checked: no object   •   ⬜ = still empty"
        ))
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.frame_list = QtWidgets.QListWidget()
        self.frame_list.setViewMode(QtWidgets.QListWidget.IconMode)
        self.frame_list.setIconSize(QtCore.QSize(GALLERY_THUMB_PX, GALLERY_THUMB_H))
        self.frame_list.setGridSize(QtCore.QSize(GALLERY_THUMB_PX + 26, GALLERY_THUMB_H + 46))
        self.frame_list.setResizeMode(QtWidgets.QListWidget.Adjust)
        self.frame_list.setMovement(QtWidgets.QListWidget.Static)
        self.frame_list.setUniformItemSizes(True)
        self.frame_list.setSpacing(6)
        self.frame_list.setCursor(QtCore.Qt.PointingHandCursor)
        v.addWidget(self.frame_list, 1)

        bottom = QtWidgets.QHBoxLayout()
        self.gallery_progress_label = QtWidgets.QLabel(tr("No photos yet"))
        bottom.addWidget(self.gallery_progress_label)
        bottom.addStretch()
        self.mark_rest_no_object_btn = QtWidgets.QPushButton(tr("🚫  Mark all ⬜ photos as 'no object'"))
        self.mark_rest_no_object_btn.setToolTip(tr(
            "For a folder of photos that do NOT contain the object (empty floor, other things). "
            "Photos without the object teach the AI what is NOT the object - fewer false alarms."
        ))
        self.mark_rest_no_object_btn.clicked.connect(self._mark_rest_no_object)
        bottom.addWidget(self.mark_rest_no_object_btn)
        self.open_editor_btn = QtWidgets.QPushButton(tr("✏️  Open This Photo"))
        self.open_editor_btn.setObjectName("primaryButton")
        self.open_editor_btn.setEnabled(False)
        bottom.addWidget(self.open_editor_btn)
        v.addLayout(bottom)
        return page

    def _build_editor_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        self.back_to_gallery_btn = QtWidgets.QPushButton(tr("◀  All Photos"))
        self.back_to_gallery_btn.setToolTip(tr("Back to the photo gallery"))
        top.addWidget(self.back_to_gallery_btn)
        section = QtWidgets.QLabel(tr("DRAG A BOX AROUND EACH OBJECT"))
        section.setObjectName("sectionTitle")
        top.addWidget(section)
        top.addStretch()
        self.frame_nav_label = QtWidgets.QLabel(tr("Frame -- / --"))
        top.addWidget(self.frame_nav_label)
        v.addLayout(top)

        middle = QtWidgets.QHBoxLayout()
        middle.setSpacing(10)

        center = QtWidgets.QVBoxLayout()
        self.canvas = AnnotationCanvas()
        self.canvas_scroll = _ZoomScrollArea()
        self.canvas_scroll.setWidget(self.canvas)
        self.canvas_scroll.setWidgetResizable(True)  # fit-to-window at 100% zoom
        self.canvas_scroll.setMinimumHeight(240)
        center.addWidget(self.canvas_scroll, 1)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setToolTip(tr("Slide to jump to any photo"))
        center.addWidget(self.frame_slider)

        self.current_class_label = QtWidgets.QLabel("")
        self.current_class_label.setObjectName("muted")
        center.addWidget(self.current_class_label)

        help_label = QtWidgets.QLabel(tr(
            "Drag on an empty part of the photo to draw a box. Click a box "
            "to select it, then drag the middle to move it or drag a corner "
            "handle to resize it - press Delete to remove it. Hold Ctrl and "
            "scroll to zoom into a small object (or double-click a spot to "
            "zoom there). Boxes save by themselves - just move on to the "
            "next photo."
        ))
        help_label.setObjectName("muted")
        help_label.setWordWrap(True)
        center.addWidget(help_label)
        middle.addLayout(center, 1)

        boxes_card = QtWidgets.QFrame()
        boxes_card.setObjectName("card")
        boxes_card.setMaximumWidth(230)
        bv = QtWidgets.QVBoxLayout(boxes_card)
        section3 = QtWidgets.QLabel(tr("BOXES ON THIS PHOTO"))
        section3.setObjectName("sectionTitle")
        bv.addWidget(section3)
        self.box_list = QtWidgets.QListWidget()
        bv.addWidget(self.box_list, 1)
        self.delete_box_btn = QtWidgets.QPushButton(tr("🗑  Delete Selected Box"))
        self.delete_box_btn.setEnabled(False)
        bv.addWidget(self.delete_box_btn)
        self.no_object_btn = QtWidgets.QPushButton(tr("🚫  No object in this photo"))
        self.no_object_btn.setCheckable(True)
        self.no_object_btn.setToolTip(tr(
            "Tick when this photo has NONE of your objects in it. Such photos teach the AI "
            "what is NOT the object, so it stops 'seeing' it everywhere."
        ))
        self.no_object_btn.toggled.connect(self._toggle_no_object)
        bv.addWidget(self.no_object_btn)
        middle.addWidget(boxes_card)

        v.addLayout(middle, 1)

        nav_row = QtWidgets.QHBoxLayout()
        self.prev_frame_btn = QtWidgets.QPushButton(tr("◀  Previous Photo"))
        self.prev_frame_btn.setEnabled(False)
        self.next_frame_btn = QtWidgets.QPushButton(tr("Next Photo  ▶"))
        self.next_frame_btn.setObjectName("primaryButton")
        self.next_frame_btn.setEnabled(False)
        nav_row.addWidget(self.prev_frame_btn)
        nav_row.addStretch()
        nav_row.addWidget(self.next_frame_btn)
        v.addLayout(nav_row)
        return page

    def _build_train_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setSpacing(10)

        card = QtWidgets.QFrame()
        card.setObjectName("card")
        card_layout = QtWidgets.QVBoxLayout(card)

        section = QtWidgets.QLabel(tr("STEP 3 — TRAIN A YOLOV8 MODEL"))
        section.setObjectName("sectionTitle")
        card_layout.addWidget(section)

        self.train_summary_label = QtWidgets.QLabel(
            f"{tr('Frames')}: 0   •   {tr('Labeled')}: 0   •   {tr('Classes')}: {tr('(none yet)')}"
        )
        self.train_summary_label.setWordWrap(True)
        card_layout.addWidget(self.train_summary_label)

        epochs_row = QtWidgets.QHBoxLayout()
        epochs_row.addWidget(QtWidgets.QLabel(tr("Epochs:")))
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setLocale(EN_US_LOCALE)
        self.epochs_spin.setRange(1, 300)
        self.epochs_spin.setValue(DEFAULT_EPOCHS)
        epochs_row.addWidget(self.epochs_spin)
        epochs_row.addWidget(QtWidgets.QLabel(
            f"{tr('Base model')}: {BASE_MODEL_NAME} ({tr('fast/light preset - downloads automatically the first time if not already cached')})"
        ))
        min_frames_tip = QtWidgets.QLabel(
            f"⚠️ {MIN_LABELED_FRAMES} " + tr(
                "labeled frames is just the minimum to "
                "unlock this button, not a good target - a model trained on "
                "that few frames from one video angle will very likely fail on "
                "a webcam test or a different flight. Label frames from as "
                "many different angles/distances as you extracted for the "
                "best real-world results."
            )
        )
        min_frames_tip.setWordWrap(True)
        min_frames_tip.setObjectName("muted")
        card_layout.addWidget(min_frames_tip)
        epochs_row.addStretch()
        card_layout.addLayout(epochs_row)

        self.train_readiness_label = QtWidgets.QLabel("")
        self.train_readiness_label.setWordWrap(True)
        self.train_readiness_label.setObjectName("muted")
        card_layout.addWidget(self.train_readiness_label)

        self.train_btn = QtWidgets.QPushButton(tr("🚀  Start Training"))
        self.train_btn.setObjectName("primaryButton")
        self.train_btn.setEnabled(False)
        card_layout.addWidget(self.train_btn)

        self.train_progress = QtWidgets.QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setVisible(False)
        card_layout.addWidget(self.train_progress)
        self.train_progress_label = QtWidgets.QLabel("")
        self.train_progress_label.setObjectName("muted")
        card_layout.addWidget(self.train_progress_label)

        card_layout.addWidget(QtWidgets.QFrame())  # spacer line

        export_section = QtWidgets.QLabel(tr("SAVED MODEL"))
        export_section.setObjectName("sectionTitle")
        card_layout.addWidget(export_section)

        export_hint = QtWidgets.QLabel(tr(
            "As soon as training finishes, the best model is saved into "
            "this project's models folder automatically - named after "
            "what you labeled. No extra step needed."
        ))
        export_hint.setObjectName("muted")
        export_hint.setWordWrap(True)
        card_layout.addWidget(export_hint)

        self.save_status_label = QtWidgets.QLabel("")
        self.save_status_label.setWordWrap(True)
        card_layout.addWidget(self.save_status_label)

        test_hint = QtWidgets.QLabel(tr(
            "Want to see it in action right away? Import a different video "
            "clip below and watch every model in this project's models "
            "folder label it live - no webcam or drone needed."
        ))
        test_hint.setObjectName("muted")
        test_hint.setWordWrap(True)
        card_layout.addWidget(test_hint)

        self.test_clip_btn = QtWidgets.QPushButton(tr("🎬  Test with a Video Clip"))
        self.test_clip_btn.setObjectName("primaryButton")
        card_layout.addWidget(self.test_clip_btn)

        card_layout.addStretch()
        v.addWidget(card)
        v.addStretch()
        return page

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------
    def _wire_signals(self):
        self.new_project_btn.clicked.connect(self._new_project)
        self.open_project_btn.clicked.connect(self._open_project)
        self.delete_project_btn.clicked.connect(self._delete_project)
        self.project_combo.currentIndexChanged.connect(self._on_project_selected)
        self.import_photos_btn.clicked.connect(self._import_photo_files)
        self.import_folder_btn.clicked.connect(self._import_photo_folder)
        self.open_video_btn.clicked.connect(self._open_video)
        self.extract_btn.clicked.connect(self._start_extraction)
        self.extract_progress_signal.connect(self._on_extract_progress)
        self.extract_finished_signal.connect(self._on_extract_finished)

        self.add_class_btn.clicked.connect(self._add_class)
        self.class_name_edit.returnPressed.connect(self._add_class)
        self.class_list.currentRowChanged.connect(self._on_class_list_row_changed)
        self.delete_class_btn.clicked.connect(self._delete_class)
        self.canvas.boxes_changed.connect(self._on_canvas_boxes_changed)
        self.canvas.selection_changed.connect(self._on_canvas_selection_changed)

        self.box_list.currentRowChanged.connect(self._on_box_list_row_changed)
        self.delete_box_btn.clicked.connect(self._delete_selected_box)
        self.frame_list.currentRowChanged.connect(self._on_frame_list_row_changed)
        # Clicking a photo in the gallery is the main way into the editor.
        # itemActivated covers Enter/Return for keyboard users; both land on
        # the same idempotent handler, so a double-click firing each once is
        # harmless.
        self.frame_list.itemClicked.connect(self._on_gallery_item_chosen)
        self.frame_list.itemActivated.connect(self._on_gallery_item_chosen)
        self.open_editor_btn.clicked.connect(self._open_editor_for_current_frame)
        self.back_to_gallery_btn.clicked.connect(self._show_gallery)
        self.thumbnail_ready_signal.connect(self._on_thumbnail_ready)

        self.import_model_btn.clicked.connect(self._import_model)
        self.import_finished_signal.connect(self._on_import_finished)

        self.canvas_scroll.zoom_at_requested.connect(self._on_wheel_zoom)
        self.canvas.zoom_to_point_requested.connect(self._on_canvas_double_click)

        self.prev_frame_btn.clicked.connect(self._prev_frame)
        self.next_frame_btn.clicked.connect(self._next_frame)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)

        self.train_btn.clicked.connect(self._start_training)
        self.train_progress_signal.connect(self._on_train_progress)
        self.train_finished_signal.connect(self._on_train_finished)
        self.test_clip_btn.clicked.connect(self._open_video_test_dialog)

        self.status_signal.connect(self._on_status)

        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, activated=self._next_frame)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self, activated=self._prev_frame)

    # ------------------------------------------------------------------
    # Tab 1 - importing photos, video or a finished model
    # ------------------------------------------------------------------
    def _new_project(self):
        name, ok = QtWidgets.QInputDialog.getText(
            self, tr("New Project"), tr("Name this project (e.g. cones_field_test):")
        )
        if not ok or not name.strip():
            return
        project_dir = project_dir_for_name(name)
        if project_dir.exists():
            reply = QtWidgets.QMessageBox.question(
                self, tr("Project Already Exists"),
                f"“{project_dir.name}” {tr('already exists.')}\n\n" + tr(
                    "Open it and carry on with it instead?"
                ),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            self._activate_project(project_dir)
            return
        try:
            (project_dir / "frames").mkdir(parents=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, tr("Could Not Create Project"), str(exc))
            return
        self._log(f"🗂 {tr('New project')}: {project_dir.name}")
        self._activate_project(project_dir)

    def _open_project(self):
        picker = ProjectPickerDialog(self)
        if picker.exec() != QtWidgets.QDialog.Accepted or picker.selected is None:
            return
        self._activate_project(picker.selected)

    def _delete_project(self):
        """Permanently removes the project currently open - its whole folder
        on disk (photos, label .txt files, classes.txt, any models trained
        into it). Switches to another project if one exists, otherwise
        leaves the dialog in the same "no project open yet" state it starts
        up in."""
        if self.session_dir is None:
            return
        if self._train_thread is not None or self._extract_thread is not None:
            QtWidgets.QMessageBox.information(
                self, tr("Delete Project"),
                tr("Wait until training / importing has finished before deleting a project."),
            )
            return
        project_dir = self.session_dir
        name = project_dir.name
        reply = QtWidgets.QMessageBox.question(
            self, tr("Delete Project"),
            f"{tr('Permanently delete project')} “{name}”?\n\n"
            f"{tr('This removes all')} {len(self.frame_paths)} {tr('photo(s), their labels, and any models trained into it. This cannot be undone.')}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        try:
            shutil.rmtree(project_dir)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, tr("Could Not Delete Project"), str(exc))
            return
        self._log(f"🗑️ {tr('Deleted project')}: {name}")

        remaining = list_projects()
        if remaining:
            self._activate_project(remaining[0])
            return

        self.session_dir = None
        self.frames_dir = None
        self.classes = []
        self.frame_paths = []
        self._boxes_cache = {}
        self.current_frame_index = -1
        self._last_best_pt = None
        self.video_path = None
        self.video_path_label.setText(tr("No video selected yet"))
        self.frame_count_label.setText(f"{tr('Photos')}: 0")
        self._refresh_class_ui()
        self._refresh_frame_list()
        self._load_current_frame_into_canvas()
        self._show_gallery()
        self._refresh_tab2_state()
        self._refresh_tab3_state()
        self._refresh_project_selector()
        self._refresh_project_ui()

    def _activate_project(self, project_dir: Path) -> None:
        """Makes `project_dir` the one being worked on, restoring everything
        it already holds - photos, their boxes and its classes."""
        if self._extract_thread is not None:
            # Photos/frames are still being written into the CURRENT
            # project's folder - switching now would mix them up.
            QtWidgets.QMessageBox.information(
                self, tr("Please wait"),
                tr("Wait until the photo import / video cutting has finished before switching project."),
            )
            self._sync_project_selector()
            return
        self.session_dir = project_dir
        self.frames_dir = project_dir / "frames"
        self.classes = read_classes(project_dir / "classes.txt")
        self.frame_paths = sorted(self.frames_dir.glob("*.jpg")) if self.frames_dir.is_dir() else []
        self._boxes_cache = {}
        self.current_frame_index = 0 if self.frame_paths else -1
        self._last_best_pt = None
        self.video_path = None
        self.video_path_label.setText(tr("No video selected yet"))
        self.frame_count_label.setText(f"{tr('Photos')}: {len(self.frame_paths)}")
        self._log(
            f"📂 {tr('Project')} “{project_dir.name}”: {len(self.frame_paths)} {tr('photo(s)')}, "
            f"{tr('classes')}: {', '.join(self.classes) if self.classes else tr('(none yet)')}"
        )

        self._refresh_class_ui()
        self._refresh_frame_list()
        self._load_current_frame_into_canvas()
        self._show_gallery()
        self._refresh_tab2_state()
        self._refresh_tab3_state()
        self._refresh_project_selector()
        self._refresh_project_ui()

    def _refresh_project_selector(self) -> None:
        """Repopulates the project dropdown. Lists folder names only - no
        photo counting - so switching projects stays instant no matter how
        many photos are stored across all of them; the Browse dialog is
        where the per-project counts get gathered."""
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        projects = list_projects()
        if projects:
            self.project_combo.setEnabled(True)
            for project in projects:
                self.project_combo.addItem(project.name, str(project))
        else:
            self.project_combo.setEnabled(False)
            self.project_combo.addItem(tr("(no projects yet)"), None)
        self._sync_project_selector()
        self.project_combo.blockSignals(False)

    def _sync_project_selector(self) -> None:
        if self.session_dir is None:
            return
        index = self.project_combo.findData(str(self.session_dir))
        if index >= 0:
            self.project_combo.setCurrentIndex(index)

    def _on_project_selected(self, index: int) -> None:
        if index < 0:
            return
        data = self.project_combo.itemData(index)
        if not data:
            return
        chosen = Path(data)
        if self.session_dir is not None and chosen == self.session_dir:
            return
        self._activate_project(chosen)

    def _refresh_project_ui(self) -> None:
        if self.session_dir is None:
            self.project_label.setText(tr(
                "⚠️ No project open yet — press “➕ New Project”, or just import "
                "photos below and one gets created for you."
            ))
            return
        labeled = sum(
            1 for p in self.frame_paths
            if label_path_for(p).exists() and label_path_for(p).stat().st_size > 0
        )
        classes = ", ".join(self.classes) if self.classes else tr("no classes yet")
        self.project_label.setText(
            f"📁 <b>{self.session_dir.name}</b> — {len(self.frame_paths)} {tr('photos')}, "
            f"{labeled} {tr('labeled')}, {tr('classes')}: {classes}"
        )

    def _ensure_project(self, source_path: Path) -> bool:
        """Importing without having made a project first just makes one,
        named after what is being imported - the shortest path from "I have
        photos" to labeling them stays one click."""
        if self.session_dir is not None:
            return True
        project_dir = new_session_dir(source_path)
        try:
            (project_dir / "frames").mkdir(parents=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, tr("Could Not Create Project"), str(exc))
            return False
        self._log(f"🗂 {tr('New project')}: {project_dir.name}")
        self._activate_project(project_dir)
        return True

    def _set_import_controls_enabled(self, enabled: bool) -> None:
        self.import_photos_btn.setEnabled(enabled)
        self.import_folder_btn.setEnabled(enabled)
        self.open_video_btn.setEnabled(enabled)
        self.extract_btn.setEnabled(enabled and self.video_path is not None)

    def _import_photo_files(self):
        patterns = " ".join(f"*{suffix}" for suffix in IMAGE_IMPORT_SUFFIXES)
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, tr("Import Photos"), "", f"{tr('Photos')} ({patterns});;{tr('All Files')} (*)",
        )
        if not paths:
            return
        sources = [Path(p) for p in paths]
        self._start_photo_import(sources, sources[0])

    def _import_photo_folder(self):
        start_dir = str(config.DRONE_PHOTOS_DIR) if config.DRONE_PHOTOS_DIR.is_dir() else ""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, tr("Import a Folder of Photos"), start_dir
        )
        if not folder:
            return
        folder_path = Path(folder)
        sources = sorted(
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_IMPORT_SUFFIXES
        )
        if not sources:
            QtWidgets.QMessageBox.warning(
                self, tr("No Photos Found"),
                f"{folder_path.name} " + tr("has no photo files in it")
                + f" ({', '.join(IMAGE_IMPORT_SUFFIXES)}).",
            )
            return
        self._start_photo_import(sources, folder_path)

    def _start_photo_import(self, sources: list[Path], session_source: Path):
        if self._extract_thread is not None:
            return
        if not self._ensure_project(session_source):
            return
        self._log(f"🖼 {tr('Adding')} {len(sources)} {tr('photo(s) to')} “{self.session_dir.name}”")

        self._set_import_controls_enabled(False)
        self.extract_progress.setValue(0)
        self.extract_progress.setVisible(True)
        # Reuses the extraction signals/handler: once photos are in the
        # project's frames folder, "imported photos" and "frames cut from a
        # video" are the same thing to every step after this.
        self._extract_thread = ImportImagesThread(
            sources=sources,
            out_dir=self.frames_dir,
            start_index=next_frame_index(self.frames_dir),
            on_progress=lambda cur, total: self.extract_progress_signal.emit(cur, total),
            on_status=lambda msg: self.status_signal.emit(msg),
            on_error=lambda msg: self.status_signal.emit(msg),
            on_finished=lambda n: self.extract_finished_signal.emit(n),
        )
        self._extract_thread.start()

    def _open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("Import a Video"), "",
            f"{tr('Video Files')} (*.mp4 *.avi *.mov *.mkv);;{tr('All Files')} (*)",
        )
        if not path:
            return
        video_path = Path(path)
        if not self._ensure_project(video_path):
            return
        self.video_path = video_path
        self.video_path_label.setText(f"📹 {self.video_path.name}")
        self.extract_btn.setEnabled(True)
        self._log(f"📂 {tr('Opened video')}: {self.video_path.name}")

    def _start_extraction(self):
        if self.video_path is None or self._extract_thread is not None:
            return
        interval = self.interval_spin.value()
        self._set_import_controls_enabled(False)
        self.extract_progress.setValue(0)
        self.extract_progress.setVisible(True)
        self._extract_thread = FrameExtractThread(
            video_path=self.video_path,
            out_dir=self.frames_dir,
            start_index=next_frame_index(self.frames_dir),
            interval_sec=interval,
            on_progress=lambda cur, total: self.extract_progress_signal.emit(cur, total),
            on_status=lambda msg: self.status_signal.emit(msg),
            on_error=lambda msg: self.status_signal.emit(msg),
            on_finished=lambda n: self.extract_finished_signal.emit(n),
        )
        self._extract_thread.start()

    @QtCore.Slot(int, int)
    def _on_extract_progress(self, cur, total):
        if total > 0:
            self.extract_progress.setValue(min(100, int(cur * 100 / total)))

    @QtCore.Slot(int)
    def _on_extract_finished(self, n):
        self._extract_thread = None
        self._set_import_controls_enabled(True)
        self.extract_progress.setVisible(False)
        if n > 0 and self.frames_dir is not None:
            already_had = len(self.frame_paths)
            self.frame_paths = sorted(self.frames_dir.glob("*.jpg"))
            self.frame_count_label.setText(f"{tr('Photos')}: {len(self.frame_paths)}")
            self._boxes_cache = {}
            # Land on the first of the photos just added, so topping up a
            # half-labeled project doesn't dump the student back at photo 1
            # with no idea which ones are new.
            self.current_frame_index = min(already_had, len(self.frame_paths) - 1)
            self._refresh_frame_list()
            self._load_current_frame_into_canvas()
            # Jump straight to the labeling tab, showing the photo gallery
            # rather than one lone frame - seeing every photo at once is
            # what makes the next step ("pick one and box it") obvious.
            self.tabs.setCurrentIndex(1)
            self._show_gallery()
            if not self.classes:
                # Naming the object is what has to happen before any box can
                # be drawn - but only for a project that has no classes yet.
                self.class_name_edit.setFocus()
                self.class_name_edit.selectAll()
        self._refresh_tab2_state()
        self._refresh_tab3_state()

    # ------------------------------------------------------------------
    # Tab 2 - classes + labeling
    # ------------------------------------------------------------------
    def _add_class(self):
        """Wired to the "+ Add" button and the class-name text field's
        Enter key - type an object's name and press Enter (or click Add)
        to create it, then click it in the list to draw with it."""
        name = self.class_name_edit.text().strip()
        if not name:
            return
        if self._add_class_by_name(name):
            self.class_name_edit.clear()

    def _add_class_by_name(self, name: str) -> bool:
        """Adds `name` as a new class (persisting it and refreshing every
        widget that lists classes), or - if it already exists - just
        selects it. Returns True only when a *new* class was created."""
        name = name.strip()
        if not name:
            return False
        if name in self.classes:
            self._log(f"ℹ️ {tr('Class')} '{name}' {tr('already exists - selected it')}")
            self._select_class_by_name(name)
            return False
        self.classes.append(name)
        if self.session_dir is not None:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            write_classes(self.session_dir / "classes.txt", self.classes)
        self._log(f"🏷️ {tr('Added class')}: {name}")
        self._refresh_class_ui()
        self._select_class_by_name(name)
        return True

    def _select_class_by_name(self, name: str) -> None:
        if name not in self.classes:
            return
        idx = self.classes.index(name)
        self.class_list.setCurrentRow(idx)  # triggers _on_class_list_row_changed

    def _delete_class(self):
        """Removes the selected class everywhere it's used - not just from
        the list. Every box already labeled with it, on every photo in the
        project (not only the ones opened so far in this session), is
        deleted too, and every class after it shifts its id down by one so
        the remaining classes.txt / label files stay in sync (YOLO stores a
        class as a bare index, not by name)."""
        row = self.class_list.currentRow()
        if not (0 <= row < len(self.classes)):
            return
        name = self.classes[row]

        affected = 0
        for i, path in enumerate(self.frame_paths):
            boxes = self._boxes_cache.get(i)
            if boxes is None:
                boxes = read_yolo_labels(label_path_for(path))
            affected += sum(1 for b in boxes if b["class_id"] == row)

        detail = (
            f"{tr('This deletes')} {affected} " + tr("box(es) already labeled")
            + f" “{name}” " + tr("across every photo in this project.") + "\n\n"
            if affected else ""
        )
        reply = QtWidgets.QMessageBox.question(
            self, tr("Delete Class"),
            f"{tr('Permanently delete class')} “{name}”?\n\n{detail}{tr('This cannot be undone.')}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        for i, path in enumerate(self.frame_paths):
            boxes = self._boxes_cache.get(i)
            if boxes is None:
                boxes = read_yolo_labels(label_path_for(path))
            new_boxes = []
            changed = False
            for box in boxes:
                cid = box["class_id"]
                if cid == row:
                    changed = True
                    continue
                if cid > row:
                    box = dict(box, class_id=cid - 1)
                    changed = True
                new_boxes.append(box)
            self._boxes_cache[i] = new_boxes
            if changed:
                write_yolo_labels(label_path_for(path), new_boxes)

        del self.classes[row]
        if self.session_dir is not None:
            write_classes(self.session_dir / "classes.txt", self.classes)

        self._log(
            f"🗑️ {tr('Deleted class')} “{name}”"
            + (f" ({affected} {tr('box(es) removed')})" if affected else "")
        )
        self._refresh_class_ui()
        self._load_current_frame_into_canvas()
        self._refresh_frame_list()
        self._refresh_project_ui()

    def _refresh_class_ui(self):
        prev_index = self.class_list.currentRow()
        colors = class_colors()
        self.class_list.blockSignals(True)
        self.class_list.clear()
        for i, name in enumerate(self.classes):
            item = QtWidgets.QListWidgetItem(f"●  {name}")
            item.setForeground(colors[i % len(colors)])
            self.class_list.addItem(item)
        if self.classes:
            self.class_list.setCurrentRow(max(0, min(prev_index, len(self.classes) - 1)))
        self.class_list.blockSignals(False)

        self.canvas.set_class_palette(self.classes, colors)
        self.canvas.set_current_class(max(0, self.class_list.currentRow()))
        self.canvas.setEnabled(bool(self.classes))
        self.delete_class_btn.setEnabled(self.class_list.currentRow() >= 0)
        self._update_current_class_label()
        self._refresh_tab3_summary()

    def _on_class_list_row_changed(self, row: int):
        self.delete_class_btn.setEnabled(row >= 0)
        if row < 0:
            self._update_current_class_label()
            return
        self.canvas.set_current_class(row)
        # If a box is currently selected, treat clicking a class here as
        # "relabel this box" - lets a mislabeled box be fixed with one
        # click instead of delete + redraw.
        if self.canvas.selected_index() is not None:
            self.canvas.reclassify_selected(row)
        self._update_current_class_label()

    def _update_current_class_label(self):
        row = self.class_list.currentRow()
        if 0 <= row < len(self.classes):
            self.current_class_label.setText(f"🖊  {tr('New boxes will be labeled')}: {self.classes[row]}")
        else:
            self.current_class_label.setText(tr(
                "🖊  Add a class on the left, then click it here before drawing."
            ))

    def _refresh_tab2_state(self):
        # Gated on the session, not on a video: a session started by
        # importing photos has no video at all.
        self.tabs.setTabEnabled(1, self.session_dir is not None)

    def _load_current_frame_into_canvas(self):
        if not (self.frame_paths and 0 <= self.current_frame_index < len(self.frame_paths)):
            self.canvas.set_frame(None, [])
            self.frame_nav_label.setText(tr("Frame -- / --"))
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, 0)
            self.frame_slider.blockSignals(False)
            self._refresh_box_list()
            return
        path = self.frame_paths[self.current_frame_index]
        boxes = self._boxes_cache.get(self.current_frame_index)
        if boxes is None:
            boxes = read_yolo_labels(label_path_for(path))
            self._boxes_cache[self.current_frame_index] = boxes
        pixmap = QtGui.QPixmap(str(path))
        self.canvas.set_frame(pixmap, boxes)
        self.no_object_btn.blockSignals(True)
        self.no_object_btn.setChecked(is_no_object(path) and not boxes)
        self.no_object_btn.blockSignals(False)
        self._set_zoom(1.0)  # each new frame starts from the full view - zoom into it fresh
        self._update_frame_nav_ui()
        self._refresh_box_list()
        self.frame_list.blockSignals(True)
        if 0 <= self.current_frame_index < self.frame_list.count():
            self.frame_list.setCurrentRow(self.current_frame_index)
        self.frame_list.blockSignals(False)

    def _update_frame_nav_ui(self):
        total = len(self.frame_paths)
        idx = self.current_frame_index
        labeled = sum(
            1 for p in self.frame_paths
            if label_path_for(p).exists() and label_path_for(p).stat().st_size > 0
        )
        self.frame_nav_label.setText(f"{tr('Photo')} {idx + 1 if total else 0} / {total}   •   {tr('labeled')} {labeled}")
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, total - 1))
        self.frame_slider.setValue(max(0, idx))
        self.frame_slider.blockSignals(False)
        self.prev_frame_btn.setEnabled(idx > 0)
        self.next_frame_btn.setEnabled(0 <= idx < total - 1)
        self.open_editor_btn.setEnabled(total > 0)
        self.gallery_progress_label.setText(
            f"✅ {tr('Labeled')} {labeled} {tr('of')} {total} {tr('photos')}"
            if total else
            tr("No photos yet — extract frames in step 1 first.")
        )

    def _go_to_frame(self, index: int):
        if not (0 <= index < len(self.frame_paths)):
            return
        self.current_frame_index = index
        self._load_current_frame_into_canvas()

    def _next_frame(self):
        self._go_to_frame(self.current_frame_index + 1)

    def _prev_frame(self):
        self._go_to_frame(self.current_frame_index - 1)

    def _on_slider_changed(self, value: int):
        if value != self.current_frame_index:
            self._go_to_frame(value)

    def _on_wheel_zoom(self, step: int, viewport_pos: QtCore.QPoint):
        """Ctrl+wheel: zoom in/out anchored exactly at the cursor, like an
        image editor or map - the point under the cursor stays under the
        cursor instead of the view jumping to some corner. This is the
        only zoom control now (no on-screen buttons/slider)."""
        norm_anchor = self._canvas_point_to_norm(self._viewport_pos_to_canvas_pos(viewport_pos))
        new_factor = max(1.0, min(4.0, self._zoom_factor + step * 0.25))
        self._set_zoom(new_factor, norm_anchor=norm_anchor, viewport_anchor=viewport_pos)

    def _on_canvas_double_click(self, canvas_point: QtCore.QPointF):
        """Double-click a spot on the image: zoom in centered right there
        if not already zoomed in, or reset to Fit if already zoomed - a
        one-click way to "zoom into a specific spot" without scrolling
        first."""
        norm_anchor = self._canvas_point_to_norm(canvas_point)
        if self._zoom_factor > 1.05:
            self._set_zoom(1.0)
        elif norm_anchor is not None:
            self._set_zoom(2.5, norm_anchor=norm_anchor)

    def _viewport_pos_to_canvas_pos(self, viewport_pos: QtCore.QPoint) -> QtCore.QPointF:
        scroll = self.canvas_scroll
        return QtCore.QPointF(
            scroll.horizontalScrollBar().value() + viewport_pos.x(),
            scroll.verticalScrollBar().value() + viewport_pos.y(),
        )

    def _canvas_point_to_norm(self, canvas_point: QtCore.QPointF) -> tuple[float, float] | None:
        return self.canvas._widget_to_norm(canvas_point)

    def _viewport_center_norm_point(self) -> tuple[float, float] | None:
        """The normalized image coordinate currently centered in the
        visible viewport - used as the default zoom anchor so a plain
        slider drag or +/- click keeps looking at the same spot instead
        of jumping back to a corner."""
        vp_size = self.canvas_scroll.viewport().size()
        center = QtCore.QPoint(vp_size.width() // 2, vp_size.height() // 2)
        return self._canvas_point_to_norm(self._viewport_pos_to_canvas_pos(center))

    def _set_zoom(self, factor: float, norm_anchor: tuple[float, float] | None = None,
                   viewport_anchor: QtCore.QPoint | None = None):
        """factor == 1.0 means "fit the whole frame in the view" (the
        scroll area auto-sizes the canvas to match, exactly like before
        zoom existed). Anything above that gives the canvas a bigger
        fixed size than the viewport, so the same aspect-fit drawing code
        paints the image larger and the scroll area grows scrollbars to
        pan around it - no change needed to how boxes are drawn or
        hit-tested, they're computed from the canvas's own size either
        way.

        `norm_anchor` is the normalized (0-1, 0-1) image point that
        should stay visually anchored at `viewport_anchor` (defaults to
        the viewport's center) after the resize - this is what makes
        zooming land on the object being labeled instead of some corner.
        If not given, it defaults to whatever's currently centered."""
        factor = max(1.0, min(4.0, factor))
        if norm_anchor is None and factor > 1.0:
            norm_anchor = self._viewport_center_norm_point()

        self._zoom_factor = factor
        if factor <= 1.0:
            self.canvas_scroll.setWidgetResizable(True)
            self.canvas.setMinimumSize(320, 240)
            self.canvas.setMaximumSize(16777215, 16777215)
        else:
            self.canvas_scroll.setWidgetResizable(False)
            viewport = self.canvas_scroll.viewport().size()
            base_w = max(viewport.width(), 320)
            base_h = max(viewport.height(), 240)
            self.canvas.setFixedSize(int(base_w * factor), int(base_h * factor))
        self.canvas.update()

        if norm_anchor is not None and factor > 1.0:
            nx, ny = norm_anchor
            img_rect = self.canvas._image_rect()
            target_x = img_rect.x() + nx * img_rect.width()
            target_y = img_rect.y() + ny * img_rect.height()
            if viewport_anchor is None:
                vp_size = self.canvas_scroll.viewport().size()
                viewport_anchor = QtCore.QPoint(vp_size.width() // 2, vp_size.height() // 2)
            self.canvas_scroll.horizontalScrollBar().setValue(int(target_x - viewport_anchor.x()))
            self.canvas_scroll.verticalScrollBar().setValue(int(target_y - viewport_anchor.y()))

    def _on_canvas_boxes_changed(self):
        if not (self.frame_paths and 0 <= self.current_frame_index < len(self.frame_paths)):
            return
        boxes = self.canvas.boxes()
        self._boxes_cache[self.current_frame_index] = boxes
        path = self.frame_paths[self.current_frame_index]
        write_yolo_labels(label_path_for(path), boxes)
        if boxes and is_no_object(path):
            # Drew a box -> the photo does have an object after all.
            set_no_object(path, False)
            self.no_object_btn.blockSignals(True)
            self.no_object_btn.setChecked(False)
            self.no_object_btn.blockSignals(False)
        self._update_frame_nav_ui()
        self._refresh_tab3_summary()
        self._refresh_box_list()
        self._update_frame_list_row(self.current_frame_index)

    def _toggle_no_object(self, checked: bool):
        if not (self.frame_paths and 0 <= self.current_frame_index < len(self.frame_paths)):
            return
        idx = self.current_frame_index
        path = self.frame_paths[idx]
        if checked:
            boxes = self._boxes_cache.get(idx) or read_yolo_labels(label_path_for(path))
            if boxes:
                reply = QtWidgets.QMessageBox.question(
                    self, tr("No object in this photo"),
                    tr("This photo has boxes. Remove them and mark it as 'no object'?"),
                )
                if reply != QtWidgets.QMessageBox.Yes:
                    self.no_object_btn.blockSignals(True)
                    self.no_object_btn.setChecked(False)
                    self.no_object_btn.blockSignals(False)
                    return
                write_yolo_labels(label_path_for(path), [])
                self._boxes_cache[idx] = []
                self._load_current_frame_into_canvas()
        set_no_object(path, checked)
        self._update_frame_list_row(idx)
        self._refresh_tab3_summary()

    def _mark_rest_no_object(self):
        if not self.frame_paths:
            return
        todo = [p for p in self.frame_paths if not _is_labeled(p) and not is_no_object(p)]
        if not todo:
            QtWidgets.QMessageBox.information(self, tr("No object in this photo"), tr("There are no ⬜ photos left."))
            return
        reply = QtWidgets.QMessageBox.question(
            self, tr("No object in this photo"),
            f"{tr('Mark')} {len(todo)} {tr('photo(s) as')} 🚫 'no object'?\n\n"
            + tr("Only do this if NONE of these photos contain the object. A photo that does contain it "
                 "but is marked 'no object' teaches the AI the wrong thing."),
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        for p in todo:
            set_no_object(p, True)
        self._refresh_frame_list()
        self._refresh_tab3_summary()
        self._log(f"🚫 {tr('Marked')} {len(todo)} {tr('photo(s) as no object')}")

    def _refresh_box_list(self):
        """Rebuilds the "Boxes on this frame" list on the right so every
        labeled object on the current frame is visible at a glance,
        instead of only as small rectangles drawn on the image."""
        sel = self.canvas.selected_index()
        colors = class_colors()
        self.box_list.blockSignals(True)
        self.box_list.clear()
        for i, box in enumerate(self.canvas.boxes()):
            cid = box.get("class_id", 0)
            name = self.classes[cid] if 0 <= cid < len(self.classes) else f"{tr('class')} {cid}"
            item = QtWidgets.QListWidgetItem(f"{i + 1}.  {name}")
            item.setForeground(colors[cid % len(colors)] if colors else QtGui.QColor("#33d17a"))
            self.box_list.addItem(item)
        self.box_list.setCurrentRow(sel if sel is not None else -1)
        self.box_list.blockSignals(False)
        self.delete_box_btn.setEnabled(sel is not None)

    def _on_box_list_row_changed(self, row: int):
        self.canvas.select_index(row if row >= 0 else None)

    def _on_canvas_selection_changed(self):
        sel = self.canvas.selected_index()
        self.box_list.blockSignals(True)
        self.box_list.setCurrentRow(sel if sel is not None else -1)
        self.box_list.blockSignals(False)
        self.delete_box_btn.setEnabled(sel is not None)

    def _delete_selected_box(self):
        self.canvas.delete_selected()

    def _refresh_frame_list(self):
        """Full rebuild of the photo gallery - called once after extraction
        so the whole video's labeling progress is visible up front. The
        thumbnails themselves stream in afterwards from
        ThumbnailLoadThread; each item starts on a blank placeholder so the
        grid never reflows as they arrive."""
        self._stop_thumbnail_load()
        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        placeholder = QtGui.QIcon(self._placeholder_thumb())
        for i, p in enumerate(self.frame_paths):
            item = QtWidgets.QListWidgetItem(placeholder, self._frame_item_text(i, p))
            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
            item.setToolTip(f"{tr('Click to label photo')} {i + 1}")
            self.frame_list.addItem(item)
        if 0 <= self.current_frame_index < self.frame_list.count():
            self.frame_list.setCurrentRow(self.current_frame_index)
        self.frame_list.blockSignals(False)
        self._update_frame_nav_ui()
        if self.frame_paths:
            self._thumb_generation += 1
            self._thumb_thread = ThumbnailLoadThread(
                paths=list(self.frame_paths),
                generation=self._thumb_generation,
                on_thumbnail=lambda gen, idx, img: self.thumbnail_ready_signal.emit(gen, idx, img),
            )
            self._thumb_thread.start()

    def _placeholder_thumb(self) -> QtGui.QPixmap:
        if self._placeholder_pixmap is None:
            pixmap = QtGui.QPixmap(GALLERY_THUMB_PX, GALLERY_THUMB_H)
            pixmap.fill(QtGui.QColor("#2a2f3a"))
            self._placeholder_pixmap = pixmap
        return self._placeholder_pixmap

    def _stop_thumbnail_load(self):
        if self._thumb_thread is not None:
            self._thumb_thread.stop()
            self._thumb_thread = None

    @staticmethod
    def _frame_item_text(index: int, path: Path) -> str:
        labeled = label_path_for(path).exists() and label_path_for(path).stat().st_size > 0
        icon = "✅" if labeled else ("🚫" if is_no_object(path) else "⬜")
        return f"{icon}  {tr('Photo')} {index + 1}"

    @QtCore.Slot(int, int, object)
    def _on_thumbnail_ready(self, generation: int, index: int, image: QtGui.QImage):
        if generation != self._thumb_generation:
            return
        if not (0 <= index < self.frame_list.count()):
            return
        item = self.frame_list.item(index)
        if item is not None:
            item.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(image)))

    def _update_frame_list_row(self, index: int):
        """Incremental update (called after any box edit) so the gallery's
        labeled/unlabeled marker stays current without a full rebuild
        throwing away every thumbnail already loaded."""
        if not (0 <= index < self.frame_list.count() and index < len(self.frame_paths)):
            return
        item = self.frame_list.item(index)
        if item is not None:
            item.setText(self._frame_item_text(index, self.frame_paths[index]))

    def _on_frame_list_row_changed(self, row: int):
        if row >= 0 and row != self.current_frame_index:
            self._go_to_frame(row)

    def _on_gallery_item_chosen(self, item: QtWidgets.QListWidgetItem):
        self._open_editor_for_row(self.frame_list.row(item))

    def _open_editor_for_current_frame(self):
        self._open_editor_for_row(self.frame_list.currentRow())

    def _open_editor_for_row(self, row: int):
        if not (0 <= row < len(self.frame_paths)):
            return
        self._go_to_frame(row)
        self.label_stack.setCurrentIndex(1)
        self.canvas.setFocus()

    def _show_gallery(self):
        self.label_stack.setCurrentIndex(0)
        if 0 <= self.current_frame_index < self.frame_list.count():
            self.frame_list.scrollToItem(self.frame_list.item(self.current_frame_index))
        self.frame_list.setFocus()

    # ------------------------------------------------------------------
    # Tab 3 - train & export
    # ------------------------------------------------------------------
    def _refresh_tab3_state(self):
        # Deliberately always enabled, unlike the labeling tab: importing a
        # model trained elsewhere lives here and must not require opening a
        # video first. Training itself stays gated by _refresh_tab3_summary.
        self.tabs.setTabEnabled(2, True)
        self._refresh_tab3_summary()

    def _refresh_tab3_summary(self):
        total = len(self.frame_paths)
        labeled = 0
        if total:
            labeled = sum(
                1 for p in self.frame_paths
                if label_path_for(p).exists() and label_path_for(p).stat().st_size > 0
            )
        no_obj = sum(1 for p in self.frame_paths if is_no_object(p) and not _is_labeled(p)) if total else 0
        self.train_summary_label.setText(
            f"{tr('Frames')}: {total}   •   {tr('Labeled')}: {labeled}   •   "
            f"🚫 {tr('No object')}: {no_obj}   •   "
            f"{tr('Classes')}: {', '.join(self.classes) if self.classes else tr('(none yet)')}"
        )
        can_train = (
            labeled >= MIN_LABELED_FRAMES
            and bool(self.classes)
            and self._train_thread is None
        )
        self.train_btn.setEnabled(can_train)

        # Spell out exactly why the button is (not) clickable - "just
        # disabled with no explanation" is how a real class count of 3/4
        # gets mistaken for a broken button.
        if self._train_thread is not None:
            self.train_readiness_label.setText(tr("⏳ Training is already running..."))
        elif self.session_dir is None:
            self.train_readiness_label.setText(tr(
                "⚠️ Start Training is disabled: import some photos and label "
                "them first (steps 1 and 2). Already have a model trained "
                "somewhere else? Import it in step 1 instead - no photos, "
                "no training needed."
            ))
        elif not self.classes:
            self.train_readiness_label.setText(tr(
                "⚠️ Start Training is disabled: add at least one class in the "
                "Label Objects tab first."
            ))
        elif labeled < MIN_LABELED_FRAMES:
            remaining = MIN_LABELED_FRAMES - labeled
            self.train_readiness_label.setText(
                f"⚠️ {tr('Start Training is disabled: label')} {remaining} {tr('more frame')}"
                f"{'s' if remaining != 1 else ''} {tr('in the Label Objects tab')} "
                f"({labeled}/{MIN_LABELED_FRAMES} {tr('labeled so far')})."
            )
        elif no_obj == 0:
            self.train_readiness_label.setText(tr(
                "✅ Ready to train.  💡 Tip: add some photos WITHOUT the object and mark them 🚫 "
                "— without them the AI tends to 'see' the object everywhere."
            ))
        else:
            self.train_readiness_label.setText(tr("✅ Ready to train."))

        self._refresh_project_ui()

    def _start_training(self):
        if self._train_thread is not None or self.session_dir is None or self.frames_dir is None:
            return
        labeled = sum(
            1 for p in self.frame_paths
            if label_path_for(p).exists() and label_path_for(p).stat().st_size > 0
        )
        if labeled < MIN_LABELED_FRAMES or not self.classes:
            QtWidgets.QMessageBox.warning(
                self, tr("Not enough labeled frames"),
                f"{tr('Label at least')} {MIN_LABELED_FRAMES} " + tr(
                    "frames (with at least one class defined) before starting training."
                ),
            )
            return

        labeled_paths = [
            p for p in self.frame_paths
            if label_path_for(p).exists() and label_path_for(p).stat().st_size > 0
        ]
        diversity = _estimate_frame_diversity(labeled_paths)
        if diversity is not None and diversity < MIN_FRAME_DIVERSITY:
            proceed = QtWidgets.QMessageBox.warning(
                self, tr("Training frames look very similar to each other"),
                tr(
                    "These labeled frames all look almost identical to each other"
                ) + f" (diversity score {diversity:.3f}, below {MIN_FRAME_DIVERSITY:.2f}) - "
                + tr(
                    "they're probably consecutive frames from one short, mostly-static "
                    "video clip."
                ) + "\n\n" + tr(
                    "A model trained this way tends to memorize the background and "
                    "lighting of that one clip instead of learning what the object "
                    "actually looks like - it can score well here in validation "
                    "(the val frames come from the same clip) and still fire false "
                    "positives everywhere once it sees real, different-looking footage "
                    "(e.g. an actual outdoor flight)."
                ) + "\n\n" + tr(
                    "For a model that generalizes, label frames from several different "
                    "clips: different backgrounds, distances, lighting, and - ideally - "
                    "conditions similar to where it will actually fly."
                ) + "\n\n" + tr("Train anyway with what you have?"),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if proceed != QtWidgets.QMessageBox.Yes:
                return

        epochs = self.epochs_spin.value()
        dataset_dir = self.session_dir / "dataset"
        # Remember WHICH project/classes this run is for. Training keeps
        # going if the window is closed or another project is opened, and
        # the finished model must be named/saved for the project it was
        # actually trained on - not whatever is open when it finishes.
        self._training_context = (self.session_dir, list(self.classes))
        runs_dir = self.session_dir / "runs"

        self.train_btn.setEnabled(False)
        self.train_progress.setValue(0)
        self.train_progress.setVisible(True)
        self.train_progress_label.setText(tr("Preparing dataset..."))
        self.save_status_label.setText("")

        self._train_thread = TrainAIThread(
            frames_dir=self.frames_dir,
            classes=list(self.classes),
            dataset_dir=dataset_dir,
            runs_dir=runs_dir,
            run_name="train",
            epochs=epochs,
            on_progress=lambda cur, total: self.train_progress_signal.emit(cur, total),
            on_status=lambda msg: self.status_signal.emit(msg),
            on_error=lambda msg: self.status_signal.emit(msg),
            on_finished=lambda p, c: self.train_finished_signal.emit((p, c)),
            device=self._train_device_choice(),
        )
        self._train_thread.start()

    def _train_device_choice(self) -> str:
        """The Train page's device menu (Studio), else config.TRAIN_DEVICE."""
        combo = getattr(self, "train_device_combo", None)
        if combo is not None and combo.currentData():
            return combo.currentData()
        return getattr(config, "TRAIN_DEVICE", "auto")

    @QtCore.Slot(int, int)
    def _on_train_progress(self, epoch, total):
        if total > 0:
            self.train_progress.setValue(min(100, int(epoch * 100 / total)))
            self.train_progress_label.setText(f"{tr('Epoch')} {epoch} / {total}")

    @QtCore.Slot(object)
    def _on_train_finished(self, payload):
        best_path, calibration = payload if isinstance(payload, tuple) else (payload, None)
        self._train_thread = None
        self.train_progress.setVisible(False)
        self._refresh_tab3_summary()
        if best_path is not None:
            self._last_best_pt = Path(best_path)
            # Save it into the project automatically - no extra click
            # or filename needed.
            self._save_model_to_project(calibration)
            reply = QtWidgets.QMessageBox.question(
                self, tr("Training Complete"),
                f"🎉 {tr('Training complete!')}\n\n" + self.save_status_label.text()
                + "\n\n" + tr("Test it now with a video clip?"),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes,
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self._open_video_test_dialog()
        else:
            QtWidgets.QMessageBox.warning(
                self, tr("Training Failed"),
                tr(
                    "Training did not finish successfully - see the Session "
                    "Log below for what went wrong."
                ),
            )

    def _save_model_to_project(self, calibration: dict | None = None):
        if self._last_best_pt is None or not self._last_best_pt.exists():
            self._log(f"❌ {tr('No trained model available to save')}")
            self.save_status_label.setText(tr("❌ No trained model available to save yet."))
            return
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        ctx = getattr(self, "_training_context", None)
        trained_dir, trained_classes = ctx if ctx else (self.session_dir, list(self.classes))
        base_name = suggest_model_name(
            trained_classes, trained_dir.name if trained_dir else "custom_model"
        )
        filename = unique_model_filename(config.MODEL_DIR, base_name)
        dest = config.MODEL_DIR / filename
        shutil.copy2(self._last_best_pt, dest)

        if calibration is not None:
            self._write_confidence_registry_entry(filename, calibration)
            status_text = (
                f"✅ {tr('Saved as')} {dest.name} " + tr(
                    "in this project's models folder - ready to use on the next AI "
                    "detection run. Confidence threshold auto-calibrated to"
                ) + f" {calibration['conf_thres']:.2f} " + tr("for this model (global default is")
                + f" {config.YOLO_CONF_THRES:.2f})."
            )
        else:
            status_text = (
                f"✅ {tr('Saved as')} {dest.name} " + tr(
                    "in this project's models folder - ready to use on the next AI "
                    "detection run. Using the global confidence default"
                ) + f" ({config.YOLO_CONF_THRES:.2f}) - " + tr(
                    "there wasn't enough validation data from this session to "
                    "calibrate one specifically for this model."
                )
            )
        self.save_status_label.setText(status_text)
        saved_prefix = tr("Model saved automatically to this project's models folder as")
        self._log(
            f"✅ {saved_prefix} {dest.name} - " + tr("ready to use on the next AI detection run.")
        )

    def _open_video_test_dialog(self):
        """Opens the "test a just-trained model on a video clip" popup -
        created once and reused (shown via .show(), never destroyed) so
        reopening it after testing a few clips doesn't pile up windows or
        background threads."""
        if self._video_test_dialog is None:
            self._video_test_dialog = VideoClipTestDialog(self)
        self._video_test_dialog.show()
        self._video_test_dialog.raise_()
        self._video_test_dialog.activateWindow()

    def _import_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("Import a trained YOLO model"), "",
            f"{tr('YOLO model')} (*.pt);;{tr('All Files')} (*)",
        )
        if not path or self._import_thread is not None:
            return
        src = Path(path)
        self.import_model_btn.setEnabled(False)
        self.import_model_btn.setText(tr("⏳  Checking model..."))
        self.import_status_label.setText(f"⏳ {tr('Opening')} {src.name} {tr('to check it is a detection model...')}")
        self._log(f"📥 {tr('Importing model')}: {src}")
        self._import_thread = ImportModelThread(
            src=src,
            on_finished=lambda payload: self.import_finished_signal.emit(payload),
        )
        self._import_thread.start()

    @QtCore.Slot(object)
    def _on_import_finished(self, payload: dict):
        self._import_thread = None
        self.import_model_btn.setEnabled(True)
        self.import_model_btn.setText(tr("📥  Import Model (.pt)"))

        if not payload.get("ok"):
            error = payload.get("error", tr("Unknown error"))
            self.import_status_label.setText(f"❌ {error}")
            self._log(f"❌ {tr('Model import failed')}: {error}")
            QtWidgets.QMessageBox.warning(self, tr("Could Not Import Model"), error)
            return

        dest = payload["dest"]
        classes = payload.get("classes") or []
        class_text = ", ".join(str(c) for c in classes[:6]) or tr("(no class names in the file)")
        if len(classes) > 6:
            class_text += f", +{len(classes) - 6} {tr('more')}"
        self.import_status_label.setText(
            f"✅ {tr('Imported as')} {dest.name} — {tr('it detects')}: {class_text}. " + tr(
                "Ready to use on the next AI detection run, at the global confidence "
                "default"
            ) + f" ({config.YOLO_CONF_THRES:.2f}) " + tr(
                "since a model trained elsewhere has no calibration measured in this app."
            )
        )
        self._log(f"✅ {tr('Model imported into the models folder as')} {dest.name} ({tr('classes')}: {class_text})")
        now_in_models = tr("is now in this project's models folder.")
        QtWidgets.QMessageBox.information(
            self, tr("Model Imported"),
            f"{dest.name} {now_in_models}\n\n"
            f"{tr('It detects')}: {class_text}\n\n"
            + tr("Start the AI camera or the webcam test to use it - no restart needed."),
        )

    def _write_confidence_registry_entry(self, filename: str, calibration: dict) -> None:
        """Merges this model's calibration into models/model_confidence.json
        rather than overwriting the file, so every previously trained
        model's calibrated threshold survives training a new one."""
        registry_path = config.MODEL_DIR / MODEL_CONF_REGISTRY_FILENAME
        registry: dict = {}
        try:
            if registry_path.exists():
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
                if not isinstance(registry, dict):
                    registry = {}
        except Exception as exc:
            self._log(f"⚠️ {tr('Could not read existing')} {MODEL_CONF_REGISTRY_FILENAME}, {tr('starting fresh')}: {exc}")
            registry = {}

        registry[filename] = calibration
        try:
            registry_path.write_text(
                json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            self._log(f"⚠️ {tr('Could not save calibrated confidence for')} {filename}: {exc}")

    # ------------------------------------------------------------------
    def _log(self, msg: str):
        self.status_signal.emit(msg)

    @QtCore.Slot(str)
    def _on_status(self, msg: str):
        self.log.appendPlainText(msg)

    def closeEvent(self, event):
        # Best-effort: ask any running background thread to wind down.
        # The dialog itself is kept alive by the main window (re-shown via
        # .show(), never destroyed), so a late signal emission from either
        # thread after this still lands on a live, just-hidden widget.
        if self._extract_thread is not None:
            self._extract_thread.stop()
        # Training is deliberately NOT stopped here. Stopping it used to end
        # training after the current epoch and then save that barely-trained
        # model as if it had finished normally - a student who closed the
        # window to check something got a model that detects nothing. The
        # dialog is only hidden (MainWindow keeps it), so training carries
        # on and the "Training complete" message still appears at the end.
        # Quitting the whole app still stops it (MainWindow.closeEvent).
        if self._video_test_dialog is not None:
            self._video_test_dialog.close()
        self._stop_thumbnail_load()
        event.accept()


from .label_studio import StudioLayout


class LabelTrainDialog(StudioLayout, _LabelTrainWorkflow):
    """Version 5.1 Studio UI with the original local training workflow."""
    pass
