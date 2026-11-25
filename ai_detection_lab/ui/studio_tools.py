"""Tools for the Label & Train Studio.

Dataset charts, split statistics, Predict and PyTorch (.pt) export.
(v6.0.0: SAM, auto-label and the Ultralytics Platform upload removed.) Heavy
AI runtimes are imported lazily inside worker threads so opening the Studio
stays fast.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from .pixel_theme import draw_pixel_rect

from .. import config

# A frozen app cannot pip-install; never let ultralytics try at export time.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

# (label, file suffix, format id) - PyTorch weights only.
EXPORT_FORMATS = [
    ("PyTorch (.pt)", ".pt", "pt"),
]


class BackgroundTask(QtCore.QObject):
    """Run one callable off the GUI thread and report back on it."""
    finished = QtCore.Signal(object, object)  # result, exception
    progress = QtCore.Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            result = self._fn(self.progress.emit)
        except Exception as exc:  # reported to the user by the owner
            self.finished.emit(None, exc)
        else:
            self.finished.emit(result, None)


# ---------------------------------------------------------------------------
# Model runtime (cached per weights file; ultralytics models are not
# re-entrant, so every call holds the lock)
# ---------------------------------------------------------------------------
_model_lock = threading.Lock()
_models: dict[tuple[str, float], object] = {}


def _cached(kind: str, path: Path):
    key = (kind + ":" + str(path), path.stat().st_mtime)
    if key not in _models:
        from ultralytics import YOLO
        for old in [k for k in _models if k[0] == key[0]]:
            del _models[old]
        _models[key] = YOLO(str(path))
    return _models[key]


def predict_image(model_path: Path, image_path: Path, conf: float = 0.25, iou: float = 0.7) -> dict:
    with _model_lock:
        model = _cached("yolo", model_path)
        result = model.predict(str(image_path), conf=conf, iou=iou, verbose=False)[0]
    names = result.names
    detections = []
    for xyxy, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
        detections.append(dict(label=str(names[int(cls)]), conf=float(score), box=[float(v) for v in xyxy]))
    height, width = result.orig_shape
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return dict(detections=detections, width=width, height=height, speed=dict(result.speed))


def _unique_destination(dest: Path) -> Path:
    if not dest.exists():
        return dest
    for i in range(2, 1000):
        candidate = dest.with_name(f"{dest.stem}_{i}{dest.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(dest)


def export_model(model_path: Path, fmt: str, dest_dir: Path, report=lambda _msg: None) -> Path:
    """Copy the weights to dest_dir as PyTorch .pt (the only export format)."""
    if fmt != "pt":
        raise ValueError(f"Only PyTorch (.pt) export is supported, not {fmt!r}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_destination(dest_dir / model_path.name)
    shutil.copy2(model_path, dest)
    return dest


# ---------------------------------------------------------------------------
# Dataset statistics (pure; drives the Charts card and split view)
# ---------------------------------------------------------------------------
def dataset_statistics(images: list[Path], classes: list[str], membership: dict) -> dict:
    from .label_train_dialog import label_path_for, read_yolo_labels
    instances = [0] * len(classes)
    centers, sizes = [], []
    split_counts = {"train": 0, "val": 0, "test": 0, "unused": 0}
    for p in images:
        split_counts[membership.get(p, "unused")] += 1
        for b in read_yolo_labels(label_path_for(p)):
            if 0 <= b["class_id"] < len(classes):
                instances[b["class_id"]] += 1
            centers.append((b["cx"], b["cy"], b["class_id"]))
            sizes.append((b["w"], b["h"], b["class_id"]))
    return dict(instances=instances, classes=list(classes), centers=centers,
                sizes=sizes, splits=split_counts, boxes=len(centers))


class DatasetCharts(QtWidgets.QWidget):
    """Four Ultralytics-style dataset charts drawn from the real labels:
    class instances, split sizes, box locations and box sizes."""
    GRID = 16

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(430)
        self.stats = None
        self.colors: list[QtGui.QColor] = []

    def set_stats(self, stats: dict, colors: list[QtGui.QColor]):
        self.stats, self.colors = stats, colors
        self.update()

    def _color(self, cid: int) -> QtGui.QColor:
        return self.colors[cid % len(self.colors)] if self.colors else QtGui.QColor("#c084fc")

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        gap = 14
        cw = (self.width() - gap) / 2
        ch = (self.height() - gap) / 2
        panels = [QtCore.QRectF(0, 0, cw, ch), QtCore.QRectF(cw + gap, 0, cw, ch),
                  QtCore.QRectF(0, ch + gap, cw, ch), QtCore.QRectF(cw + gap, ch + gap, cw, ch)]
        titles = ["Class instances", "Split", "Box locations", "Box sizes (w × h)"]
        for rect, title in zip(panels, titles):
            p.setPen(QtGui.QPen(QtGui.QColor("#30303c")))
            p.setBrush(QtGui.QColor("#16161c"))
            draw_pixel_rect(p, rect, 8)
            p.setPen(QtGui.QColor("#cfc8dc"))
            f = p.font(); f.setPointSize(11); f.setBold(True); p.setFont(f)
            p.drawText(rect.adjusted(12, 8, -12, 0), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, title)
            f.setBold(False); f.setPointSize(10); p.setFont(f)
        if not self.stats or not self.stats["boxes"] and not sum(self.stats["splits"].values()):
            for rect in panels:
                p.setPen(QtGui.QColor("#8d8698"))
                p.drawText(rect, QtCore.Qt.AlignCenter, "No data yet")
            return
        inner = [r.adjusted(14, 34, -14, -14) for r in panels]
        self._bars(p, inner[0])
        self._splits(p, inner[1])
        self._heatmap(p, inner[2])
        self._scatter(p, inner[3])

    def _bars(self, p, r):
        names, counts = self.stats["classes"], self.stats["instances"]
        if not names:
            p.setPen(QtGui.QColor("#8d8698")); p.drawText(r, QtCore.Qt.AlignCenter, "No classes"); return
        top = max(1, max(counts))
        label_w = min(120, r.width() * .38)
        row_h = min(26, r.height() / len(names))
        for i, (name, n) in enumerate(zip(names, counts)):
            y = r.top() + i * row_h
            if y + row_h > r.bottom() + 1:
                break
            p.setPen(QtGui.QColor("#b9b1c6"))
            p.drawText(QtCore.QRectF(r.left(), y, label_w - 6, row_h), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft,
                       p.fontMetrics().elidedText(name, QtCore.Qt.ElideRight, int(label_w - 6)))
            bar_w = (r.width() - label_w - 40) * n / top
            p.setPen(QtCore.Qt.NoPen); p.setBrush(self._color(i))
            draw_pixel_rect(p, QtCore.QRectF(r.left() + label_w, y + row_h * .2, max(2, bar_w), row_h * .6), 3)
            p.setPen(QtGui.QColor("#e8e2f2"))
            p.drawText(QtCore.QRectF(r.left() + label_w + bar_w + 6, y, 40, row_h), QtCore.Qt.AlignVCenter, str(n))

    def _splits(self, p, r):
        s = self.stats["splits"]
        order = [("train", "#a855f7"), ("val", "#38bdf8"), ("test", "#f59e0b"), ("unused", "#4b4658")]
        total = max(1, sum(s.values()))
        bar = QtCore.QRectF(r.left(), r.top() + 8, r.width(), 22)
        x = bar.left()
        for key, color in order:
            w = bar.width() * s[key] / total
            if w > 0:
                p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(color))
                p.drawRect(QtCore.QRectF(x, bar.top(), w, bar.height()))
                x += w
        y = bar.bottom() + 18
        for key, color in order:
            p.setBrush(QtGui.QColor(color)); p.setPen(QtCore.Qt.NoPen)
            draw_pixel_rect(p, QtCore.QRectF(r.left(), y + 4, 10, 10), 2)
            p.setPen(QtGui.QColor("#d8d1e4"))
            label = {"unused": "Not used"}.get(key, key.capitalize())
            p.drawText(QtCore.QRectF(r.left() + 18, y, r.width() - 18, 18), QtCore.Qt.AlignVCenter,
                       f"{label}   {s[key]} images  ·  {100 * s[key] / total:.0f}%")
            y += 24

    def _heatmap(self, p, r):
        side = min(r.width(), r.height())
        area = QtCore.QRectF(r.center().x() - side / 2, r.top(), side, side)
        n = self.GRID
        grid = [[0] * n for _ in range(n)]
        for cx, cy, _ in self.stats["centers"]:
            grid[min(n - 1, max(0, int(cy * n)))][min(n - 1, max(0, int(cx * n)))] += 1
        top = max(1, max(max(row) for row in grid))
        cell = side / n
        for gy in range(n):
            for gx in range(n):
                t = grid[gy][gx] / top
                color = QtGui.QColor.fromRgbF(.09 + .57 * t, .09 + .24 * t, .12 + .79 * t)
                p.fillRect(QtCore.QRectF(area.left() + gx * cell, area.top() + gy * cell, cell + .5, cell + .5), color)
        p.setPen(QtGui.QColor("#3a3546")); p.setBrush(QtCore.Qt.NoBrush); p.drawRect(area)

    def _scatter(self, p, r):
        side = min(r.width(), r.height())
        area = QtCore.QRectF(r.center().x() - side / 2, r.top(), side, side)
        p.setPen(QtGui.QColor("#3a3546")); p.setBrush(QtCore.Qt.NoBrush); p.drawRect(area)
        for frac in (.25, .5, .75):
            p.drawLine(QtCore.QPointF(area.left() + side * frac, area.top()), QtCore.QPointF(area.left() + side * frac, area.bottom()))
            p.drawLine(QtCore.QPointF(area.left(), area.top() + side * frac), QtCore.QPointF(area.right(), area.top() + side * frac))
        p.setPen(QtCore.Qt.NoPen)
        for w, h, cid in self.stats["sizes"]:
            c = QtGui.QColor(self._color(cid)); c.setAlphaF(.75); p.setBrush(c)
            p.drawEllipse(QtCore.QPointF(area.left() + w * side, area.bottom() - h * side), 3.2, 3.2)
        p.setPen(QtGui.QColor("#8d8698"))
        p.drawText(QtCore.QRectF(area.left(), area.bottom() - 16, side - 4, 14), QtCore.Qt.AlignRight, "width →")
        p.drawText(QtCore.QRectF(area.left() + 4, area.top() + 2, side, 14), QtCore.Qt.AlignLeft, "↑ height")


class PredictionView(QtWidgets.QWidget):
    """Image with predicted boxes, scaled to fit (Predict tab preview).
    Accepts a dropped image file, like the Platform's Predict tab."""
    image_dropped = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(380)
        self.setAcceptDrops(True)
        self.pixmap: QtGui.QPixmap | None = None
        self.detections: list[dict] = []
        self.placeholder = "Drop an image here, or choose one, to run the selected model"

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.image_dropped.emit(Path(event.mimeData().urls()[0].toLocalFile()))
        event.acceptProposedAction()

    def set_result(self, pixmap, detections):
        self.pixmap, self.detections = pixmap, detections
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor("#121217"))
        if self.pixmap is None or self.pixmap.isNull():
            p.setPen(QtGui.QColor("#8d8698"))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, self.placeholder)
            return
        size = self.pixmap.size().scaled(self.size(), QtCore.Qt.KeepAspectRatio)
        target = QtCore.QRectF((self.width() - size.width()) / 2, (self.height() - size.height()) / 2, size.width(), size.height())
        p.drawPixmap(target, self.pixmap, QtCore.QRectF(self.pixmap.rect()))
        sx = target.width() / self.pixmap.width()
        sy = target.height() / self.pixmap.height()
        palette = ["#a855f7", "#22d3ee", "#f59e0b", "#34d399", "#f472b6", "#60a5fa"]
        names = sorted({d["label"] for d in self.detections})
        for d in self.detections:
            color = QtGui.QColor(palette[names.index(d["label"]) % len(palette)])
            x1, y1, x2, y2 = d["box"]
            rect = QtCore.QRectF(target.left() + x1 * sx, target.top() + y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)
            p.setPen(QtGui.QPen(color, 2)); p.setBrush(QtCore.Qt.NoBrush); p.drawRect(rect)
            text = f"{d['label']} {d['conf']:.2f}"
            tw = p.fontMetrics().horizontalAdvance(text) + 10
            tag = QtCore.QRectF(rect.left(), max(target.top(), rect.top() - 19), tw, 18)
            p.fillRect(tag, color)
            p.setPen(QtGui.QColor("#0d0d10")); p.drawText(tag, QtCore.Qt.AlignCenter, text)


# ---------------------------------------------------------------------------
# Dataset archive
# ---------------------------------------------------------------------------
def make_dataset_zip(frames_dir: Path, classes: list[str], out_zip: Path, report=lambda _msg: None) -> dict:
    """Package the project exactly as it would be trained (same split) in
    standard Ultralytics YOLO layout, with a portable data.yaml."""
    from .label_train_dialog import build_dataset_split
    with tempfile.TemporaryDirectory(prefix="ai-lab-dataset-") as tmp:
        root = Path(tmp) / "dataset"
        report("Building train / val / test split…")
        info = build_dataset_split(frames_dir, classes, root)
        yaml = (root / "data.yaml").read_text(encoding="utf-8").splitlines()
        (root / "data.yaml").write_text("\n".join(l for l in yaml if not l.startswith("path:")) + "\n", encoding="utf-8")
        files = sorted(p for p in root.rglob("*") if p.is_file())
        partial = out_zip.with_name(out_zip.name + ".part")
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as z:
            for i, path in enumerate(files, 1):
                z.write(path, path.relative_to(root).as_posix())
                if i % 25 == 0:
                    report(f"Packaging {i}/{len(files)} files…")
        os.replace(partial, out_zip)
    info = {k: v for k, v in info.items() if k != "data_yaml"}
    info["zip"] = out_zip
    info["bytes"] = out_zip.stat().st_size
    return info
