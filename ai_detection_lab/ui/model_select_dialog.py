"""🧠 Choose Models - pick which .pt files the AI uses.

Applies everywhere the app detects (drone AI Camera, Webcam Test, Video Clip
Test, missions) because all of them go through YOLODetector, which reads the
saved choice (models/active_models.json). A running AI Camera picks up the
change on its next frame - no restart.
"""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .. import config
from ..i18n import tr
from ..ai.yolo_detector import (
    STOCK_MODEL_NAMES,
    apply_model_selection,
    load_model_selection,
    save_model_selection,
)
from .code_editor import DIALOG_STYLESHEET

_ROLE_NAME = QtCore.Qt.UserRole + 1


def list_model_files(model_dir: Path) -> list[Path]:
    return sorted((p for p in Path(model_dir).glob("*.pt") if p.is_file()), key=lambda p: p.name.lower())


class ModelSelectDialog(QtWidgets.QDialog):
    info_signal = QtCore.Signal(str, object)  # filename, {"task":..., "classes":[...]} or {"error":...}
    saved = QtCore.Signal(str, object)        # mode, [selected names]

    def __init__(self, parent=None, model_dir: Path | None = None):
        super().__init__(parent)
        self.model_dir = Path(model_dir or config.MODEL_DIR)
        self.setWindowTitle(tr("🧠 Choose Models"))
        self.setStyleSheet(DIALOG_STYLESHEET)
        self.resize(820, 690)
        self.setMinimumSize(680, 540)
        self._info: dict[str, dict] = {}

        root = QtWidgets.QVBoxLayout(self)
        from .panel_theme import heading, prepare, card
        prepare(root)
        heading(root, tr("Model library"), tr("Choose the models used by your AI camera and tests."), "model")
        intro = QtWidgets.QLabel(tr(
            "Choose which trained models (.pt) the AI uses to detect. This applies to the "
            "drone AI Camera, the Webcam Test and the Video Clip Test — and takes effect "
            "immediately, even while the camera is running."
        ))
        intro.setWordWrap(True)
        root.addWidget(intro)

        self.all_radio = QtWidgets.QRadioButton(tr("All models (All.pt) — use every .pt file in the folder"))
        self.sel_radio = QtWidgets.QRadioButton(tr("Only the models I tick below"))
        self.auto_radio = QtWidgets.QRadioButton(tr(
            "Automatic — every model I trained (the stock yolov8n.pt only if there is nothing else)"
        ))
        modes, modes_layout = card(tr("Selection policy"))
        for r in (self.sel_radio, self.all_radio, self.auto_radio):
            modes_layout.addWidget(r)
        root.addWidget(modes)

        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText(tr("Search models…"))
        self.search.textChanged.connect(self._filter_models)
        root.addWidget(self.search)
        self.list = QtWidgets.QListWidget()
        root.addWidget(self.list, 1)

        row = QtWidgets.QHBoxLayout()
        self.tick_all_btn = QtWidgets.QPushButton(tr("☑ Tick all"))
        self.tick_none_btn = QtWidgets.QPushButton(tr("☐ Untick all"))
        self.folder_btn = QtWidgets.QPushButton(tr("📂 Open models folder"))
        for b in (self.tick_all_btn, self.tick_none_btn, self.folder_btn):
            row.addWidget(b)
        row.addStretch()
        root.addLayout(row)

        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-weight:800; color:#8ce8c0;")
        root.addWidget(self.summary)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        buttons.button(QtWidgets.QDialogButtonBox.Save).setObjectName("primaryButton")
        root.addWidget(buttons)

        self.tick_all_btn.clicked.connect(lambda: self._tick_all(True))
        self.tick_none_btn.clicked.connect(lambda: self._tick_all(False))
        self.folder_btn.clicked.connect(self._open_folder)
        for r in (self.sel_radio, self.all_radio, self.auto_radio):
            r.toggled.connect(self._update_state)
        self.list.itemChanged.connect(lambda *_: self._update_state())
        self.info_signal.connect(self._on_info)

        self._populate()

    def _filter_models(self, text):
        for index in range(self.list.count()):
            item = self.list.item(index)
            item.setHidden(text.casefold() not in item.text().casefold())

    # -- building the list ---------------------------------------------------
    def _populate(self):
        files = list_model_files(self.model_dir)
        sel = load_model_selection(self.model_dir)
        currently_used = {p.name for p in apply_model_selection(files, sel)}
        self.list.blockSignals(True)
        self.list.clear()
        for p in files:
            item = QtWidgets.QListWidgetItem(self._item_text(p.name))
            item.setData(_ROLE_NAME, p.name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if p.name in currently_used else QtCore.Qt.Unchecked)
            self.list.addItem(item)
        self.list.blockSignals(False)
        {"all": self.all_radio, "selected": self.sel_radio}.get(sel["mode"], self.auto_radio).setChecked(True)
        if not files:
            self.summary.setText(tr("❌ No .pt files in the models folder yet — train a model first."))
        self._update_state()
        # Reading each model's class names needs PyTorch - do it off the GUI thread.
        threading.Thread(target=self._read_infos, args=(files,), daemon=True).start()

    def _item_text(self, name: str) -> str:
        info = self._info.get(name)
        tag = " (stock)" if name.lower() in STOCK_MODEL_NAMES else ""
        if info is None:
            return f"{name}{tag}   —  {tr('reading...')}"
        if "error" in info:
            return f"{name}{tag}   —  ❌ {tr('cannot be opened')}: {info['error']}"
        if info.get("task") != "detect":
            return f"{name}{tag}   —  ⚠️ {tr('not a detection model (skipped)')}"
        classes = info.get("classes", [])
        shown = ", ".join(classes[:6]) + (f" +{len(classes) - 6}" if len(classes) > 6 else "")
        return f"{name}{tag}   —  {tr('detects')}: {shown}"

    def _read_infos(self, files):
        if not files:
            return
        try:
            from ultralytics import YOLO
        except Exception as exc:
            for p in files:
                self._safe_emit(p.name, {"error": str(exc)})
            return
        for p in files:
            try:
                m = YOLO(str(p))
                names = m.names if hasattr(m, "names") else {}
                classes = list(names.values()) if isinstance(names, dict) else list(names)
                self._safe_emit(p.name, {"task": getattr(m, "task", None), "classes": [str(c) for c in classes]})
            except Exception as exc:
                self._safe_emit(p.name, {"error": str(exc)[:80]})

    def _safe_emit(self, name, info):
        try:
            self.info_signal.emit(name, info)
        except RuntimeError:
            pass  # dialog already closed

    @QtCore.Slot(str, object)
    def _on_info(self, name, info):
        self._info[name] = info
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(_ROLE_NAME) == name:
                self.list.blockSignals(True)
                item.setText(self._item_text(name))
                if info.get("task") not in (None, "detect") or "error" in info:
                    item.setCheckState(QtCore.Qt.Unchecked)
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled)
                self.list.blockSignals(False)
        self._update_state()

    # -- state -----------------------------------------------------------------
    def mode(self) -> str:
        if self.all_radio.isChecked():
            return "all"
        if self.sel_radio.isChecked():
            return "selected"
        return "auto"

    def ticked(self) -> list[str]:
        return [
            self.list.item(i).data(_ROLE_NAME)
            for i in range(self.list.count())
            if self.list.item(i).checkState() == QtCore.Qt.Checked
            and self.list.item(i).flags() & QtCore.Qt.ItemIsEnabled
        ]

    def _tick_all(self, on):
        self.sel_radio.setChecked(True)
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.flags() & QtCore.Qt.ItemIsEnabled:
                item.setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)

    def _update_state(self):
        mode = self.mode()
        self.list.setEnabled(mode == "selected")
        self.tick_all_btn.setEnabled(mode == "selected")
        self.tick_none_btn.setEnabled(mode == "selected")
        files = list_model_files(self.model_dir)
        if not files:
            return
        if mode == "all":
            names = [p.name for p in files]
        elif mode == "selected":
            names = self.ticked()
        else:
            names = [p.name for p in apply_model_selection(files, {"mode": "auto"})]
        self.summary.setText(
            f"🧠 {tr('The AI will use')} {len(names)} {tr('model(s)')}: " + (", ".join(names) or "—")
            if names else "⚠️ " + tr("Tick at least one model")
        )

    def _open_folder(self):
        from PySide6 import QtGui
        self.model_dir.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.model_dir)))

    def _save(self):
        mode = self.mode()
        selected = self.ticked() if mode == "selected" else []
        if mode == "selected" and not selected:
            QtWidgets.QMessageBox.warning(self, tr("🧠 Choose Models"), tr("Tick at least one model"))
            return
        try:
            save_model_selection(self.model_dir, mode, selected)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, tr("🧠 Choose Models"), f"{tr('Could not save')}: {exc}")
            return
        self.saved.emit(mode, selected)
        self.accept()
