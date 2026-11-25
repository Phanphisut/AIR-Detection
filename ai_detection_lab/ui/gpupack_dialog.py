"""v9.0.0 "⚡ GPU Pack" window (Train page): why Train is on CPU, and a one-click
download + install of the CUDA files (ai/gpupack.py) - no manual unzipping."""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .. import config
from ..ai import gpupack
from .panel_theme import DIALOG_STYLE, heading


class GpuPackDialog(QtWidgets.QDialog):
    diagnosed = QtCore.Signal(dict)
    progressed = QtCore.Signal(str, int, int)      # stage, done, total
    finished_step = QtCore.Signal(str, str)        # zip path or "", error or ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GPU Pack")
        self.setStyleSheet(DIALOG_STYLE)
        self.setMinimumWidth(600)
        self._cancel = threading.Event()
        self._busy = False
        self._diag = None
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(22, 20, 22, 18)
        v.setSpacing(12)
        heading(v, "⚡ GPU Pack",
                "ไฟล์ CUDA ให้ Train ด้วยการ์ดจอ NVIDIA (เร็วกว่า CPU หลายเท่า) · "
                f"ดาวน์โหลดจาก Hugging Face ({gpupack.SIZE / 1024 ** 3:.1f} GB) แล้วแอปติดตั้งให้เอง")
        self.status = QtWidgets.QLabel("กำลังตรวจเครื่องนี้…")
        self.status.setObjectName("statusText")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        v.addWidget(self.status)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.hide()
        v.addWidget(self.progress)
        self.progress_label = QtWidgets.QLabel("")
        self.progress_label.setObjectName("dlgSubtitle")
        self.progress_label.hide()
        v.addWidget(self.progress_label)
        row = QtWidgets.QHBoxLayout()
        self.install_btn = QtWidgets.QPushButton("ดาวน์โหลดและติดตั้ง GPU Pack")
        self.install_btn.setObjectName("primaryButton")
        self.install_btn.clicked.connect(self._download_and_install)
        self.file_btn = QtWidgets.QPushButton("ใช้ไฟล์ zip ที่โหลดไว้แล้ว…")
        self.file_btn.clicked.connect(self._choose_zip)
        self.recheck_btn = QtWidgets.QPushButton("ตรวจอีกครั้ง")
        self.recheck_btn.clicked.connect(self.refresh)
        self.cancel_btn = QtWidgets.QPushButton("ยกเลิก")
        self.cancel_btn.clicked.connect(self._cancel.set)
        self.cancel_btn.hide()
        close_btn = QtWidgets.QPushButton("ปิด")
        close_btn.clicked.connect(self.close)
        for b in (self.install_btn, self.file_btn, self.recheck_btn, self.cancel_btn):
            row.addWidget(b)
        row.addStretch()
        row.addWidget(close_btn)
        v.addLayout(row)
        self.diagnosed.connect(self._show_diagnosis)
        self.progressed.connect(self._show_progress)
        self.finished_step.connect(self._step_done)
        self._set_buttons()
        self.refresh()

    # --- check ---------------------------------------------------------------------
    def refresh(self):
        self.status.setText("กำลังตรวจเครื่องนี้… (โหลด PyTorch ครั้งแรกอาจใช้เวลาสักครู่)")
        self.install_btn.setEnabled(False)
        self.file_btn.setEnabled(False)
        threading.Thread(target=lambda: self.diagnosed.emit(gpupack.diagnose()), daemon=True,
                         name="AI-LAB-GPUPACK-CHECK").start()

    def _show_diagnosis(self, diag):
        self._diag = diag
        self.status.setText("\n".join(diag["lines"]))
        self._set_buttons()
        # A word-wrapped QLabel does not ask its layout for the height its
        # text needs: long reasons (folder paths) were cut off.
        width = max(self.status.width(), self.width() - 44)
        self.status.setMinimumHeight(self.status.heightForWidth(width) + 14)   # Thai marks sit above the line box
        self.adjustSize()

    def _set_buttons(self):
        can = bool(self._diag and self._diag.get("can_install")) and not self._busy
        self.install_btn.setEnabled(can)
        self.file_btn.setEnabled(can)
        self.recheck_btn.setEnabled(not self._busy)
        self.cancel_btn.setVisible(self._busy)

    # --- download / verify ----------------------------------------------------------------
    def _target(self) -> Path:
        return Path(config.DATA_ROOT) / "gpupack" / gpupack.FILE

    def _download_and_install(self):
        existing = gpupack.downloads_copy()
        if existing is not None:
            answer = QtWidgets.QMessageBox.question(
                self, "GPU Pack", f"เจอไฟล์ที่โหลดไว้แล้ว:\n{existing}\n\nใช้ไฟล์นี้เลยไหม? (ไม่ต้องโหลดใหม่)")
            if answer == QtWidgets.QMessageBox.Yes:
                self._start(existing, download=False)
                return
        target = self._target()
        target.parent.mkdir(parents=True, exist_ok=True)
        need = gpupack.FREE_SPACE_NEEDED + gpupack.SIZE
        free = gpupack.disk_free(target.parent)
        if free and free < need:
            QtWidgets.QMessageBox.warning(self, "GPU Pack",
                                          f"พื้นที่ดิสก์ไม่พอ: ต้องมีว่างอย่างน้อย {need / 1024 ** 3:.1f} GB "
                                          f"(ตอนนี้ว่าง {free / 1024 ** 3:.1f} GB)")
            return
        self._start(target, download=True)

    def _choose_zip(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "เลือกไฟล์ GPU Pack", str(Path.home() / "Downloads"),
                                                        "GPU Pack (*.zip)")
        if path:
            self._start(Path(path), download=False)

    def _start(self, path: Path, download: bool):
        self._busy = True
        self._cancel.clear()
        self.progress.setValue(0)
        self.progress.show()
        self.progress_label.show()
        self._set_buttons()

        def work():
            try:
                size, sha = gpupack.remote_info() if download else (None, None)
                if download:
                    gpupack.download(path, lambda d, t: self.progressed.emit("download", d, t),
                                     self._cancel.is_set, size=size)
                gpupack.check_zip(path)
                if download:
                    got = gpupack.sha256_of(path, lambda d, t: self.progressed.emit("verify", d, t))
                    if got != sha:
                        path.unlink(missing_ok=True)
                        raise IOError("ไฟล์ที่โหลดมาไม่ตรงกับบน Hugging Face (เสียระหว่างโหลด) — ลบแล้ว กดดาวน์โหลดใหม่")
                self.finished_step.emit(str(path), "")
            except Exception as exc:
                self.finished_step.emit("", f"{exc}")

        threading.Thread(target=work, daemon=True, name="AI-LAB-GPUPACK-DOWNLOAD").start()

    def _show_progress(self, stage, done, total):
        self.progress.setValue(int(done * 1000 / max(1, total)))
        gb = 1024 ** 3
        what = "กำลังดาวน์โหลด" if stage == "download" else "กำลังตรวจไฟล์"
        self.progress_label.setText(f"{what}  {done / gb:.2f} / {total / gb:.2f} GB")

    def _step_done(self, zip_path, error):
        self._busy = False
        self.progress.hide()
        self.progress_label.hide()
        self._set_buttons()
        if error:
            QtWidgets.QMessageBox.warning(self, "GPU Pack", error)
            return
        answer = QtWidgets.QMessageBox.question(
            self, "ติดตั้ง GPU Pack",
            "ไฟล์พร้อมแล้ว\n\nแอปจะปิดตัวเพื่อคัดลอกไฟล์ CUDA (ประมาณ 1–3 นาที) แล้วเปิดใหม่เอง\n"
            "ถ้า Windows ถามสิทธิ์ผู้ดูแล (UAC) ให้กด Yes\n\nงานที่ยังไม่ได้บันทึกให้บันทึกก่อน — ติดตั้งเลยไหม?")
        if answer != QtWidgets.QMessageBox.Yes:
            return
        if gpupack.launch_apply(Path(zip_path)):
            QtWidgets.QApplication.quit()
        else:
            QtWidgets.QMessageBox.warning(self, "GPU Pack",
                                          "เริ่มติดตั้งไม่ได้ (ไม่ได้กด Yes ที่หน้าต่างสิทธิ์ผู้ดูแล?) — ลองใหม่อีกครั้ง")

    def closeEvent(self, event):
        self._cancel.set()
        super().closeEvent(event)
