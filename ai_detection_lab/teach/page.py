"""
page.py
=======
หน้า "สอน AI" สำหรับเด็ก 4 ขั้น: ตั้งชื่อ -> อัปโหลดคลิป -> วาดกรอบทุกรูป -> สอนแล้วทดสอบ

ซ่อนศัพท์เทคนิคทั้งหมด (Epochs, Split, mAP) และไม่มีขั้นไหนขอให้อัปโหลดของอย่างอื่นที่ไม่ใช่สิ่งที่จะสอน
ถ้า AI ทักผิด เด็กกดที่กรอบแล้วบอกว่า "ไม่ใช่" รูปนั้นจะเข้าไปอยู่ในชุดสอนรอบหน้าเอง
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import cv2
from PySide6 import QtCore, QtWidgets

from ai_detection_lab.image_io import imread_any

from . import project as tp
from .clips import MAX_CLIP_SECONDS, MAX_CLIPS, VIDEO_SUFFIXES
from .widgets import FrameView, TimelineBar, bgr_to_pixmap
from .workers import (
    PRESETS,
    ExtractWorker,
    TestWorker,
    copy_external_clip,
    epochs_for,
    make_preview_callback,
)

EXAMPLES = ("ลูกบอล", "ตุ๊กตาหมี", "กรวยจราจร")
VIDEO_FILTER = "Videos (" + " ".join(f"*{s}" for s in VIDEO_SUFFIXES) + ")"
THUMB_SIZE = (160, 90)
PROJECT_ROWS_VISIBLE = 4
PROJECT_ROW_HEIGHT = 54
GOOD_SCORE = 80
OK_SCORE = 60

QSS = """
QWidget#teachRoot { background: #0b1118; color: #e8eef5; font-family: "Segoe UI", "Leelawadee UI", "Tahoma"; font-size: 11pt; }
QWidget#teachRoot QLabel { background: transparent; }
QFrame#stepBar { background: #0f1720; border-bottom: 1px solid #263544; }
QPushButton#stepBtn { background: transparent; color: #8ea0b2; border: none; border-radius: 8px;
    padding: 7px 14px; font-weight: 600; }
QPushButton#stepBtn:hover:enabled { color: #f4f8fb; background: #16212c; }
QPushButton#stepBtn:checked { background: #123a37; color: #8ff0dc; }
QPushButton#stepBtn:disabled { color: #46535f; }
QFrame#card { background: #111923; border: 1px solid #263544; border-radius: 12px; }
QFrame#dropZone { background: #0f1720; border: 2px dashed #36506a; border-radius: 12px; }
QFrame#dropZone[hover="true"] { border-color: #6ff0d8; background: #12262a; }
QLabel#title { font-size: 20pt; font-weight: 700; color: #f4f8fb; }
QLabel#body { color: #e8eef5; }
QLabel#thumb { background: #080d13; border-radius: 6px; }
QLabel#sub { color: #b7c4d1; }
QLabel#muted { color: #8ea0b2; font-size: 10pt; }
QLabel#warn { color: #f2c46b; font-size: 10pt; }
QLabel#score { font-size: 19pt; font-weight: 700; color: #f4f8fb; }
QLineEdit { background: #0b1118; border: 1px solid #36506a; border-radius: 8px; padding: 10px 12px;
    color: #f4f8fb; font-size: 14pt; }
QLineEdit:focus { border-color: #6ff0d8; }
QPushButton#primary { background: #17866f; color: #ffffff; border: none; border-radius: 10px;
    padding: 11px 24px; font-size: 12pt; font-weight: 700; }
QPushButton#primary:hover { background: #1ca387; }
QPushButton#primary:disabled { background: #1c2a33; color: #5b6b7a; }
QPushButton#plain { background: #16212c; color: #dfe7ef; border: 1px solid #2d3f50; border-radius: 10px;
    padding: 9px 16px; font-weight: 600; }
QPushButton#plain:hover { border-color: #6ff0d8; color: #ffffff; }
QPushButton#plain:disabled { color: #56636f; border-color: #1f2b36; }
QPushButton#chip { background: #16212c; color: #dfe7ef; border: 1px solid #2d3f50; border-radius: 16px;
    padding: 6px 14px; }
QPushButton#chip:hover { border-color: #6ff0d8; }
QPushButton#chip:checked { background: #123a37; color: #8ff0dc; border-color: #2f8f80; }
QPushButton#link { background: transparent; color: #8ea0b2; border: none; padding: 6px 8px; }
QPushButton#link:hover { color: #6ff0d8; }
QPushButton#project { background: #16212c; color: #dfe7ef; border: 1px solid #2d3f50; border-radius: 10px;
    padding: 10px 14px; text-align: left; }
QPushButton#project:hover { border-color: #6ff0d8; }
QProgressBar { background: #16212c; border: none; border-radius: 6px; height: 12px; }
QProgressBar::chunk { background: #2ccfa9; border-radius: 6px; }
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: #0f1720; width: 10px; border-radius: 5px; margin: 0; }
QScrollBar::handle:vertical { background: #36506a; border-radius: 5px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
"""


def _button(text, kind="plain", on_click=None):
    btn = QtWidgets.QPushButton(text)
    btn.setObjectName(kind)
    btn.setCursor(QtCore.Qt.PointingHandCursor)
    btn.setFocusPolicy(QtCore.Qt.NoFocus)  # ให้ลูกศร/Space ใช้เลื่อนรูปได้ตลอด
    if on_click is not None:
        btn.clicked.connect(lambda _=False: on_click())
    return btn


def _label(text="", kind="body", wrap=False):
    label = QtWidgets.QLabel(text)
    label.setObjectName(kind)
    label.setWordWrap(wrap)
    return label


def _card():
    frame = QtWidgets.QFrame()
    frame.setObjectName("card")
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(22, 18, 22, 18)
    layout.setSpacing(10)
    return frame, layout


class DropZone(QtWidgets.QFrame):
    dropped = QtCore.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)

    def _set_hover(self, flag):
        self.setProperty("hover", "true" if flag else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self._set_hover(True)
            event.acceptProposedAction()

    def dragLeaveEvent(self, _event):
        self._set_hover(False)

    def dropEvent(self, event):
        self._set_hover(False)
        self.dropped.emit([u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()])


class TeachPage(QtWidgets.QWidget):
    use_model_requested = QtCore.Signal()       # "ใช้กับโดรนลำเดียว"
    # งานสอนรันใน thread ของมันเอง: ส่งกลับ GUI ผ่าน signal เท่านั้น
    _train_progress = QtCore.Signal(int, int)
    _train_status = QtCore.Signal(str)
    _train_finished = QtCore.Signal(object, object)
    _train_preview = QtCore.Signal(int, object, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("teachRoot")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(QSS)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.project = None
        self.busy_probe = None      # ฟังก์ชันจากหน้าต่างหลัก: ตอนนี้โดรนกำลังเชื่อมต่อไหม
        self._queue = []
        self._extract = None
        self._train_thread = None
        self._test_worker = None
        self._retired_workers = []  # ตัวทดสอบของโปรเจกต์ที่ออกไปแล้ว: ต้องเก็บไว้จนจบจริง ไม่งั้น Qt ปิดโปรแกรม
        self._clip_id = None
        self._frame = None
        self._results = []
        self._result_index = -1
        self._preset = "quick"
        self._train_started = 0.0
        self._last_epoch_seen = 0
        self._current_step = 0
        self._build()
        self._train_progress.connect(self._on_train_progress)
        self._train_status.connect(self._on_train_status)
        self._train_finished.connect(self._on_train_finished)
        self._train_preview.connect(self._on_train_preview)
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setInterval(350)
        self._play_timer.timeout.connect(self._play_tick)
        self._go(0)

    # ------------------------------------------------------------
    # โครงหน้า
    # ------------------------------------------------------------
    def _build(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        bar = QtWidgets.QFrame()
        bar.setObjectName("stepBar")
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(16, 8, 16, 8)
        self.step_buttons = []
        group = QtWidgets.QButtonGroup(self)
        for i, text in enumerate(("1  Project", "2  Upload clips", "3  Draw boxes", "4  Train and test")):
            btn = _button(text, "stepBtn", (lambda i=i: self._go(i)) if i else self._leave_project)
            btn.setCheckable(True)
            group.addButton(btn)
            self.step_buttons.append(btn)
            row.addWidget(btn)
        self.project_label = _label("", "muted")
        row.addSpacing(16)
        row.addWidget(self.project_label)
        self.change_btn = _button("Change project", "chip", self._leave_project)
        row.addWidget(self.change_btn)
        row.addStretch(1)
        root.addWidget(bar)
        self.stack = QtWidgets.QStackedWidget()
        root.addWidget(self.stack, 1)
        for page in (self._build_name(), self._build_upload(), self._build_draw(), self._build_teach()):
            self.stack.addWidget(page)

    def _go(self, step):
        self._play_timer.stop()
        if step > 0 and self.project is None:
            step = 0
        self._current_step = step
        self.step_buttons[step].setChecked(True)
        self.stack.setCurrentIndex(step)
        if step == 0:
            self._refresh_projects()
        elif step == 1:
            self._refresh_clips()
        elif step == 2:
            self._enter_draw()
        elif step == 3:
            self._enter_teach()
        self._refresh_steps()
        self.setFocus()

    def _refresh_steps(self):
        has_project = self.project is not None
        has_clips = has_project and bool(self.project.clips)
        self.step_buttons[1].setEnabled(has_project)
        self.step_buttons[2].setEnabled(has_clips)
        self.step_buttons[3].setEnabled(has_clips and self.project.labeled_count() >= tp.MIN_LABELED)
        self.project_label.setText(f"กำลังสอน: {self.project.name}" if has_project else "")
        self.change_btn.setVisible(has_project)

    # ------------------------------------------------------------
    # ขั้นที่ 1: สร้างหรือเลือกโปรเจกต์
    # ------------------------------------------------------------
    def _build_name(self):
        page = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(page)
        outer.addStretch(1)
        col = QtWidgets.QVBoxLayout()
        col.addStretch(1)
        card, lay = _card()
        card.setMinimumWidth(620)
        lay.addWidget(_label("What should the AI learn to find?", "title"))
        lay.addWidget(_label("พิมพ์ชื่อของ 1 อย่าง แล้วเตรียมคลิปวิดีโอของสิ่งนั้นไว้", "sub"))
        self.name_input = QtWidgets.QLineEdit()
        self.name_input.setPlaceholderText("ลูกบอล")
        self.name_input.setMaxLength(30)
        self.name_input.returnPressed.connect(self._create_project)
        lay.addWidget(self.name_input)
        chips = QtWidgets.QHBoxLayout()
        for name in EXAMPLES:
            chips.addWidget(_button(name, "chip", lambda n=name: self.name_input.setText(n)))
        chips.addStretch(1)
        lay.addLayout(chips)
        self.name_error = _label("", "warn")
        lay.addWidget(self.name_error)
        lay.addWidget(_button("Start", "primary", self._create_project), 0, QtCore.Qt.AlignLeft)
        col.addWidget(card)
        col.addSpacing(16)
        self.projects_card, projects_lay = _card()
        projects_lay.addWidget(_label("Or continue a project", "sub"))
        scroll = self.projects_scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QtWidgets.QWidget()
        self.projects_box = QtWidgets.QVBoxLayout(holder)
        self.projects_box.setContentsMargins(0, 0, 0, 0)
        self.projects_box.setAlignment(QtCore.Qt.AlignTop)
        scroll.setWidget(holder)
        projects_lay.addWidget(scroll)
        col.addWidget(self.projects_card)
        col.addStretch(2)
        outer.addLayout(col)
        outer.addStretch(1)
        return page

    def _refresh_projects(self):
        while self.projects_box.count():
            item = self.projects_box.takeAt(0)
            if item.widget():
                item.widget().hide()  # deleteLater รอรอบถัดไป ระหว่างนั้นแถวเก่าจะยังโผล่ซ้อนอยู่
                item.widget().deleteLater()
        projects = tp.list_projects()
        self.projects_card.setVisible(bool(projects))
        for project in projects:
            self.projects_box.addWidget(self._project_row(project))
        # ScrollArea ไม่ขยายตามเนื้อหาเอง: ตั้งความสูงให้เห็นได้ไม่เกิน 4 แถว เกินจากนั้นเลื่อนดู
        self.projects_scroll.setFixedHeight(min(len(projects), PROJECT_ROWS_VISIBLE) * PROJECT_ROW_HEIGHT)

    def _project_row(self, project):
        text = f"{project.name}   ·   {len(project.clips)} คลิป   ·   มีกรอบ {project.labeled_count()} รูป"
        if project.model:
            text += "   ·   สอนแล้ว"
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(_button(text, "project", lambda p=project: self._open_project(p)), 1)
        row.addWidget(_button("Delete", "plain", lambda p=project: self._delete_project(p)))
        return box

    def _delete_project(self, project):
        extra = "\nโมเดลที่สอนไว้จะถูกลบด้วย" if project.model else ""
        answer = QtWidgets.QMessageBox.question(
            self, "Delete project", f"ลบโปรเจกต์ {project.name} ไหม?\nคลิป รูป และกรอบทั้งหมดจะหายไป กู้คืนไม่ได้{extra}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            return
        try:
            project.delete()
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Couldn't delete", f"ลบโปรเจกต์นี้ไม่ได้ ไฟล์อาจถูกเปิดอยู่\n\n{exc}")
        self._refresh_projects()

    def _leave_project(self):
        """กลับไปหน้าเลือกโปรเจกต์ ทุกอย่างบันทึกไว้แล้ว จึงกดเข้าโปรเจกต์ผิดแล้วออกได้ทันที"""
        if self._train_thread is not None or self._extract is not None:
            work = "สอน AI" if self._train_thread is not None else "ตัดรูปจากคลิป"
            QtWidgets.QMessageBox.information(self, "Can't leave yet", f"กำลัง{work}อยู่ รอให้เสร็จก่อน หรือกด Stop training")
            self.step_buttons[self._current_step].setChecked(True)
            return
        self._retire_test_worker()
        self.project = None
        self._results = []
        self._clip_id = None
        self._frame = None
        self.name_input.clear()
        self.name_error.setText("")
        self._go(0)

    def _retire_test_worker(self):
        worker = self._test_worker
        if worker is None:
            return
        self._test_worker = None
        worker.stop()
        worker.failed.disconnect()
        self._retired_workers.append(worker)
        worker.finished.connect(lambda w=worker: self._retired_workers.remove(w))

    def _create_project(self):
        name = self.name_input.text().strip()
        if not name:
            self.name_error.setText("พิมพ์ชื่อสิ่งที่จะสอนก่อน")
            return
        self.name_error.setText("")
        self._open_project(tp.TeachProject.create(name))

    def _open_project(self, project):
        self.project = project
        self._results = []
        self._clip_id = None
        self._frame = None
        if project.labeled_count() >= tp.MIN_LABELED and project.todo_count() == 0:
            self._go(3)
        elif project.clips:
            self._go(2)
        else:
            self._go(1)

    # ------------------------------------------------------------
    # ขั้นที่ 2: อัปโหลดคลิป
    # ------------------------------------------------------------
    def _build_upload(self):
        page = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(page)
        col.setContentsMargins(40, 24, 40, 24)
        col.setSpacing(14)
        self.upload_title = _label("", "title")
        col.addWidget(self.upload_title)
        zone = DropZone()
        zone.dropped.connect(self._add_clips)
        z = QtWidgets.QVBoxLayout(zone)
        z.setContentsMargins(24, 26, 24, 26)
        z.setAlignment(QtCore.Qt.AlignCenter)
        self.drop_text = _label("", "sub")
        self.drop_text.setAlignment(QtCore.Qt.AlignCenter)
        z.addWidget(self.drop_text)
        z.addWidget(_label(f"คลิปละไม่เกิน {MAX_CLIP_SECONDS} วินาที ไม่เกิน {MAX_CLIPS} คลิป", "muted"),
                    0, QtCore.Qt.AlignCenter)
        self.browse_btn = _button("Choose clips", "primary", self._browse_clips)
        z.addWidget(self.browse_btn, 0, QtCore.Qt.AlignCenter)
        col.addWidget(zone)
        self.extract_label = _label("", "sub")
        self.extract_bar = QtWidgets.QProgressBar()
        self.extract_bar.setTextVisible(False)
        col.addWidget(self.extract_label)
        col.addWidget(self.extract_bar)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(16)
        clips_card, clips_lay = _card()
        clips_lay.addWidget(_label("Your clips", "sub"))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QtWidgets.QWidget()
        self.clips_box = QtWidgets.QVBoxLayout(holder)
        self.clips_box.setContentsMargins(0, 0, 0, 0)
        self.clips_box.setAlignment(QtCore.Qt.AlignTop)
        scroll.setWidget(holder)
        clips_lay.addWidget(scroll, 1)
        row.addWidget(clips_card, 3)
        tips_card, tips_lay = _card()
        tips_lay.addWidget(_label("Filming tips", "sub"))
        for tip in ("เดินวนรอบ ๆ ให้เห็นหลายมุม", "ถ่ายทั้งใกล้และไกล", "ลองเปลี่ยนพื้นหลังหรือที่วาง",
                    "เริ่มหรือจบคลิปตอนที่ยังไม่เห็นของ จะช่วยให้ AI เก่งขึ้น"):
            tips_lay.addWidget(_label("•  " + tip, "muted", wrap=True))
        tips_lay.addStretch(1)
        row.addWidget(tips_card, 2)
        col.addLayout(row, 1)
        self.upload_next = _button("Next: draw boxes", "primary", lambda: self._go(2))
        col.addWidget(self.upload_next, 0, QtCore.Qt.AlignRight)
        return page

    def _refresh_clips(self):
        name = self.project.name
        self.upload_title.setText(f"Upload clips of the {name}")
        self.drop_text.setText(f"ลากคลิปวิดีโอของ{name}มาวางที่นี่")
        while self.clips_box.count():
            item = self.clips_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for clip in self.project.clips:
            self.clips_box.addWidget(self._clip_row(clip))
        if not self.project.clips:
            self.clips_box.addWidget(_label("ยังไม่มีคลิป เริ่มจากคลิปเดียวก็ได้", "muted"))
        busy = self._extract is not None
        self.extract_label.setVisible(busy)
        self.extract_bar.setVisible(busy)
        self.browse_btn.setEnabled(not busy)
        self.upload_next.setEnabled(bool(self.project.clips) and not busy)
        self._refresh_steps()

    def _clip_thumbnail(self, clip):
        """รูปกลางคลิป: เห็นได้ทันทีว่าเป็นคลิปอะไร (ต้นคลิปมักยังไม่เห็นของ)"""
        thumb = QtWidgets.QLabel()
        thumb.setObjectName("thumb")
        thumb.setFixedSize(*THUMB_SIZE)
        thumb.setAlignment(QtCore.Qt.AlignCenter)
        frames = self.project.clip_frames(clip["id"])
        image = imread_any(self.project.frames_dir / frames[len(frames) // 2]["file"]) if frames else None
        if image is not None:
            thumb.setPixmap(bgr_to_pixmap(image).scaled(*THUMB_SIZE, QtCore.Qt.KeepAspectRatio,
                                                        QtCore.Qt.SmoothTransformation))
        return thumb

    def _clip_row(self, clip):
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 4, 0, 8)
        row.setSpacing(14)
        row.addWidget(self._clip_thumbnail(clip), 0, QtCore.Qt.AlignTop)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(2)
        frames = self.project.clip_frames(clip["id"])
        done = sum(1 for f in frames if f["state"] != "unset")
        text.addWidget(_label(clip["name"]))
        text.addWidget(_label(f"{clip['duration']:.0f} วินาที   ·   {len(frames)} รูป   ·   ทำแล้ว {done} รูป", "muted"))
        for warning in clip.get("warnings", []):
            text.addWidget(_label(warning, "warn", wrap=True))
        text.addStretch(1)
        row.addLayout(text, 1)
        row.addWidget(_button("Remove", "link", lambda c=clip: self._remove_clip(c)), 0, QtCore.Qt.AlignTop)
        return box

    def _browse_clips(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose video clips", "", VIDEO_FILTER)
        self._add_clips(paths)

    def _add_clips(self, paths):
        wanted = [p for p in paths if Path(p).suffix.lower() in VIDEO_SUFFIXES]
        if paths and not wanted:
            QtWidgets.QMessageBox.information(self, "Not a video", "ใช้ได้เฉพาะไฟล์วิดีโอ เช่น .mp4 หรือ .mov")
            return
        room = MAX_CLIPS - len(self.project.clips) - len(self._queue) - (1 if self._extract else 0)
        if len(wanted) > room:
            QtWidgets.QMessageBox.information(self, "Too many clips", f"ใส่ได้ไม่เกิน {MAX_CLIPS} คลิป เอาคลิปเดิมออกก่อนได้")
            wanted = wanted[:max(0, room)]
        self._queue.extend(wanted)
        self._next_extract()

    def _next_extract(self):
        if self._extract is not None or not self._queue:
            if self.project is not None and self._current_step == 1:
                self._refresh_clips()
            return
        source = self._queue.pop(0)
        self.extract_label.setText(f"กำลังตัดรูปจาก {Path(source).name}…")
        self.extract_bar.setValue(0)
        self._extract = ExtractWorker(self.project, source)
        self._extract.progress.connect(lambda cur, total: (self.extract_bar.setMaximum(max(1, total)),
                                                           self.extract_bar.setValue(cur)))
        self._extract.finished_clip.connect(lambda cid, dest, res, s=source: self.project.add_clip(
            cid, Path(s).name, dest, res))
        self._extract.failed.connect(lambda msg: QtWidgets.QMessageBox.warning(self, "Can't use this clip", msg))
        self._extract.finished.connect(self._extract_done)
        self._extract.start()
        self._refresh_clips()

    def _extract_done(self):
        self._extract = None
        self._next_extract()
        if self.project is not None and self._current_step == 1:
            self._refresh_clips()

    def _remove_clip(self, clip):
        answer = QtWidgets.QMessageBox.question(
            self, "Remove clip", f"เอา {clip['name']} ออกไหม?\nกรอบที่วาดในคลิปนี้จะหายไปด้วย")
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.project.remove_clip(clip["id"])
        self._clip_id = None
        self._frame = None
        self._refresh_clips()

    # ------------------------------------------------------------
    # ขั้นที่ 3: วาดกรอบ
    # ------------------------------------------------------------
    def _build_draw(self):
        page = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(page)
        col.setContentsMargins(24, 14, 24, 14)
        col.setSpacing(8)
        self.draw_hint = _label("", "title")
        self.draw_hint.setWordWrap(True)
        col.addWidget(self.draw_hint)
        self.draw_sub = _label("", "sub", wrap=True)
        col.addWidget(self.draw_sub)
        self.clip_row = QtWidgets.QHBoxLayout()
        col.addLayout(self.clip_row)
        self.draw_view = FrameView()
        self.draw_view.set_draw_enabled(True)
        self.draw_view.box_drawn.connect(self._on_box_drawn)
        col.addWidget(self.draw_view, 1)
        self.timeline = TimelineBar()
        self.timeline.picked.connect(self._on_timeline_pick)
        col.addWidget(self.timeline)
        legend = _label("เขียว = วาดกรอบแล้ว    เทา = ไม่มีของในรูปนี้    ดำ = ยังไม่ได้ทำ", "muted")
        col.addWidget(legend)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(_button("◀", "plain", lambda: self._step_frame(-1)))
        self.play_btn = _button("Play", "plain", self._toggle_play)
        row.addWidget(self.play_btn)
        row.addWidget(_button("▶", "plain", lambda: self._step_frame(1)))
        self.frame_label = _label("", "muted")
        row.addSpacing(8)
        row.addWidget(self.frame_label)
        row.addStretch(1)
        self.none_btn = _button("Nothing in this picture", "plain", self._mark_none)
        row.addWidget(self.none_btn)
        self.todo_btn = _button("Next unfinished", "plain", self._goto_todo)
        row.addWidget(self.todo_btn)
        self.draw_next = _button("Done, train the AI", "plain", self._finish_draw)
        row.addWidget(self.draw_next)
        self.next_btn = _button("Next picture", "primary", self._advance)
        row.addWidget(self.next_btn)
        col.addLayout(row)
        return page

    def _enter_draw(self):
        while self.clip_row.count():
            item = self.clip_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.clip_buttons = {}
        group = QtWidgets.QButtonGroup(self.clip_row)
        for clip in self.project.clips:
            btn = _button(clip["name"], "chip", lambda c=clip["id"]: self._select_clip(c))
            btn.setCheckable(True)
            group.addButton(btn)
            self.clip_buttons[clip["id"]] = btn
            self.clip_row.addWidget(btn)
        self.clip_row.addStretch(1)
        if self._frame is None or self._frame not in self.project.frames:
            target = self.project.next_todo() or next((f for f in self.project.frames if f["clip"] >= 0), None)
            if target is None:
                self._go(1)
                return
            self._show_frame(target)
        else:
            self._show_frame(self._frame)

    def _select_clip(self, clip_id):
        frames = self.project.clip_frames(clip_id)
        todo = next((f for f in frames if f["state"] == "unset"), None)
        self._show_frame(todo or frames[0])

    def _show_frame(self, frame):
        self._frame = frame
        self._clip_id = frame["clip"]
        image = imread_any(self.project.frames_dir / frame["file"])
        self.draw_view.set_image(image)
        self._refresh_draw()

    def _refresh_draw(self):
        frame = self._frame
        project = self.project
        frames = project.clip_frames(self._clip_id)
        if self._clip_id in self.clip_buttons:
            self.clip_buttons[self._clip_id].setChecked(True)
        self.timeline.set_cells(f["state"] for f in frames)
        self.timeline.set_current(frames.index(frame))
        name = project.name
        boxes = [dict(box=b, color="#6ff0d8", label=name) for b in frame["boxes"]]
        self.draw_view.set_boxes(boxes)
        state = frame["state"]
        todo = project.todo_count()
        if state == "unset":
            hint = f"ลากกรอบรอบ{name}ในรูปนี้"
            sub = "ลากกรอบเสร็จแล้วปรับได้ ตรงแล้วกด Next picture ถ้ารูปนี้ไม่มี กด Nothing in this picture"
        elif state == "none":
            hint = f"รูปนี้ไม่มี{name}"
            sub = f"ถ้าจริง ๆ แล้วมี ลากกรอบรอบ{name}ได้เลย"
        elif todo:
            hint = "กรอบตรงไหม? ตรงแล้วกด Next picture"
            sub = f"ลากมุมหรือขอบเพื่อปรับขนาด ลากกลางกรอบเพื่อย้าย ลากพื้นที่ว่างเพื่อวาดใหม่  ·  เหลืออีก {todo} รูป"
        else:
            hint = "ครบทุกรูปแล้ว"
            sub = "ลากมุมหรือขอบเพื่อปรับกรอบได้ กด Play เพื่อตรวจทุกรูป ถ้าตรงแล้วไปสอน AI ได้เลย"
        self.draw_hint.setText(hint)
        self.draw_sub.setText(sub)
        self.frame_label.setText(f"Picture {frames.index(frame) + 1} / {len(frames)}")
        self.todo_btn.setEnabled(todo > 0)
        self.none_btn.setEnabled(state != "none")
        self.next_btn.setEnabled(frames.index(frame) + 1 < len(frames) or todo > 0)
        self._refresh_steps()

    def _on_timeline_pick(self, index):
        frames = self.project.clip_frames(self._clip_id)
        if 0 <= index < len(frames) and frames[index] is not self._frame:
            self._show_frame(frames[index])

    def _step_frame(self, delta):
        frames = self.project.clip_frames(self._clip_id)
        index = frames.index(self._frame) + delta
        if 0 <= index < len(frames):
            self._show_frame(frames[index])

    def _toggle_play(self):
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_btn.setText("Play")
        else:
            self._play_timer.start()
            self.play_btn.setText("Stop")

    def _play_tick(self):
        frames = self.project.clip_frames(self._clip_id)
        index = frames.index(self._frame)
        if index + 1 >= len(frames):
            self._toggle_play()
            return
        self._show_frame(frames[index + 1])

    def keyPressEvent(self, event):
        if self._current_step == 2:
            if event.key() == QtCore.Qt.Key_Left:
                self._step_frame(-1)
                return
            if event.key() in (QtCore.Qt.Key_Right, QtCore.Qt.Key_Space):
                self._step_frame(1)
                return
        super().keyPressEvent(event)

    def _on_box_drawn(self, box):
        self._stop_play()
        self.project.set_box(self._frame, box)
        self.project.save()
        self._refresh_draw()  # อยู่รูปเดิมให้ปรับกรอบต่อได้ เด็กกด รูปถัดไป เอง

    def _mark_none(self):
        self._stop_play()
        self.project.mark_none(self._frame)
        self.project.save()
        self._advance()

    def _stop_play(self):
        self._play_timer.stop()
        self.play_btn.setText("Play")

    def _advance(self):
        """ทำรูปนี้เสร็จ: ไปรูปถัดไปในคลิป จบคลิปแล้วไปรูปที่เหลือ ถ้าครบแล้วอยู่รูปเดิม"""
        frames = self.project.clip_frames(self._clip_id)
        index = frames.index(self._frame) + 1
        target = frames[index] if index < len(frames) else self.project.next_todo(self._frame)
        if target is not None:
            self._show_frame(target)
        else:
            self._refresh_draw()

    def _goto_todo(self):
        todo = self.project.next_todo(self._frame)
        if todo is not None:
            self._show_frame(todo)

    def _finish_draw(self):
        project = self.project
        labeled = project.labeled_count()
        if labeled < tp.MIN_LABELED:
            QtWidgets.QMessageBox.information(
                self, "Not enough boxes",
                f"ตอนนี้มีรูปที่มีกรอบ {labeled} รูป ต้องมีอย่างน้อย {tp.MIN_LABELED} รูป AI ถึงจะเริ่มเรียนได้\n"
                "ลองวาดกรอบเพิ่ม หรือเพิ่มคลิปอีกคลิป")
            return
        todo = project.todo_count()
        if todo:
            answer = QtWidgets.QMessageBox.question(
                self, "Some pictures left", f"ยังเหลืออีก {todo} รูปที่ยังไม่เสร็จ รูปพวกนี้จะไม่ถูกใช้สอน\nไปสอน AI เลยไหม?")
            if answer != QtWidgets.QMessageBox.Yes:
                return
        self._go(3)

    # ------------------------------------------------------------
    # ขั้นที่ 4: สอนและทดสอบ
    # ------------------------------------------------------------
    def _build_teach(self):
        self.teach_stack = QtWidgets.QStackedWidget()
        self.teach_stack.addWidget(self._build_train_page())
        self.teach_stack.addWidget(self._build_test_page())
        return self.teach_stack

    def _build_train_page(self):
        page = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(page)
        col.setContentsMargins(40, 24, 40, 24)
        col.setSpacing(12)
        self.train_title = _label("", "title")
        col.addWidget(self.train_title)
        self.train_sub = _label("", "sub", wrap=True)
        col.addWidget(self.train_sub)
        self.preset_row = QtWidgets.QHBoxLayout()
        self.preset_buttons = {}
        group = QtWidgets.QButtonGroup(self)
        for key, preset in PRESETS.items():
            btn = _button(preset["title"], "chip", lambda k=key: self._select_preset(k))
            btn.setCheckable(True)
            group.addButton(btn)
            self.preset_buttons[key] = btn
            self.preset_row.addWidget(btn)
        self.preset_row.addSpacing(10)
        self.preset_note = _label("", "muted")
        self.preset_row.addWidget(self.preset_note)
        self.preset_row.addStretch(1)
        col.addLayout(self.preset_row)
        self.train_bar = QtWidgets.QProgressBar()
        self.train_bar.setTextVisible(False)
        col.addWidget(self.train_bar)
        self.train_status = _label("", "muted")
        col.addWidget(self.train_status)
        self.train_view = FrameView()
        self.train_view.placeholder = "ระหว่างสอน จะเห็น AI ลองทายรูปนี้ และเก่งขึ้นเรื่อย ๆ"
        col.addWidget(self.train_view, 1)
        self.preview_label = _label("", "sub")
        col.addWidget(self.preview_label)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(_button("Back to the boxes", "plain", lambda: self._go(2)))
        row.addStretch(1)
        self.stop_btn = _button("Stop training", "plain", self._stop_training)
        row.addWidget(self.stop_btn)
        self.start_btn = _button("Start training", "primary", self._start_training)
        row.addWidget(self.start_btn)
        col.addLayout(row)
        return page

    def _enter_teach(self):
        if self._train_thread is not None:
            self.teach_stack.setCurrentIndex(0)
            return
        if self.project.model and self._results:
            self.teach_stack.setCurrentIndex(1)
            return
        if self.project.model and (tp.config.MODEL_DIR / self.project.model).is_file():
            self.teach_stack.setCurrentIndex(1)
            self._start_test()
            return
        self._show_train_setup()

    def _show_train_setup(self):
        self.teach_stack.setCurrentIndex(0)
        project = self.project
        plan = project.split_plan()
        labeled = sum(1 for split in plan.values() for f in split if f["boxes"])
        self.train_title.setText(f"Train the AI to find the {project.name}")
        self.train_sub.setText(f"ใช้ {labeled} รูปจาก {len(project.clips)} คลิป  ·  "
                               "เก็บช่วงท้ายของคลิปไว้ทดสอบ AI จะได้รู้ว่าเก่งจริงไหม")
        self.train_bar.setVisible(False)
        self.train_status.setText("")
        self.preview_label.setText("")
        self.stop_btn.setVisible(False)
        self.start_btn.setVisible(True)
        self.train_view.set_image(None)
        self.train_view.set_boxes([])
        for btn in self.preset_buttons.values():
            btn.setEnabled(True)
        self._select_preset(self._preset)

    def _select_preset(self, key):
        self._preset = key
        self.preset_buttons[key].setChecked(True)
        self.preset_note.setText(f"{PRESETS[key]['epochs']} รอบ  ·  " + {
            "quick": "เหมาะกับลองครั้งแรก",
            "normal": "ใช้เวลามากขึ้น แต่ AI แม่นขึ้น",
            "detailed": "นานที่สุด เหมาะกับเครื่องที่มีการ์ดจอ",
        }[key])

    def _start_training(self):
        from ai_detection_lab.ui.label_train_dialog import TrainAIThread

        if self.busy_probe is not None and self.busy_probe():
            answer = QtWidgets.QMessageBox.question(
                self, "Drone still connected",
                "การสอน AI ทำให้เครื่องช้ามาก ถ้าโดรนกำลังบินอาจสั่งงานไม่ทัน\nควรลงจอดก่อน จะสอนเลยไหม?")
            if answer != QtWidgets.QMessageBox.Yes:
                return
        project = self.project
        plan = project.split_plan()
        labeled = [f for split in plan.values() for f in split if f["boxes"]]
        if len(labeled) < tp.MIN_LABELED:
            QtWidgets.QMessageBox.information(self, "Not enough boxes", f"ต้องมีรูปที่มีกรอบอย่างน้อย {tp.MIN_LABELED} รูป")
            return
        preset = PRESETS[self._preset]
        self._train_imgsz = preset["imgsz"]
        epochs = epochs_for(preset, len(plan["train"]))
        dataset_dir = project.path / "dataset"

        def builder(_frames_dir, _classes, _dataset_dir):
            info = project.build_dataset(dataset_dir, plan)
            info["test"] = 0  # หน้าทดสอบวัดคะแนนเองอยู่แล้ว ไม่ต้องให้ TrainAIThread วัดซ้ำ
            return info

        # รูปตัวอย่างพรีวิว: รูปที่ AI ไม่ได้ใช้เรียน ถ้าไม่มีใช้รูปสุดท้ายที่มีกรอบ
        sample = next((f for f in plan["val"] + plan["test"] if f["boxes"]), labeled[-1])
        image = imread_any(project.frames_dir / sample["file"])
        self.train_view.set_image(image)
        self.train_view.set_boxes([])
        preview = make_preview_callback(image, sample["boxes"][0], self._train_preview.emit, preset["imgsz"])

        self.start_btn.setVisible(False)
        self.stop_btn.setVisible(True)
        self.stop_btn.setEnabled(True)
        for btn in self.preset_buttons.values():
            btn.setEnabled(False)
        self.train_bar.setVisible(True)
        self.train_bar.setRange(0, 0)  # ยังไม่รู้เวลา แสดงแบบวิ่ง
        self.train_status.setText("กำลังเตรียมรูป…")
        self.preview_label.setText("")
        self._train_started = time.monotonic()
        self._last_epoch_seen = 0
        self._stopped = False
        self._train_thread = TrainAIThread(
            frames_dir=project.frames_dir, classes=[project.name], dataset_dir=dataset_dir,
            runs_dir=project.path / "runs", run_name="train", epochs=epochs,
            on_progress=self._train_progress.emit,
            on_status=self._train_status.emit,
            on_error=self._train_status.emit,
            on_finished=self._train_finished.emit,
            dataset_builder=builder, imgsz=preset["imgsz"], batch=preset["batch"],
            extra_callbacks={"on_model_save": preview}, train_overrides=preset["overrides"],
        )
        self._train_thread.start()

    def _stop_training(self):
        if self._train_thread is not None:
            self._stopped = True
            self._train_thread.stop()
            self.stop_btn.setEnabled(False)
            self.train_status.setText("กำลังหยุดเมื่อจบรอบนี้…")

    @QtCore.Slot(int, int)
    def _on_train_progress(self, epoch, total):
        if self.train_bar.maximum() != total:
            self.train_bar.setRange(0, total)
            self._train_started = time.monotonic()
            self._last_epoch_seen = epoch - 1
        self.train_bar.setValue(epoch)
        done = epoch - self._last_epoch_seen
        left = ""
        if done > 0 and epoch < total:
            seconds = (time.monotonic() - self._train_started) / done * (total - epoch)
            left = f"   ·   เหลืออีกประมาณ {max(1, round(seconds / 60))} นาที" if seconds >= 60 else "   ·   ใกล้เสร็จแล้ว"
        self.train_status.setText(f"AI กำลังเรียน… รอบที่ {epoch} / {total}{left}")

    @QtCore.Slot(str)
    def _on_train_status(self, message):
        # ข้อความเทคนิคจาก TrainAIThread เก็บลง log เด็กไม่ต้องเห็น ยกเว้นตอนมีปัญหา
        print(message)
        if message.startswith("❌"):
            self._train_error = message

    @QtCore.Slot(int, object, float)
    def _on_train_preview(self, epoch, best, iou):
        name = self.project.name
        if best is None:
            text = f"รอบ {epoch}: ยังหา{name}ไม่เจอ"
            self.train_view.set_boxes([])
        else:
            color = "#6ff0d8" if iou >= 0.5 else "#ffb454"
            text = f"รอบ {epoch}: " + (f"เจอ{name}แล้ว" if iou >= 0.5 else f"ลองทาย แต่กรอบยังไม่ตรง{name}")
            self.train_view.set_boxes([dict(box=best["box"], color=color, label=f"{best['conf']:.0%}")])
        self.preview_label.setText(text)

    @QtCore.Slot(object, object)
    def _on_train_finished(self, best_path, calibration):
        self._train_thread = None
        project = self.project
        if best_path is None:
            error = getattr(self, "_train_error", "")
            self._train_error = ""
            self._show_train_setup()
            if not self._stopped:
                QtWidgets.QMessageBox.warning(self, "Training failed", f"สอน AI ไม่สำเร็จ ลองใหม่อีกครั้ง\n\n{error}")
            return
        project.install_model(best_path, calibration, self._train_imgsz)
        self._start_test()

    # หน้าทดสอบ --------------------------------------------------
    def _build_test_page(self):
        page = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(page)
        col.setContentsMargins(24, 14, 24, 14)
        col.setSpacing(8)
        self.score_label = _label("", "score")
        self.score_label.setWordWrap(True)
        col.addWidget(self.score_label)
        self.score_sub = _label("", "sub", wrap=True)
        col.addWidget(self.score_sub)
        self.test_view = FrameView()
        self.test_view.box_clicked.connect(self._on_test_box_clicked)
        col.addWidget(self.test_view, 1)
        self.test_timeline = TimelineBar()
        self.test_timeline.picked.connect(self._show_result)
        col.addWidget(self.test_timeline)
        col.addWidget(_label("เขียว = เจอ    แดง = หาไม่เจอ    ส้ม = ทักผิด    เทา = ไม่มีของ    "
                             "ถ้ามีกรอบผิด กดที่กรอบนั้นเพื่อบอก AI", "muted"))
        row = QtWidgets.QHBoxLayout()
        self.other_clip_btn = _button("Try another clip", "plain", self._test_other_clip)
        row.addWidget(self.other_clip_btn)
        self.add_clip_btn = _button("Add clips", "plain", lambda: self._go(1))
        row.addWidget(self.add_clip_btn)
        self.retrain_btn = _button("Train again", "plain", self._show_train_setup)
        row.addWidget(self.retrain_btn)
        row.addStretch(1)
        self.use_btn = _button("Use with the drone", "primary", lambda: self._use_model(self.use_model_requested))
        row.addWidget(self.use_btn)
        col.addLayout(row)
        return page

    def _start_test(self, video=None):
        project = self.project
        self.teach_stack.setCurrentIndex(1)
        self._results = []
        self._result_index = -1
        self._test_video = video
        self.test_timeline.set_cells([])
        self.test_view.set_image(None)
        self.test_view.set_boxes([])
        self.score_label.setStyleSheet("")
        self.score_label.setText("กำลังทดสอบ…")
        self.score_sub.setText("ให้ AI ลองดูวิดีโอที่ไม่เคยเห็น" if video is None else "ให้ AI ลองดูคลิปนี้")
        frames = None
        if video is None:
            plan = project.split_plan()
            frames = plan["test"] or plan["val"]
        self._set_test_buttons(False)
        conf, imgsz = project.model_settings()
        self._test_worker = TestWorker(tp.config.MODEL_DIR / project.model, conf, imgsz,
                                       frames=frames, frames_dir=project.frames_dir, video=video)
        self._test_worker.item.connect(self._on_test_item)
        self._test_worker.failed.connect(lambda msg: QtWidgets.QMessageBox.warning(self, "Problem", msg))
        self._test_worker.finished.connect(self._on_test_done)  # รอ thread จบจริงก่อนปล่อยอ็อบเจกต์ ไม่งั้น Qt ปิดโปรแกรม
        self._test_worker.start()

    def _set_test_buttons(self, enabled):
        for btn in (self.other_clip_btn, self.add_clip_btn, self.retrain_btn, self.use_btn):
            btn.setEnabled(enabled)

    def _use_model(self, signal):
        """ส่งโมเดลนี้ไปใช้: เลือกไว้ให้แล้วในขั้น เลือก AI (เพิ่มเข้าไป ไม่ลบตัวที่เลือกไว้เดิม)"""
        from ai_detection_lab.ai.yolo_detector import load_model_selection, save_model_selection

        model = self.project.model if self.project is not None else None
        if model:
            selection = load_model_selection(tp.config.MODEL_DIR)
            chosen = list(selection.get("selected") or []) if selection.get("mode") == "selected" else []
            if model not in chosen:
                chosen.append(model)
            save_model_selection(tp.config.MODEL_DIR, "selected", chosen)
        signal.emit()

    def _result_cell(self, result):
        if result.get("corrected"):
            return "empty"
        if result["wrong"]:
            return "wrong"
        if result["found"] is True:
            return "found"
        if result["found"] is False:
            return "miss"
        return "empty"

    def _on_test_item(self, result):
        if self.sender() is not self._test_worker:
            return  # ผลที่ค้างมาจากโปรเจกต์ที่ออกไปแล้ว
        self._results.append(result)
        self.test_timeline.set_cells(self._result_cell(r) for r in self._results)
        self._show_result(len(self._results) - 1)

    def _show_result(self, index):
        if not 0 <= index < len(self._results):
            return
        self._result_index = index
        result = self._results[index]
        self.test_view.set_image(result["image"])
        name = self.project.name
        boxes = []
        for det in result["dets"]:
            wrong = det.get("wrong")
            boxes.append(dict(box=det["box"], color="#e5534b" if wrong else "#6ff0d8", clickable=not wrong,
                              dashed=bool(wrong), label=f"ไม่ใช่{name}" if wrong else f"{name} {det['conf']:.0%}"))
        self.test_view.set_boxes(boxes)
        self.test_timeline.set_current(index)

    def _on_test_done(self):
        if self.sender() is not self._test_worker:
            return
        self._test_worker = None
        self._set_test_buttons(True)
        results = self._results
        name = self.project.name
        if not results:
            self.score_label.setText("ไม่มีรูปให้ทดสอบ")
            self.score_sub.setText("ลองเพิ่มคลิปแล้วสอนอีกรอบ")
            return
        if self._test_video is not None:
            seen = sum(1 for r in results if r["dets"])
            self.score_label.setText(f"AI เห็น{name}ใน {seen} จาก {len(results)} รูปของคลิปนี้")
            self.score_sub.setText(f"ดูว่าถูกจริงไหม ถ้ากรอบไหนไม่ใช่{name} กดที่กรอบนั้น แล้วกด Train again")
            return
        truth = [r for r in results if r["found"] is not None]
        found = sum(1 for r in truth if r["found"])
        wrong = sum(r["wrong"] for r in results)
        score = round(100 * found / len(truth)) if truth else 0
        color = "#3ec98a" if score >= GOOD_SCORE else "#e9a23b" if score >= OK_SCORE else "#e5534b"
        self.score_label.setStyleSheet(f"color: {color};")
        self.score_label.setText(f"AI เจอ{name} {score}% ของเวลาที่{name}อยู่ในจอ")
        self.score_sub.setText(self._advice(results, truth, wrong, score))
        self.project.last_result = dict(score=score, frames=len(truth), wrong=wrong)
        self.project.save()
        self._refresh_steps()

    def _advice(self, results, truth, wrong, score):
        name = self.project.name
        parts = [f"ทดสอบกับ {len(truth)} รูปจากช่วงท้ายคลิปที่ AI ไม่เคยเห็น"]
        if len(truth) < 5:
            parts[0] += " (รูปน้อย คะแนนอาจแกว่ง ถ้าเพิ่มคลิปจะวัดได้แม่นขึ้น)"
        missed = [r for r in truth if not r["found"]]
        hit = [r for r in truth if r["found"]]
        if missed and hit:
            area = lambda r: (r["truth"][0][2] - r["truth"][0][0]) * (r["truth"][0][3] - r["truth"][0][1])  # noqa: E731
            bright = lambda r: float(cv2.cvtColor(r["image"], cv2.COLOR_BGR2GRAY).mean())  # noqa: E731
            if statistics.mean(map(area, missed)) < 0.6 * statistics.mean(map(area, hit)):
                parts.append(f"หาไม่เจอตอน{name}อยู่ไกล ลองเพิ่มคลิปที่ถ่ายจากไกล ๆ")
            elif statistics.mean(map(bright, missed)) < 0.75 * statistics.mean(map(bright, hit)):
                parts.append("หาไม่เจอตอนภาพมืด ลองเพิ่มคลิปที่ถ่ายในที่มืด")
            elif score < GOOD_SCORE:
                parts.append("ลองเพิ่มคลิปที่ถ่ายหลายมุมหรือหลายพื้นหลัง")
        elif score < OK_SCORE:
            parts.append("ลองเพิ่มคลิป หรือเลือกสอนแบบ Normal")
        if wrong:
            parts.append(f"AI ทักผิด {wrong} ครั้ง กดที่กรอบที่ผิด แล้วกด Train again")
        return "  ·  ".join(parts)

    def _on_test_box_clicked(self, index):
        if self._test_worker is not None or self._train_thread is not None:
            return  # รอทดสอบจบก่อน ไม่ให้รูปในจอเปลี่ยนใต้มือตอนกำลังตอบ
        result = self._results[self._result_index]
        det = result["dets"][index]
        name = self.project.name
        if result["truth"] and any(tp.box_iou(det["box"], t) >= 0.5 for t in result["truth"]):
            QtWidgets.QMessageBox.information(self, "This box is right", f"กรอบนี้ตรงกับ{name}ที่เราวาดไว้ตอนแรก")
            return
        answer = QtWidgets.QMessageBox.question(self, "Tell the AI", f"กรอบนี้ไม่ใช่{name}ใช่ไหม?")
        if answer != QtWidgets.QMessageBox.Yes:
            return
        det["wrong"] = True
        keep = [d["box"] for d in result["dets"] if not d.get("wrong")]
        if result["frame"] is not None:
            self.project.add_correction(result["frame"], keep)
        else:
            if result.get("saved") is None:
                result["saved"] = self.project.add_external_correction(result["image"], keep)
            else:  # รูปเดียวกัน แก้กรอบที่ 2
                result["saved"]["boxes"] = [list(b) for b in keep]
                self.project.save()
        result["corrected"] = True
        self.test_timeline.set_cells(self._result_cell(r) for r in self._results)
        self._show_result(self._result_index)
        self.score_sub.setText(f"จำไว้แล้ว  ·  กด Train again แล้ว AI จะทักผิดน้อยลง")

    def _test_other_clip(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose a clip to try", "", VIDEO_FILTER)
        if path:
            self._start_external_test(path)

    def _start_external_test(self, path):
        try:
            self._start_test(video=copy_external_clip(self.project, path))
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Problem", f"อ่านคลิปไม่ได้: {exc}")

    # ------------------------------------------------------------
    def closeEvent(self, event):
        self._queue.clear()
        for worker in (self._extract, self._test_worker, *self._retired_workers):
            if worker is not None:
                worker.stop()
                worker.wait(5000)
        if self._train_thread is not None:
            self._train_thread.stop()  # daemon thread: หยุดเมื่อจบรอบนี้ ไม่กันโปรแกรมปิด
        event.accept()
