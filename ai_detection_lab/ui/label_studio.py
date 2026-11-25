"""Dataset → annotation → training workspace.

A presentation layer over the existing, tested project/YOLO workers, plus
the tools in studio_tools: split view, dataset charts, keyboard shortcuts
with undo/redo, Predict and multi-format export. v6.0.0 removed the
automatic labeling tools (SAM smart annotation and model-assisted
auto-label): boxes are drawn by hand. Nothing here contacts the network.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets
from .. import config
from .panel_theme import DIALOG_STYLE, card, scroll
from .mission_widgets import icon, ElidedLabel
from . import studio_tools as tools


def tx(en, th):
    return th if config.LANGUAGE == "th" else en


def label(text, name="muted"):
    w = QtWidgets.QLabel(text)
    w.setObjectName(name)
    w.setWordWrap(True)
    return w


def one_line(text, name="muted"):
    """A short value that should never wrap (wrapped labels inside a
    QFormLayout get a too-small row height and are cut in half)."""
    w = QtWidgets.QLabel(text)
    w.setObjectName(name)
    w.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    return w


def button(text, handler=None, primary=False):
    w = QtWidgets.QPushButton(text)
    if primary:
        w.setObjectName("primaryButton")
    if handler:
        w.clicked.connect(handler)
    return w


STUDIO_STYLE = DIALOG_STYLE + """
QDialog#labelStudio {background:#101014;}
QWidget {color:#ededf3; font-family:"SF Pro Text","Leelawadee UI","Thonburi"; font-size:13px;}
QFrame#studioSidebar {background:#16161d; border-right:1px solid #2b2b38;}
QFrame#studioTopbar {background:#131319; border-bottom:1px solid #2b2b38;}
QFrame#card {background:#19191f; border:1px solid #30303c; border-radius:10px;}
QFrame#uploadZone {background:#1a1725; border:1px dashed #756094; border-radius:10px;}
QLabel#muted, QLabel#dlgSubtitle {color:#9b9aaa;}
QLabel#eyebrow {color:#888697; font-size:10px; font-weight:600;}
QLabel#sectionTitle {color:#e7e5ed; font-size:14px; font-weight:600;}
QLabel#studioTitle {color:#f4f2fa; font-size:26px; font-weight:700;}
QLabel#statValue {color:#f4f2fa; font-size:29px; font-weight:600;}
QLabel#badge {background:#262031; border:1px solid #4a365d; color:#d6b3fb; border-radius:6px; padding:5px 10px;}
QPushButton {background:#26262f; border-color:#3d3c49; border-radius:6px; color:#e7e5ed; min-height:22px;}
QPushButton:hover {background:#34303f; border-color:#79668e;}
QPushButton#primaryButton {background:#9333ea; border-color:#a651ef; color:white;}
QPushButton#primaryButton:hover {background:#a344ee;}
QPushButton#primaryButton:disabled, QPushButton:disabled {background:#222129; color:#777381; border-color:#34303d;}
QPushButton#studioNav {background:transparent; border:0; text-align:left; padding:10px 12px; color:#aaa5b7;}
QPushButton#studioNav:checked {background:#30203f; color:#e3bdff;}
QPushButton#studioNav:hover {background:#24212d;}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {background:#141419; border-color:#41404e; color:#e9e6f0; padding:7px;}
QListWidget, QTableWidget {background:#141419; border-color:#30303d; color:#e9e6f0; alternate-background-color:#19191f;}
QListWidget::item:selected, QTableWidget::item:selected {background:#392249; color:#f4e4ff;}
QHeaderView::section {background:#1e1d25; color:#b0aaba; border-bottom:1px solid #34303d;}
QComboBox QAbstractItemView {background:#22212a; color:#f1eaf9; selection-background-color:#4b2863;}
QProgressBar {background:#17141e; border-color:#3f3150;}
QProgressBar::chunk {background:#8030bc;}
QPlainTextEdit#terminal {background:#111016; border-color:#36303f; color:#bcb0cd;}
QTabBar::tab:selected {border-bottom-color:#ae64f1; color:#d1a1ff;}
QSlider::sub-page:horizontal, QSlider::handle:horizontal {background:#b477ef;}
QScrollBar::handle:vertical {background:#494052;}
"""


class UploadZone(QtWidgets.QFrame):
    files_dropped = QtCore.Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and all(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()


class MetricChart(QtWidgets.QWidget):
    """Render only recorded values from Ultralytics results.csv."""
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(190)
        self.rows = []
        self.column = "metrics/mAP50(B)"

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        plot = QtCore.QRectF(46, 15, max(1, self.width()-65), max(1, self.height()-50))
        values = []
        for row in self.rows:
            try:
                import math
                value = float(row[self.column])
                if math.isfinite(value):
                    values.append((float(row.get("epoch", len(values)+1)), value))
            except (ValueError, KeyError):
                continue
        max_value = max(1.0, max((v for _, v in values), default=1.0))
        for i in range(5):
            y = plot.bottom() - plot.height()*i/4
            p.setPen(QtGui.QColor("#302b3b"))
            p.drawLine(QtCore.QPointF(plot.left(), y), QtCore.QPointF(plot.right(), y))
            p.setPen(QtGui.QColor("#92849f"))
            p.drawText(QtCore.QRectF(0, y-8, 38, 18), QtCore.Qt.AlignRight, f"{max_value*i/4:.2f}")
        if not values:
            p.setPen(QtGui.QColor("#b1a3bf"))
            p.drawText(plot, QtCore.Qt.AlignCenter, tx("Metrics appear after the first epoch", "กราฟจะแสดงเมื่อจบ Epoch แรก"))
            return
        points = [QtCore.QPointF(plot.left()+i*plot.width()/max(1,len(values)-1), plot.bottom()-v/max_value*plot.height()) for i,(_,v) in enumerate(values)]
        path = QtGui.QPainterPath(points[0])
        for pt in points[1:]:
            path.lineTo(pt)
        p.setPen(QtGui.QPen(QtGui.QColor("#c084fc"), 2.5))
        p.drawPath(path)
        p.setBrush(QtGui.QColor("#dbafff"))
        p.drawEllipse(points[-1], 3, 3)
        p.setPen(QtGui.QColor("#9c8daa"))
        p.drawText(QtCore.QRectF(plot.left(), plot.bottom()+10, plot.width(), 20), QtCore.Qt.AlignCenter, f"Epoch {int(values[0][0])} — {int(values[-1][0])}")


class StudioLayout:
    """Mixin: retain every existing IO, label, train and safety handler."""
    def _build_ui(self):
        from . import label_train_dialog as legacy
        self.setObjectName("labelStudio")
        self.setWindowTitle("AI Detection Lab · Label & Train Studio")
        self._studio_ready = False
        self._metrics_stamp = None
        self._metric_rows = []
        self._run_state = "idle"
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.studio_sidebar = QtWidgets.QFrame()
        self.studio_sidebar.setObjectName("studioSidebar")
        self.studio_sidebar.setFixedWidth(188)
        side = QtWidgets.QVBoxLayout(self.studio_sidebar)
        side.setContentsMargins(14, 24, 14, 18)
        side.setSpacing(6)
        side.addWidget(label("AI LAB", "brand"))
        side.addWidget(label("VISION WORKSPACE", "eyebrow"))
        side.addSpacing(26)
        side.addWidget(label("WORKSPACE", "eyebrow"))
        self.studio_nav = {}
        self.nav_group = QtWidgets.QButtonGroup(self)
        for index, name, symbol in [(4,"Projects","folder"),(0,"Datasets","dashboard"),(1,"Annotate","photo"),(2,"Train","train"),(3,"Models","model")]:
            b = button(name, lambda checked=False, i=index: self._navigate(i))
            b.setObjectName("studioNav")
            b.setIcon(icon(symbol, "#b4a7c5"))
            b.setIconSize(QtCore.QSize(19,19))
            b.setCheckable(True)
            self.nav_group.addButton(b)
            side.addWidget(b)
            self.studio_nav[index] = b
        side.addStretch()
        side.addWidget(label("LOCAL WORKSPACE", "eyebrow"))
        side.addWidget(label(tx("Images and training stay on this computer.","จัดการข้อมูลและ Train บนเครื่องนี้")))
        side.addSpacing(12)
        side.addWidget(label(f"AI Detection Lab  /  v{config.APP_VERSION}", "eyebrow"))
        root.addWidget(self.studio_sidebar)
        body = QtWidgets.QVBoxLayout()
        body.setSpacing(0)
        root.addLayout(body, 1)
        top = QtWidgets.QFrame()
        top.setObjectName("studioTopbar")
        bar = QtWidgets.QHBoxLayout(top)
        bar.setContentsMargins(24,12,24,12)
        bar.addWidget(label("Workspace  /", "muted"))
        self.project_combo = QtWidgets.QComboBox()
        self.project_combo.setMinimumWidth(150)
        self.project_combo.setMaximumWidth(270)
        bar.addWidget(self.project_combo)
        bar.addStretch()
        bar.addWidget(label("Object detection", "badge"))
        self.fullscreen_btn = button("⛶", self._toggle_fullscreen)
        self.fullscreen_btn.setToolTip("Full screen")
        bar.addWidget(self.fullscreen_btn)
        body.addWidget(top)
        content = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(content)
        cv.setContentsMargins(24,20,24,14)
        cv.setSpacing(14)
        title_row = QtWidgets.QHBoxLayout()
        self.studio_title = label("Datasets", "studioTitle")
        title_row.addWidget(self.studio_title)
        title_row.addStretch()
        self.studio_status = label("No dataset", "badge")
        title_row.addWidget(self.studio_status)
        cv.addLayout(title_row)
        self.studio_subtitle = label("")
        cv.addWidget(self.studio_subtitle)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.tabBar().hide()
        # Indexes 0–2 are part of the legacy import/train workflow contract.
        self.tabs.addTab(self._build_import_tab(), "Datasets")
        self.tabs.addTab(self._build_label_tab(), "Annotate")
        self.tabs.addTab(self._build_train_tab(), "Train")
        self.tabs.addTab(self._build_models_page(), "Models")
        self.tabs.addTab(self._build_projects_page(), "Projects")
        self.tabs.currentChanged.connect(self._page_changed)
        cv.addWidget(self.tabs, 1)
        footer = QtWidgets.QHBoxLayout()
        self.log_toggle = button(tx("Activity log", "บันทึกการทำงาน"))
        self.log_toggle.setCheckable(True)
        self.log_toggle.toggled.connect(self._toggle_activity)
        footer.addWidget(self.log_toggle)
        self.activity_label = ElidedLabel(tx("Ready", "พร้อมใช้งาน"))
        self.activity_label.setObjectName("muted")
        footer.addWidget(self.activity_label, 1)
        self.continue_btn = button("Continue →", self._continue, True)
        footer.addWidget(self.continue_btn)
        cv.addLayout(footer)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setObjectName("terminal")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        self.log.setFixedHeight(115)
        self.log.hide()
        cv.addWidget(self.log)
        body.addWidget(content, 1)
        self._studio_ready = True
        self.setStyleSheet(STUDIO_STYLE)
        self.tabs.setCurrentIndex(4)
        self._page_changed(4)
        self._metrics_timer = QtCore.QTimer(self)
        self._metrics_timer.setInterval(2000)
        self._metrics_timer.timeout.connect(self._read_metrics)
        self._metrics_timer.start()

    def _build_import_tab(self):
        from .label_train_dialog import DEFAULT_FRAME_INTERVAL_SEC, EN_US_LOCALE
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(0,0,0,0)
        v.setSpacing(16)
        stats = QtWidgets.QHBoxLayout()
        self.dataset_stats = []
        for name in ["Images", "Annotated", "Background", "Classes"]:
            box, layout = card()
            val = label("0", "statValue")
            layout.addWidget(val)
            layout.addWidget(label(name))
            self.dataset_stats.append(val)
            stats.addWidget(box, 1)
        v.addLayout(stats)
        self.project_label = label("")
        v.addWidget(self.project_label)
        zone = UploadZone()
        zone.files_dropped.connect(self._import_dropped_files)
        zone.setObjectName("uploadZone")
        z = QtWidgets.QVBoxLayout(zone)
        z.setContentsMargins(28,25,28,25)
        glyph = QtWidgets.QLabel()
        glyph.setPixmap(icon("folder", "#cb9ef8").pixmap(36,36))
        glyph.setAlignment(QtCore.Qt.AlignCenter)
        z.addWidget(glyph)
        for text, style in [(tx("Drop images to build your dataset", "ลากภาพมาวางเพื่อสร้าง Dataset"),"sectionTitle"), (tx("Drop images or a folder here, or browse files below.","ลากภาพหรือโฟลเดอร์มาวางที่นี่ หรือเลือกไฟล์จากปุ่มด้านล่าง"),"muted")]:
            w = label(text, style)
            w.setAlignment(QtCore.Qt.AlignCenter)
            z.addWidget(w)
        row = QtWidgets.QHBoxLayout()
        row.addStretch()
        self.import_photos_btn = button(tx("Upload images", "นำเข้าภาพ"), primary=True)
        self.import_folder_btn = button(tx("Choose folder", "เลือกโฟลเดอร์"))
        row.addWidget(self.import_photos_btn)
        row.addWidget(self.import_folder_btn)
        row.addStretch()
        z.addLayout(row)
        v.addWidget(zone)
        video, vl = card(tx("Extract frames from video", "แปลงวิดีโอเป็นภาพ"))
        row = QtWidgets.QHBoxLayout()
        self.open_video_btn = button(tx("Choose video", "เลือกวิดีโอ"))
        row.addWidget(self.open_video_btn)
        self.video_path_label = label(tx("MP4 · AVI · MOV · MKV", "MP4 · AVI · MOV · MKV"))
        row.addWidget(self.video_path_label, 1)
        vl.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(label(tx("Frame interval", "ระยะห่างแต่ละภาพ")))
        self.interval_spin = QtWidgets.QDoubleSpinBox()
        self.interval_spin.setLocale(EN_US_LOCALE)
        self.interval_spin.setRange(.1,30)
        self.interval_spin.setDecimals(1)
        self.interval_spin.setValue(DEFAULT_FRAME_INTERVAL_SEC)
        self.interval_spin.setSuffix(" s")
        row.addWidget(self.interval_spin)
        row.addStretch()
        self.extract_btn = button(tx("Extract frames", "เริ่มแยกภาพ"))
        self.extract_btn.setEnabled(False)
        row.addWidget(self.extract_btn)
        vl.addLayout(row)
        self.extract_progress = QtWidgets.QProgressBar()
        self.extract_progress.hide()
        vl.addWidget(self.extract_progress)
        self.frame_count_label = label("0 images")
        vl.addWidget(self.frame_count_label)
        v.addWidget(video)
        v.addWidget(label(tx("Use varied angles, distances and backgrounds. Review every object before training.","ใช้ภาพหลายมุม ระยะ และฉากหลัง แล้วตรวจ Label ให้ครบก่อน Train")))
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(14)
        splits, sl = card("Splits")
        sl.addWidget(label(tx("Labeled images are divided once, the same way every training run.","ภาพที่ทำ Label จะถูกแบ่งแบบเดิมทุกครั้งที่ Train")))
        form = QtWidgets.QFormLayout()
        self.val_split_spin = QtWidgets.QSpinBox()
        self.val_split_spin.setRange(5, 50)
        self.val_split_spin.setSuffix(" %")
        self.test_split_spin = QtWidgets.QSpinBox()
        self.test_split_spin.setRange(0, 30)
        self.test_split_spin.setSuffix(" %")
        form.addRow("Validation", self.val_split_spin)
        form.addRow("Test", self.test_split_spin)
        sl.addLayout(form)
        self.split_summary = label("")
        sl.addWidget(self.split_summary)
        self.val_split_spin.valueChanged.connect(self._split_changed)
        self.test_split_spin.valueChanged.connect(self._split_changed)
        row.addWidget(splits, 1)
        share, shl = card(tx("Export & share", "ส่งออกและแชร์"))
        shl.addWidget(label(tx("Download the dataset in YOLO format.","ดาวน์โหลด Dataset แบบ YOLO")))
        self.export_dataset_btn = button(tx("Download dataset (.zip)", "ดาวน์โหลด Dataset (.zip)"), self._export_dataset_zip, True)
        shl.addWidget(self.export_dataset_btn)
        self.share_status = label("")
        shl.addWidget(self.share_status)
        shl.addStretch()
        row.addWidget(share, 1)
        v.addLayout(row)
        charts, cl = card("Charts")
        self.dataset_charts = tools.DatasetCharts()
        cl.addWidget(self.dataset_charts)
        v.addWidget(charts)
        v.addStretch()
        return scroll(page)

    def _build_label_tab(self):
        page = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(page)
        row.setContentsMargins(0,0,0,0)
        row.setSpacing(16)
        self.label_stack = QtWidgets.QStackedWidget()
        self.label_stack.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Ignored)
        self.label_stack.addWidget(self._build_gallery_page())
        self.label_stack.addWidget(self._build_editor_page())
        row.addWidget(self.label_stack, 1)
        sidebar, side = card("Classes")
        self.class_name_edit = QtWidgets.QLineEdit()
        self.class_name_edit.setPlaceholderText(tx("Class name", "ชื่อคลาส"))
        side.addWidget(self.class_name_edit)
        self.add_class_btn = button("+ Add class")
        side.addWidget(self.add_class_btn)
        self.class_list = QtWidgets.QListWidget()
        side.addWidget(self.class_list, 1)
        self.delete_class_btn = button(tx("Delete class", "ลบคลาส"))
        self.delete_class_btn.setEnabled(False)
        side.addWidget(self.delete_class_btn)
        self.box_sidebar, boxes = card("Annotations")
        self.box_list = QtWidgets.QListWidget()
        boxes.addWidget(self.box_list, 1)
        self.delete_box_btn = button(tx("Delete annotation", "ลบกรอบที่เลือก"))
        self.delete_box_btn.setEnabled(False)
        boxes.addWidget(self.delete_box_btn)
        self.no_object_btn = button(tx("Background image", "ภาพไม่มีวัตถุ"))
        self.no_object_btn.setCheckable(True)
        self.no_object_btn.toggled.connect(self._toggle_no_object)
        boxes.addWidget(self.no_object_btn)
        side.addWidget(self.box_sidebar, 1)
        side.addWidget(label(tx("Select a class, then draw a box. Labels save automatically.","เลือกคลาสแล้วลากกรอบ ระบบบันทึก Label อัตโนมัติ")))
        class_scroll = scroll(sidebar)
        class_scroll.setFixedWidth(224)
        row.addWidget(class_scroll)
        self.label_stack.currentChanged.connect(lambda i: self.box_sidebar.setVisible(i==1))
        self.box_sidebar.hide()
        return page

    def _build_gallery_page(self):
        from .label_train_dialog import GALLERY_THUMB_PX, GALLERY_THUMB_H
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(0,0,0,0)
        toolbar = QtWidgets.QHBoxLayout()
        self.image_search = QtWidgets.QLineEdit()
        self.image_search.setPlaceholderText(tx("Search images…", "ค้นหาภาพ…"))
        self.image_search.textChanged.connect(self._filter_gallery)
        toolbar.addWidget(self.image_search, 1)
        self.annotation_filter = QtWidgets.QComboBox()
        self.annotation_filter.addItems(["All images", "Unannotated", "Annotated", "Background", "Train split", "Val split", "Test split"])
        self.annotation_filter.currentIndexChanged.connect(self._filter_gallery)
        toolbar.addWidget(self.annotation_filter)
        v.addLayout(toolbar)
        self.frame_list = QtWidgets.QListWidget()
        self.frame_list.setViewMode(QtWidgets.QListWidget.IconMode)
        self.frame_list.setIconSize(QtCore.QSize(GALLERY_THUMB_PX,GALLERY_THUMB_H))
        self.frame_list.setGridSize(QtCore.QSize(GALLERY_THUMB_PX+26,GALLERY_THUMB_H+46))
        self.frame_list.setResizeMode(QtWidgets.QListWidget.Adjust)
        self.frame_list.setMovement(QtWidgets.QListWidget.Static)
        self.frame_list.setUniformItemSizes(True)
        self.frame_list.setSpacing(8)
        self.frame_list.setCursor(QtCore.Qt.PointingHandCursor)
        v.addWidget(self.frame_list, 1)
        self.gallery_empty = label(tx("No images yet. Add images in Datasets to start annotating.","ยังไม่มีภาพ — นำเข้าภาพจากหน้า Datasets เพื่อเริ่มทำ Label"))
        v.addWidget(self.gallery_empty)
        self.gallery_progress_label = label("0 images")
        v.addWidget(self.gallery_progress_label)
        row = QtWidgets.QHBoxLayout()
        self.mark_rest_no_object_btn = button(tx("Mark remaining as background", "ระบุภาพที่เหลือว่าไม่มีวัตถุ"), self._mark_rest_no_object)
        row.addWidget(self.mark_rest_no_object_btn)
        row.addStretch()
        self.open_editor_btn = button("Annotate →", primary=True)
        self.open_editor_btn.setEnabled(False)
        row.addWidget(self.open_editor_btn)
        v.addLayout(row)
        return page

    def _build_editor_page(self):
        from .label_train_dialog import AnnotationCanvas, _ZoomScrollArea
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(0,0,0,0)
        row = QtWidgets.QHBoxLayout()
        self.back_to_gallery_btn = button("← Images")
        row.addWidget(self.back_to_gallery_btn)
        self.tool_group = QtWidgets.QButtonGroup(self)
        self.box_tool_btn = button("□  Box")
        self.box_tool_btn.setToolTip("Draw bounding boxes (V)")
        for b in (self.box_tool_btn,):
            b.setCheckable(True)
            self.tool_group.addButton(b)
            row.addWidget(b)
        self.box_tool_btn.setChecked(True)
        self.undo_btn = button("Undo", self._undo)
        self.undo_btn.setToolTip("Undo (⌘Z)")
        self.redo_btn = button("Redo", self._redo)
        self.redo_btn.setToolTip("Redo (⇧⌘Z)")
        row.addWidget(self.undo_btn)
        row.addWidget(self.redo_btn)
        row.addStretch()
        self.shortcuts_btn = button("Shortcuts  ?", self._show_shortcuts)
        self.shortcuts_btn.setToolTip("Keyboard shortcuts (?)")
        row.addWidget(self.shortcuts_btn)
        row.addWidget(button("−", lambda: self._set_zoom(self._zoom_factor/1.25)))
        row.addWidget(button("Fit", lambda: self._set_zoom(1)))
        row.addWidget(button("+", lambda: self._set_zoom(self._zoom_factor*1.25)))
        v.addLayout(row)
        self.canvas = AnnotationCanvas()
        self.canvas.installEventFilter(self)
        self.canvas.boxes_changed.connect(self._record_history)
        self._history = {}
        self._snapshot = []
        self.canvas_scroll = _ZoomScrollArea()
        self.canvas_scroll.setWidget(self.canvas)
        self.canvas_scroll.setWidgetResizable(True)
        self.canvas_scroll.setMinimumHeight(120)
        self.canvas.setMinimumSize(260, 120)
        v.addWidget(self.canvas_scroll, 1)
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0,0)
        v.addWidget(self.frame_slider)
        self.current_class_label = label("")
        v.addWidget(self.current_class_label)
        self.editor_help = label(tx("Drag to draw · Click to select · Drag corners to resize · Delete to remove", "ลากเพื่อวาด · คลิกเพื่อเลือก · ลากมุมเพื่อปรับขนาด · Delete เพื่อลบ"))
        v.addWidget(self.editor_help)
        row = QtWidgets.QHBoxLayout()
        self.prev_frame_btn = button("← Previous")
        self.prev_frame_btn.setEnabled(False)
        self.next_frame_btn = button("Next →", primary=True)
        self.next_frame_btn.setEnabled(False)
        self.frame_nav_label = label("— / —")
        row.addWidget(self.prev_frame_btn)
        row.addStretch()
        row.addWidget(self.frame_nav_label)
        row.addStretch()
        row.addWidget(self.next_frame_btn)
        v.addLayout(row)
        self._install_editor_shortcuts(page)
        return page

    def _build_train_tab(self):
        from .label_train_dialog import DEFAULT_EPOCHS, EN_US_LOCALE, TRAIN_IMGSZ
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(0,0,0,0)
        top = QtWidgets.QHBoxLayout()
        setup, s = card("Training configuration")
        self.train_summary_label = label("")
        s.addWidget(self.train_summary_label)
        form = QtWidgets.QFormLayout()
        form.setSpacing(14)
        # Same layout on every platform. macOS' default form keeps fields at
        # their size hint and never gives wrapped text its extra height, so
        # rows were cut in half on a Mac while looking fine elsewhere.
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        form.addRow("Base model", one_line("YOLOv8n  ·  Detection", "sectionTitle"))
        form.addRow("Compute", self._build_device_picker())
        form.addRow("Image size", one_line(f"{TRAIN_IMGSZ} × {TRAIN_IMGSZ} px"))
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setLocale(EN_US_LOCALE)
        self.epochs_spin.setRange(1,300)
        self.epochs_spin.setValue(DEFAULT_EPOCHS)
        form.addRow("Epochs", self.epochs_spin)
        self.train_split_label = one_line("")
        form.addRow("Split", self.train_split_label)
        s.addLayout(form)
        s.addWidget(label(tx("Batch size adapts to the dataset. Training and validation are split automatically.","ปรับ Batch ตามจำนวนภาพ และแบ่งชุด Train / Validation อัตโนมัติ")))
        top.addWidget(setup, 3)
        ready, r = card("Dataset readiness")
        self.train_readiness_label = label("")
        r.addWidget(self.train_readiness_label)
        r.addStretch()
        self.train_btn = button("Start training", primary=True)
        self.train_btn.setEnabled(False)
        r.addWidget(self.train_btn)
        self.train_progress = QtWidgets.QProgressBar()
        self.train_progress.hide()
        r.addWidget(self.train_progress)
        self.train_progress_label = label(tx("No training in progress", "ยังไม่ได้เริ่ม Train"))
        r.addWidget(self.train_progress_label)
        top.addWidget(ready, 2)
        v.addLayout(top)
        results, rv = card("Training metrics")
        metric_row = QtWidgets.QHBoxLayout()
        self.metric_values = {}
        for title, key in [("Precision","metrics/precision(B)"),("Recall","metrics/recall(B)"),("mAP50","metrics/mAP50(B)"),("mAP50–95","metrics/mAP50-95(B)")]:
            box = QtWidgets.QVBoxLayout()
            val = label("—", "statValue")
            box.addWidget(val)
            box.addWidget(label(title))
            metric_row.addLayout(box, 1)
            self.metric_values[key] = val
        rv.addLayout(metric_row)
        self.test_metrics_label = label("", "badge")
        self.test_metrics_label.hide()
        rv.addWidget(self.test_metrics_label)
        self.chart_metric = QtWidgets.QComboBox()
        for name, column in [("mAP50","metrics/mAP50(B)"),("mAP50–95","metrics/mAP50-95(B)"),("Box loss","train/box_loss"),("Classification loss","train/cls_loss")]:
            self.chart_metric.addItem(name, column)
        self.chart_metric.currentIndexChanged.connect(self._chart_changed)
        rv.addWidget(self.chart_metric)
        self.metrics_chart = MetricChart()
        rv.addWidget(self.metrics_chart)
        self.metrics_source = label(tx("Waiting for results.csv from a real training run.","รอผล results.csv จากการ Train จริง"))
        rv.addWidget(self.metrics_source)
        v.addWidget(results)
        v.addStretch()
        return scroll(page)

    def _build_models_page(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(0,0,0,0)
        row = QtWidgets.QHBoxLayout()
        self.import_model_btn = button("Import .pt model", primary=True)
        row.addWidget(self.import_model_btn)
        row.addWidget(button("Refresh", self._refresh_models))
        row.addStretch()
        v.addLayout(row)
        self.models_table = QtWidgets.QTableWidget(0,3)
        self.models_table.setHorizontalHeaderLabels(["Model", "Size", "Format"])
        self.models_table.horizontalHeader().setSectionResizeMode(0,QtWidgets.QHeaderView.Stretch)
        self.models_table.verticalHeader().hide()
        self.models_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.models_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.models_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.models_table.itemSelectionChanged.connect(self._model_selection_changed)
        self.models_table.setShowGrid(False)
        self.models_table.setMinimumHeight(170)
        v.addWidget(self.models_table,1)
        self.models_empty = label(tx("No models yet. Train a dataset or import your .pt weights.","ยังไม่มีโมเดล — Train จาก Dataset หรือนำเข้าไฟล์ .pt"))
        v.addWidget(self.models_empty)
        self.import_status_label = label("")
        v.addWidget(self.import_status_label)
        self.save_status_label = label("")
        v.addWidget(self.save_status_label)
        predict, pv = card("Predict")
        row = QtWidgets.QHBoxLayout()
        self.predict_image_btn = button(tx("Choose image", "เลือกภาพ"), self._choose_predict_image, True)
        self.predict_sample_btn = button(tx("Dataset image", "ภาพจาก Dataset"), self._predict_dataset_image)
        row.addWidget(self.predict_image_btn)
        row.addWidget(self.predict_sample_btn)
        row.addStretch()
        self.predict_conf = QtWidgets.QDoubleSpinBox()
        self.predict_conf.setRange(.05, .95)
        self.predict_conf.setSingleStep(.05)
        self.predict_conf.setValue(.25)
        self.predict_iou = QtWidgets.QDoubleSpinBox()
        self.predict_iou.setRange(.1, .95)
        self.predict_iou.setSingleStep(.05)
        self.predict_iou.setValue(.7)
        row.addWidget(label("Confidence"))
        row.addWidget(self.predict_conf)
        row.addWidget(label("IoU"))
        row.addWidget(self.predict_iou)
        pv.addLayout(row)
        self.prediction_view = tools.PredictionView()
        self.prediction_view.image_dropped.connect(self._run_prediction)
        pv.addWidget(self.prediction_view)
        self.predict_result = label(tx("Drop an image here or choose one. Runs locally on the selected model.","ลากภาพมาวางหรือเลือกภาพ ระบบประมวลผลบนเครื่องด้วยโมเดลที่เลือก"))
        pv.addWidget(self.predict_result)
        self.predict_conf.valueChanged.connect(self._rerun_prediction)
        self.predict_iou.valueChanged.connect(self._rerun_prediction)
        v.addWidget(predict)
        actions, a = card(tx("Export & deploy", "ส่งออกและใช้งาน"))
        a.addWidget(label(tx("Export the selected model, choose the models used by the AI camera, then test a video clip.","ส่งออกโมเดลที่เลือก ตั้งค่าโมเดลสำหรับกล้อง AI แล้วทดสอบด้วยวิดีโอ")))
        row = QtWidgets.QHBoxLayout()
        # Export is PyTorch (.pt) only - no TorchScript / ONNX / CoreML choice.
        self.export_model_btn = button("Export .pt", self._export_pt)
        self.select_models_btn = button("Choose active models", self._choose_models)
        self.test_clip_btn = button("Test video →", primary=True)
        for w in (self.export_model_btn,self.select_models_btn,self.test_clip_btn):
            row.addWidget(w)
        a.addLayout(row)
        v.addWidget(actions)
        return scroll(page)

    def _build_projects_page(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(0,0,0,0)
        row = QtWidgets.QHBoxLayout()
        self.new_project_btn = button("+ New project", primary=True)
        self.open_project_btn = button("Browse projects")
        self.delete_project_btn = button("Delete current project")
        self.delete_project_btn.setObjectName("dangerButton")
        row.addWidget(self.new_project_btn)
        row.addWidget(self.open_project_btn)
        row.addStretch()
        row.addWidget(self.delete_project_btn)
        v.addLayout(row)
        self.project_table = QtWidgets.QTableWidget(0,2)
        self.project_table.setHorizontalHeaderLabels(["Project", "Task"])
        self.project_table.horizontalHeader().setSectionResizeMode(0,QtWidgets.QHeaderView.Stretch)
        self.project_table.verticalHeader().hide()
        self.project_table.setShowGrid(False)
        self.project_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.project_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.project_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.project_table.itemDoubleClicked.connect(self._open_table_project)
        v.addWidget(self.project_table,1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(label(tx("Double-click a project to continue working.","ดับเบิลคลิกโปรเจกต์เพื่อทำงานต่อ")),1)
        self.open_selected_project_btn = button("Open selected →", self._open_table_project)
        row.addWidget(self.open_selected_project_btn)
        v.addLayout(row)
        self.projects_empty = label(tx("Create your first project to organize images, annotations and training runs.","สร้างโปรเจกต์แรกเพื่อเก็บภาพ Label และผลการ Train ไว้ด้วยกัน"))
        v.addWidget(self.projects_empty)
        return page

    def _navigate(self, index):
        if index==1 and not self.tabs.isTabEnabled(1):
            index=0
        self.tabs.setCurrentIndex(index)
        self._page_changed(index)

    def _page_changed(self, index):
        if not self._studio_ready:
            return
        titles = ["Datasets", "Annotate", "Train", "Models", "Projects"]
        hints = [tx("01 / Add and prepare your training data", "01 / เพิ่มและเตรียมข้อมูลสำหรับฝึกโมเดล"),tx("02 / Review images and annotate every object", "02 / ตรวจภาพและทำ Label ให้ครบทุกวัตถุ"),tx("03 / Configure, train and inspect real metrics", "03 / ตั้งค่า Train และตรวจผลลัพธ์จริง"),tx("04 / Test, export and use your models", "04 / ทดสอบ ส่งออก และใช้งานโมเดล"),tx("Your computer vision projects, in one place", "รวมโปรเจกต์ Computer Vision ของคุณไว้ที่เดียว")]
        self.studio_title.setText(titles[index])
        self.studio_subtitle.setText(hints[index])
        self.studio_nav[index].setChecked(True)
        self.continue_btn.setText(["Annotate →","Train →","Models →","Datasets →","Open dataset →"][index])
        self.continue_btn.setEnabled(index not in (0,4) or self.session_dir is not None)
        if index==3:
            self._refresh_models()
        if index==4:
            self._refresh_project_table()
        if index==2:
            self._read_metrics()

    def _continue(self):
        self._navigate({0:1,1:2,2:3,3:0,4:0}[self.tabs.currentIndex()])

    def _toggle_activity(self, checked):
        self.log.setVisible(checked)
        self._adapt_editor()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._adapt_editor()

    def _adapt_editor(self):
        if not getattr(self, "_studio_ready", False):
            return
        compact = self.height() < 780
        self.editor_help.setVisible(not compact)
        self.current_class_label.setVisible(not (compact and self.log_toggle.isChecked()))
        # Narrow windows: short tool labels; undo/redo/help stay on the keyboard.
        narrow = self.width() < 1240
        for w, wide, short in ((self.back_to_gallery_btn, "← Images", "←"), (self.box_tool_btn, "□  Box", "Box")):
            w.setText(short if narrow else wide)
        for w in (self.undo_btn, self.redo_btn, self.shortcuts_btn):
            w.setVisible(not narrow)

    def _set_zoom(self, factor, norm_anchor=None, viewport_anchor=None):
        super()._set_zoom(factor, norm_anchor, viewport_anchor)
        if factor <= 1:
            self.canvas.setMinimumSize(260, 120)

    def _new_project(self):
        from .label_train_dialog import project_dir_for_name
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("New project")
        dialog.setMinimumWidth(460)
        dialog.setStyleSheet(STUDIO_STYLE)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(28,24,28,24)
        layout.setSpacing(16)
        layout.addWidget(label(tx("Create project", "สร้างโปรเจกต์"), "studioTitle"))
        layout.addWidget(label(tx("Organize images, labels and local training runs.","เก็บภาพ Label และผล Train ไว้ในโปรเจกต์เดียว")))
        name = QtWidgets.QLineEdit()
        name.setPlaceholderText(tx("Project name, e.g. field_objects", "ชื่อโปรเจกต์ เช่น field_objects"))
        layout.addWidget(name)
        layout.addWidget(label("Task  ·  Object detection", "badge"))
        actions = QtWidgets.QHBoxLayout()
        actions.addStretch()
        actions.addWidget(button("Cancel", dialog.reject))
        create = button("Create project", dialog.accept, True)
        create.setEnabled(False)
        name.textChanged.connect(lambda text: create.setEnabled(bool(text.strip())))
        name.returnPressed.connect(lambda: dialog.accept() if name.text().strip() else None)
        actions.addWidget(create)
        layout.addLayout(actions)
        if dialog.exec() != QtWidgets.QDialog.Accepted or not name.text().strip():
            return
        path = project_dir_for_name(name.text())
        if path.exists():
            result = QtWidgets.QMessageBox.question(self, "Project exists", tx("Open the existing project?", "เปิดโปรเจกต์ที่มีชื่อนี้อยู่แล้วหรือไม่?"), QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if result != QtWidgets.QMessageBox.Yes:
                return
        try:
            (path / "frames").mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Could not create project", str(exc))
            return
        self._activate_project(path)

    def _import_dropped_files(self, paths):
        from .label_train_dialog import IMAGE_IMPORT_SUFFIXES
        if self._extract_thread is not None or self._train_thread is not None:
            self.activity_label.setText(tx("Wait for the current operation to finish.", "รอให้งานปัจจุบันเสร็จก่อน"))
            return
        sources = []
        for path in paths:
            if path.is_dir():
                sources.extend(p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_IMPORT_SUFFIXES)
            elif path.is_file() and path.suffix.lower() in IMAGE_IMPORT_SUFFIXES:
                sources.append(path)
        sources = list(dict.fromkeys(sources))
        if sources:
            self._start_photo_import(sources, paths[0])
        else:
            self.activity_label.setText(tx("Drop image files or a folder containing images.", "กรุณาวางไฟล์ภาพหรือโฟลเดอร์ที่มีภาพ"))

    def _refresh_project_table(self):
        from .label_train_dialog import list_projects
        paths = list_projects()
        self.project_table.setRowCount(len(paths))
        for row, path in enumerate(paths):
            item = QtWidgets.QTableWidgetItem(path.name)
            item.setData(QtCore.Qt.UserRole, str(path))
            self.project_table.setItem(row,0,item)
            self.project_table.setItem(row,1,QtWidgets.QTableWidgetItem("Detection"))
            self.project_table.setRowHeight(row,48)
        self.projects_empty.setVisible(not paths)
        self.open_selected_project_btn.setEnabled(bool(paths))
        if paths:
            self.project_table.selectRow(0)

    def _open_table_project(self, *_):
        row = self.project_table.currentRow()
        if row >= 0:
            self._activate_project(Path(self.project_table.item(row,0).data(QtCore.Qt.UserRole)))

    def _activate_project(self, project_dir):
        if self._train_thread is not None:
            self.activity_label.setText(tx("Wait for training before switching projects.","รอให้ Train เสร็จก่อนเปลี่ยนโปรเจกต์"))
            self._sync_project_selector()
            return
        super()._activate_project(project_dir)
        self._history.clear()
        if self.session_dir == project_dir:
            self._metrics_stamp = None
            self._run_state = "idle"
            self.save_status_label.clear()
            self._navigate(0)

    def _refresh_project_ui(self):
        super()._refresh_project_ui()
        if not getattr(self,"_studio_ready",False):
            return
        from .label_train_dialog import _is_labeled, is_no_object, MIN_LABELED_FRAMES
        total = len(self.frame_paths)
        annotated = sum(_is_labeled(p) for p in self.frame_paths)
        background = sum(is_no_object(p) and not _is_labeled(p) for p in self.frame_paths)
        for w,value in zip(self.dataset_stats,[total,annotated,background,len(self.classes)]):
            w.setText(str(value))
        ready = annotated >= MIN_LABELED_FRAMES and bool(self.classes)
        self.studio_status.setText("Training" if self._train_thread else "Ready to train" if ready else "Not ready" if self.session_dir else "No dataset")
        self.project_label.setText(tx("No project selected. Create a project or upload images to begin.","ยังไม่ได้เลือกโปรเจกต์ — สร้างโปรเจกต์หรือนำเข้าภาพเพื่อเริ่มต้น") if not self.session_dir else f"{self.session_dir.name}  ·  {annotated + background}/{total} images reviewed")
        self.studio_nav[1].setEnabled(self.session_dir is not None)
        busy = self._train_thread is not None or self._extract_thread is not None
        self.new_project_btn.setEnabled(not busy)
        self.open_project_btn.setEnabled(not busy)
        self.delete_project_btn.setEnabled(self.session_dir is not None and not busy)
        self.project_combo.setEnabled(self.project_combo.count()>0 and self.project_combo.itemData(0) is not None and not busy)
        self._set_import_controls_enabled(not busy)
        self.epochs_spin.setEnabled(self._train_thread is None)
        self._refresh_split_view()
        self._filter_gallery()
        self._page_changed(self.tabs.currentIndex())

    def _filter_gallery(self, *_):
        if not hasattr(self,"frame_list"):
            return
        from .label_train_dialog import _is_labeled, is_no_object
        query = self.image_search.text().casefold()
        mode = self.annotation_filter.currentIndex()
        membership = getattr(self, "_membership", {})
        split_filter = {4: "train", 5: "val", 6: "test"}.get(mode)
        visible=0
        for i,path in enumerate(self.frame_paths):
            item=self.frame_list.item(i)
            if item is None:
                continue
            annotated, bg = _is_labeled(path), is_no_object(path)
            match = query in path.name.casefold() and (mode==0 or mode==1 and not annotated and not bg or mode==2 and annotated or mode==3 and bg and not annotated or split_filter is not None and membership.get(path)==split_filter)
            item.setHidden(not match)
            visible += bool(match)
        self.gallery_empty.setVisible(visible==0)
        if self.frame_paths:
            self.gallery_empty.setText(tx("No images match these filters.","ไม่มีภาพตรงกับตัวกรองนี้"))
        self.mark_rest_no_object_btn.setEnabled(bool(self.frame_paths))

    def _refresh_frame_list(self):
        super()._refresh_frame_list()
        if getattr(self,"_studio_ready",False):
            self._filter_gallery()

    def _update_current_class_label(self):
        super()._update_current_class_label()
        if not self.classes:
            self.current_class_label.setText(tx("Add a class in the right sidebar, then draw a box.","เพิ่มคลาสในแถบด้านขวา แล้วลากกรอบบนภาพ"))

    def _refresh_tab3_summary(self):
        super()._refresh_tab3_summary()
        if not getattr(self, "_studio_ready", False):
            return
        from .label_train_dialog import _is_labeled, MIN_LABELED_FRAMES
        count = sum(_is_labeled(p) for p in self.frame_paths)
        self.train_summary_label.setText(f"{len(self.frame_paths)} images  ·  {count} annotated  ·  {len(self.classes)} classes")
        self.train_readiness_label.setText("\n\n".join([
            ("✓ " if self.session_dir else "○ ") + tx("Dataset selected", "เลือก Dataset"),
            ("✓ " if self.classes else "○ ") + tx("At least one class", "มีอย่างน้อย 1 คลาส"),
            ("✓ " if count >= MIN_LABELED_FRAMES else "○ ") + tx(f"Annotated images: {count} / {MIN_LABELED_FRAMES} minimum", f"ภาพที่ทำ Label: {count} / ขั้นต่ำ {MIN_LABELED_FRAMES}"),
            tx("Label diverse scenes for useful validation.", "ใช้ภาพหลายฉากเพื่อให้ผล Validation มีความหมาย"),
        ]))

    def _refresh_models(self):
        if not getattr(self,"_studio_ready",False):
            return
        files = sorted(config.MODEL_DIR.glob("*.pt")) if config.MODEL_DIR.exists() else []
        current = self.models_table.currentRow()
        self.models_table.setRowCount(len(files))
        for row,path in enumerate(files):
            item=QtWidgets.QTableWidgetItem(path.name)
            item.setData(QtCore.Qt.UserRole,str(path))
            self.models_table.setItem(row,0,item)
            self.models_table.setItem(row,1,QtWidgets.QTableWidgetItem(f"{path.stat().st_size/1048576:.1f} MB"))
            self.models_table.setItem(row,2,QtWidgets.QTableWidgetItem("PyTorch"))
            self.models_table.setRowHeight(row,46)
        self.models_empty.setVisible(not files)
        if files:
            self.models_table.selectRow(max(0,min(current,len(files)-1)))
        self.export_model_btn.setEnabled(bool(files))
        self.test_clip_btn.setEnabled(bool(files))
        self.select_models_btn.setEnabled(bool(files))
        self.predict_image_btn.setEnabled(bool(files))
        self.predict_sample_btn.setEnabled(bool(files) and bool(self.frame_paths))

    def _model_selection_changed(self):
        self.export_model_btn.setEnabled(self.models_table.currentRow()>=0) if hasattr(self,"export_model_btn") else None

    def _export_pt(self):
        import shutil
        row=self.models_table.currentRow()
        if row<0:
            return
        source=Path(self.models_table.item(row,0).data(QtCore.Qt.UserRole))
        name,_=QtWidgets.QFileDialog.getSaveFileName(self,"Export model",source.name,"PyTorch weights (*.pt)")
        if name:
            dest=Path(name)
            if dest.suffix.lower()!=".pt":
                dest=dest.with_suffix(".pt")
            try:
                if dest.resolve()!=source.resolve():
                    shutil.copy2(source,dest)
                self.save_status_label.setText(f"Exported: {dest}")
            except OSError as exc:
                QtWidgets.QMessageBox.warning(self,"Export failed",str(exc))

    def _choose_models(self):
        from .model_select_dialog import ModelSelectDialog
        dlg=ModelSelectDialog(self)
        dlg.exec()

    def _on_import_finished(self, payload):
        super()._on_import_finished(payload)
        self._refresh_models()

    def _chart_changed(self, *_):
        if hasattr(self,"metrics_chart"):
            self.metrics_chart.column=self.chart_metric.currentData()
            self.metrics_chart.update()

    def _read_metrics(self):
        if not self._studio_ready or not self.isVisible():
            return
        path=self.session_dir / "runs" / "train" / "results.csv" if self.session_dir else None
        try:
            stamp=(str(path),path.stat().st_mtime_ns) if path and path.exists() else None
        except OSError:
            return  # a results file may be replaced while the worker starts
        if stamp==self._metrics_stamp and stamp is not None:
            return
        rows=[]
        if stamp:
            try:
                with path.open(newline="",encoding="utf-8-sig") as stream:
                    rows=[{k.strip():v.strip() for k,v in row.items() if k and v is not None} for row in csv.DictReader(stream)]
            except (OSError,csv.Error):
                return
        # A restarted run must not show the previous run as live progress.
        if self._train_thread is not None and stamp==getattr(self,"_pretrain_stamp",None):
            rows=[]
        self._metrics_stamp=stamp
        self._metric_rows=rows
        self.metrics_chart.rows=rows
        self.metrics_chart.update()
        for key,w in self.metric_values.items():
            try:
                value=float(rows[-1][key])
                w.setText(f"{value:.3f}")
            except (IndexError,KeyError,ValueError):
                w.setText("—")
        self.metrics_source.setText(f"{path.parent.name}/results.csv · {len(rows)} recorded epochs" if rows else tx("Waiting for metrics from a real training run.","รอผลจากการ Train จริง"))
        self._read_test_metrics()

    def _read_test_metrics(self):
        import json
        path = self.session_dir / "runs" / "train" / "test_metrics.json" if self.session_dir else None
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path and path.exists() and self._train_thread is None else None
        except (OSError, ValueError):
            data = None
        if not data:
            self.test_metrics_label.hide()
            return
        self.test_metrics_label.setText(
            f"Test split · {data.get('images', 0)} images  —  "
            f"Precision {data['metrics/precision(B)']:.3f}  ·  Recall {data['metrics/recall(B)']:.3f}  ·  "
            f"mAP50 {data['metrics/mAP50(B)']:.3f}  ·  mAP50–95 {data['metrics/mAP50-95(B)']:.3f}")
        self.test_metrics_label.show()

    def _build_device_picker(self):
        """MPS / CUDA / CPU for training (ai/devices.py). Labels come from the
        platform only - importing torch here would slow the Studio's open."""
        from ..ai import devices
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.train_device_combo = QtWidgets.QComboBox()
        # Size to the column, not to the longest choice - otherwise the
        # long "cuda ✗ ..." entry widens the whole form and squeezes the
        # other rows ("Base model", "Split") until they are cut in half.
        self.train_device_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.train_device_combo.setMinimumContentsLength(16)
        self.train_device_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.train_device_combo.view().setTextElideMode(QtCore.Qt.ElideNone)
        for value, text in devices.train_choices():
            self.train_device_combo.addItem(text, value)
            self.train_device_combo.setItemData(self.train_device_combo.count() - 1, text, QtCore.Qt.ToolTipRole)
        # the open list may be wider than the closed box: show every choice in full
        view = self.train_device_combo.view()
        view.setMinimumWidth(view.sizeHintForColumn(0) + 32)
        idx = self.train_device_combo.findData(str(getattr(config, "TRAIN_DEVICE", "auto") or "auto").lower())
        self.train_device_combo.setCurrentIndex(max(0, idx))
        self.train_device_combo.currentIndexChanged.connect(self._save_train_device)
        lay.addWidget(self.train_device_combo)
        self.train_device_note = label(tx(
            "If this computer lacks the chosen device, training uses CPU and says so in the log.",
            "ถ้าเครื่องนี้ไม่มีอุปกรณ์ที่เลือก จะเทรนด้วย CPU แทน และแจ้งในบันทึกการทำงาน"))
        lay.addWidget(self.train_device_note)
        if sys.platform == "win32":
            # v9.0.0: why cuda is not used + one-click CUDA install (ai/gpupack.py)
            self.gpupack_btn = button("⚡ GPU Pack (การ์ดจอ NVIDIA)…", self._open_gpupack)
            self.gpupack_btn.setToolTip("ตรวจว่าทำไมยังเทรนด้วย CPU และติดตั้งไฟล์ CUDA ให้ใช้การ์ดจอ NVIDIA")
            lay.addWidget(self.gpupack_btn)
        return box

    def _open_gpupack(self):
        from .gpupack_dialog import GpuPackDialog
        GpuPackDialog(self).exec()

    def _save_train_device(self):
        """Remember the choice (same per-user overrides file as Quick Parameters)."""
        value = self.train_device_combo.currentData()
        config.TRAIN_DEVICE = value
        try:
            path = Path(config.USER_OVERRIDES_PATH)
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            data["TRAIN_DEVICE"] = value
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _start_training(self):
        self._pretrain_stamp=self._metrics_stamp
        super()._start_training()
        if self._train_thread is not None:
            self._run_state="running"
            self._metrics_stamp=None
            self._refresh_project_ui()
            self.log_toggle.setChecked(True)

    def _on_train_finished(self, payload):
        best_path,calibration=payload if isinstance(payload,tuple) else (payload,None)
        from ..ai import devices
        used=getattr(self._train_thread,"device_used",None)
        self._train_thread=None
        self.train_progress.hide()
        self._run_state="completed" if best_path else "failed"
        self.train_progress_label.setText((tx("Training complete", "Train เสร็จแล้ว") + (f" · {devices.describe(used)}" if used else "")) if best_path else tx("Training failed — see Activity log", "Train ไม่สำเร็จ — ดูบันทึกการทำงาน"))
        if best_path is not None:
            self._last_best_pt=Path(best_path)
            self._save_model_to_project(calibration)
        else:
            self.log_toggle.setChecked(True)
        self._refresh_tab3_summary()
        self._metrics_stamp=None
        self._read_metrics()
        self._refresh_models()
        if best_path is not None:
            self._navigate(3)

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------
    def _run_task(self, fn, done, status=None):
        task = tools.BackgroundTask(fn, self)
        self._tasks = getattr(self, "_tasks", set())
        self._tasks.add(task)
        if status is not None:
            task.progress.connect(status.setText)

        def finished(result, error):
            self._tasks.discard(task)
            task.deleteLater()
            done(result, error)
        task.finished.connect(finished)
        task.start()
        return task

    # ------------------------------------------------------------------
    # Datasets: splits, charts and export
    # ------------------------------------------------------------------
    def _split_plan(self):
        from .label_train_dialog import plan_dataset_split, read_split_config
        ratios = read_split_config(self.session_dir)
        return plan_dataset_split(list(self.frame_paths), ratios["val"], ratios["test"])

    def _split_changed(self, *_):
        from .label_train_dialog import write_split_config
        if self.session_dir is None or getattr(self, "_loading_split", False):
            return
        try:
            write_split_config(self.session_dir, self.val_split_spin.value() / 100, self.test_split_spin.value() / 100)
        except OSError as exc:
            self.activity_label.setText(str(exc))
        self._refresh_split_view()
        self._filter_gallery()

    def _refresh_split_view(self):
        from .label_train_dialog import read_split_config, split_membership, class_colors, MIN_LABELED_FRAMES
        ratios = read_split_config(self.session_dir)
        self._loading_split = True
        self.val_split_spin.setValue(round(ratios["val"] * 100))
        self.test_split_spin.setValue(round(ratios["test"] * 100))
        self._loading_split = False
        has_project = self.session_dir is not None
        for w in (self.val_split_spin, self.test_split_spin):
            w.setEnabled(has_project and self._train_thread is None)
        plan = self._split_plan()
        self._membership = split_membership(plan)
        counts = {k: len(plan[k]) + len(plan[k + "_bg"]) for k in ("train", "val", "test")}
        text = f"Train {counts['train']}  ·  Val {counts['val']}  ·  Test {counts['test']}"
        unused = len(self.frame_paths) - sum(counts.values())
        self.split_summary.setText(text + (f"  ·  {unused} " + tx("not used yet", "ภาพยังไม่ถูกใช้") if unused else ""))
        self.train_split_label.setText(text)
        stats = tools.dataset_statistics(list(self.frame_paths), list(self.classes), self._membership)
        self.dataset_charts.set_stats(stats, class_colors())
        ready = has_project and len(plan["labeled"]) >= MIN_LABELED_FRAMES and bool(self.classes)
        busy = bool(getattr(self, "_share_busy", False))
        self.export_dataset_btn.setEnabled(ready and not busy)
        hint = tx(f"Label at least {MIN_LABELED_FRAMES} images to export.", f"ทำ Label อย่างน้อย {MIN_LABELED_FRAMES} ภาพเพื่อส่งออก")
        if not ready and not busy:
            self.share_status.setText(hint if has_project else "")
        elif ready and self.share_status.text() == hint:
            self.share_status.clear()

    def _set_share_busy(self, busy):
        self._share_busy = busy
        self._refresh_split_view()

    def _export_dataset_zip(self):
        if self.session_dir is None:
            return
        name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Download dataset", f"{self.session_dir.name}.zip", "ZIP archive (*.zip)")
        if not name:
            return
        dest = Path(name) if name.lower().endswith(".zip") else Path(name + ".zip")
        frames, classes = self.frames_dir, list(self.classes)
        self._set_share_busy(True)

        def done(info, error):
            self._set_share_busy(False)
            if error:
                self.share_status.setText(f"❌ {error}")
                return
            self.share_status.setText(f"✓ {dest.name} · train {info['train']} / val {info['val']} / test {info['test']} · {info['bytes']/1048576:.1f} MB")
            self._log(f"📦 Dataset exported: {dest}")
        self._run_task(lambda report: tools.make_dataset_zip(frames, classes, dest, report), done, self.share_status)

    # ------------------------------------------------------------------
    # Annotate: undo/redo, shortcuts (v6.0.0: SAM and auto-label removed)
    # ------------------------------------------------------------------
    def _history_key(self):
        if self.frame_paths and 0 <= self.current_frame_index < len(self.frame_paths):
            return str(self.frame_paths[self.current_frame_index])
        return None

    def _record_history(self):
        key = self._history_key()
        if key is None:
            return
        undo, redo = self._history.setdefault(key, ([], []))
        undo.append(self._snapshot)
        del undo[:-100]
        redo.clear()
        self._snapshot = self.canvas.boxes()
        self._update_undo_buttons()

    def _apply_boxes(self, boxes, record=True):
        if record:
            key = self._history_key()
            if key is None:
                return
            undo, redo = self._history.setdefault(key, ([], []))
            undo.append(self._snapshot)
            redo.clear()
        self.canvas.set_frame(self.canvas._pixmap, boxes)
        self._snapshot = self.canvas.boxes()
        self._on_canvas_boxes_changed()
        self._update_undo_buttons()

    def _undo(self):
        undo, redo = self._history.get(self._history_key(), ([], []))
        if undo:
            redo.append(self.canvas.boxes())
            self._apply_boxes(undo.pop(), record=False)

    def _redo(self):
        undo, redo = self._history.get(self._history_key(), ([], []))
        if redo:
            undo.append(self.canvas.boxes())
            self._apply_boxes(redo.pop(), record=False)

    def _update_undo_buttons(self):
        undo, redo = self._history.get(self._history_key(), ([], []))
        self.undo_btn.setEnabled(bool(undo))
        self.redo_btn.setEnabled(bool(redo))

    def _load_current_frame_into_canvas(self):
        super()._load_current_frame_into_canvas()
        if getattr(self, "_studio_ready", False):
            self._snapshot = self.canvas.boxes()
            self._update_undo_buttons()

    def _delete_class(self):
        super()._delete_class()
        # Class ids may have been renumbered in every label file.
        self._history.clear()
        self._snapshot = self.canvas.boxes()
        self._update_undo_buttons()

    SHORTCUTS = [
        ("1 – 9", "Select class (relabels the selected box)", "เลือกคลาส (เปลี่ยนคลาสกรอบที่เลือก)"),
        ("V", "Box tool", "เครื่องมือวาดกรอบ"),
        ("A / ←", "Previous image", "ภาพก่อนหน้า"),
        ("D / →", "Next image", "ภาพถัดไป"),
        ("Delete", "Delete selected box", "ลบกรอบที่เลือก"),
        ("Esc", "Deselect", "ยกเลิกการเลือก"),
        ("⌘Z / ⇧⌘Z", "Undo / redo", "ย้อนกลับ / ทำซ้ำ"),
        ("B", "Mark as background image", "ระบุว่าเป็นภาพไม่มีวัตถุ"),
        ("F  ·  +  ·  −", "Fit · zoom in · zoom out", "พอดีจอ · ซูมเข้า · ซูมออก"),
        ("?", "Show shortcuts", "แสดงคีย์ลัด"),
    ]

    def _install_editor_shortcuts(self, page):
        def add(sequence, handler):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), page, activated=handler)
            shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        for n in range(1, 10):
            add(str(n), lambda n=n: self._select_class_shortcut(n - 1))
        add("V", lambda: self.box_tool_btn.setChecked(True))
        add("A", self._prev_frame)
        add("D", self._next_frame)
        add("B", lambda: self.no_object_btn.toggle())
        add("F", lambda: self._set_zoom(1))
        add("+", lambda: self._set_zoom(self._zoom_factor * 1.25))
        add("=", lambda: self._set_zoom(self._zoom_factor * 1.25))
        add("-", lambda: self._set_zoom(self._zoom_factor / 1.25))
        add("Esc", lambda: self.canvas.select_index(None))
        add(QtGui.QKeySequence.Undo, self._undo)
        add(QtGui.QKeySequence.Redo, self._redo)
        add("Ctrl+Shift+Z", self._redo)
        add("?", self._show_shortcuts)
        add("Shift+/", self._show_shortcuts)

    def _select_class_shortcut(self, row):
        if 0 <= row < len(self.classes):
            self.class_list.setCurrentRow(row)

    def _show_shortcuts(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Keyboard shortcuts")
        dialog.setStyleSheet(STUDIO_STYLE)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.addWidget(label(tx("Keyboard shortcuts", "คีย์ลัด"), "studioTitle"))
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(22)
        for row, (keys, en, th) in enumerate(self.SHORTCUTS):
            grid.addWidget(label(keys, "badge"), row, 0)
            grid.addWidget(label(tx(en, th)), row, 1)
        layout.addLayout(grid)
        layout.addWidget(button("Close", dialog.accept, True), 0, QtCore.Qt.AlignRight)
        self._shortcuts_dialog = dialog
        dialog.open()

    # ------------------------------------------------------------------
    # Models: Predict and multi-format export
    # ------------------------------------------------------------------
    def _selected_model(self):
        row = self.models_table.currentRow()
        if row < 0 or self.models_table.item(row, 0) is None:
            return None
        return Path(self.models_table.item(row, 0).data(QtCore.Qt.UserRole))

    def _choose_predict_image(self):
        from .label_train_dialog import IMAGE_IMPORT_SUFFIXES
        pattern = " ".join("*" + s for s in IMAGE_IMPORT_SUFFIXES)
        name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose image", "", f"Images ({pattern})")
        if name:
            self._run_prediction(Path(name))

    def _predict_dataset_image(self):
        if not self.frame_paths:
            self.predict_result.setText(tx("Open a project with images first.", "เปิดโปรเจกต์ที่มีภาพก่อน"))
            return
        membership = getattr(self, "_membership", {})
        pool = [p for p in self.frame_paths if membership.get(p) in ("test", "val")] or list(self.frame_paths)
        self._predict_index = (getattr(self, "_predict_index", -1) + 1) % len(pool)
        self._run_prediction(pool[self._predict_index])

    def _rerun_prediction(self, *_):
        if getattr(self, "_predict_path", None) is not None and not getattr(self, "_predict_busy", False):
            self._run_prediction(self._predict_path)

    def _run_prediction(self, image_path):
        model = self._selected_model()
        if model is None or not model.exists():
            self.predict_result.setText(tx("Select a model in the table first.", "เลือกโมเดลในตารางก่อน"))
            return
        if getattr(self, "_predict_busy", False):
            return
        self._predict_path = Path(image_path)
        self._predict_busy = True
        self.predict_result.setText(f"{model.name} → {self._predict_path.name}…")
        conf, iou = self.predict_conf.value(), self.predict_iou.value()

        def done(result, error):
            self._predict_busy = False
            if error:
                self.prediction_view.set_result(None, [])
                self.predict_result.setText(f"❌ {error}")
                return
            pixmap = QtGui.QPixmap(str(self._predict_path))
            self.prediction_view.set_result(pixmap, result["detections"])
            counts = {}
            for d in result["detections"]:
                counts[d["label"]] = counts.get(d["label"], 0) + 1
            summary = ", ".join(f"{n} {name}" for name, n in counts.items()) or tx("no objects", "ไม่พบวัตถุ")
            speed = sum(result["speed"].values())
            self.predict_result.setText(f"{self._predict_path.name}  ·  {summary}  ·  {speed:.0f} ms  ·  {model.name}")
        path = self._predict_path
        self._run_task(lambda _report: tools.predict_image(model, path, conf, iou), done)

    @QtCore.Slot(str)
    def _on_status(self, msg):
        super()._on_status(msg)
        if hasattr(self,"activity_label"):
            self.activity_label.setText(msg)
