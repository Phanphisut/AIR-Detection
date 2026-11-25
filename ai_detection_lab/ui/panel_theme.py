"""Shared window chrome for the v5.1 workspaces."""
from PySide6 import QtCore, QtWidgets
from .mission_theme import STYLE
from .mission_widgets import icon

DIALOG_STYLE = STYLE + """
QDialog { background:#0c1119; }
QLabel { background:transparent; }
QLabel#dlgTitle {font-size:23px; font-weight:700; color:#f0f5fb;}
QLabel#dlgSubtitle, QLabel#paramDesc {color:#98aabc; font-size:12px;}
QLabel#paramName {font-weight:600;}
QFrame#card, QFrame#reportCard {background:#131d29; border:1px solid #29394b; border-radius:10px;}
QGroupBox {border:1px solid #29394b; border-radius:9px; margin-top:16px; padding:14px 10px 10px; font-weight:600;}
QGroupBox::title {subcontrol-origin:margin; left:12px; padding:0 5px; color:#a9c3d7;}
QPushButton#primaryButton {background:#67ddc3; color:#0b2826; border-color:#67ddc3; font-weight:600;}
QPushButton#primaryButton:hover {background:#86ebd4;}
QPushButton#dangerButton {background:#3a202a; border-color:#82404c; color:#ffaab6;}
QPushButton#primaryButton:disabled, QPushButton#dangerButton:disabled {background:#18222f; border-color:#283545; color:#64768a;}
QPushButton:checked {background:#1e4545; border-color:#67ddc3;}
QTabWidget::pane {border:0; background:transparent;}
QTabBar::tab {background:transparent; color:#91a5ba; padding:12px 18px; border-bottom:2px solid #253444; font-weight:600;}
QTabBar::tab:selected {color:#7be1cd; border-bottom:2px solid #67ddc3;}
QTabBar::tab:hover {background:#182636;}
QTreeView, QListWidget, QTableWidget {background:#101925; alternate-background-color:#14202d; border:1px solid #2a3a4d; border-radius:8px; color:#dfeaf4; padding:5px; outline:0;}
QListWidget::item, QTreeView::item {padding:9px 7px; border-radius:5px;}
QListWidget::item:selected, QTreeView::item:selected, QTableWidget::item:selected {background:#254449; color:#e1fff7;}
QHeaderView::section {background:#172331; border:0; border-bottom:1px solid #304154; padding:10px; color:#94a9be; font-weight:600;}
QTableWidget {gridline-color:#263747;}
QComboBox QAbstractItemView {background:#172433; color:#e4edf5; selection-background-color:#275453;}
QCheckBox, QRadioButton {spacing:8px; padding:4px 0;}
QProgressBar {min-height:18px; max-height:22px; border:1px solid #2a3d50; background:#101a27; border-radius:5px; text-align:center; color:#d9e9f4;}
QProgressBar::chunk {background:#286e65; border-radius:4px;}
QSplitter::handle {background:#233245; width:4px; height:4px;}
QSlider::groove:horizontal {height:4px; background:#2a3c50; border-radius:2px;}
QSlider::sub-page:horizontal {background:#67ddc3;}
QSlider::handle:horizontal {width:12px; margin:-5px 0; background:#8ee9d6; border-radius:6px;}
QLabel#reportSection {font-size:13px; font-weight:600; color:#7de0ca;}
QLabel#reportValue {font-size:28px; font-weight:700;}
QLabel#reportMuted {font-size:12px; color:#98aabc;}
"""


def heading(layout, title, subtitle, symbol="dashboard"):
    row = QtWidgets.QHBoxLayout()
    mark = QtWidgets.QLabel()
    mark.setPixmap(icon(symbol, "#78dfca").pixmap(30, 30))
    row.addWidget(mark)
    col = QtWidgets.QVBoxLayout()
    col.setSpacing(4)
    name = QtWidgets.QLabel(title)
    name.setObjectName("dlgTitle")
    sub = QtWidgets.QLabel(subtitle)
    sub.setObjectName("dlgSubtitle")
    sub.setWordWrap(True)
    col.addWidget(name)
    col.addWidget(sub)
    row.addLayout(col, 1)
    layout.addLayout(row)
    return row


def prepare(layout):
    layout.setContentsMargins(22, 20, 22, 18)
    layout.setSpacing(14)


def card(title=None):
    frame = QtWidgets.QFrame()
    frame.setObjectName("card")
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(20, 18, 20, 18)
    layout.setSpacing(12)
    if title:
        label = QtWidgets.QLabel(title)
        label.setObjectName("sectionTitle")
        layout.addWidget(label)
    return frame, layout


def scroll(widget):
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    area.setWidget(widget)
    return area
