"""
pixel_theme.py
==============
ธีม Pixel Art ทั้งโปรแกรม (ต้นแบบ: ภาพตัวละครพิกเซลละเอียด ขอบเข้ม มุมเป็นขั้นบันได มีแสงขอบบน/เงาขอบล่าง)

หลักการ: ไม่ต้องไล่แก้ stylesheet ทีละไฟล์ (มีราว 120 จุด) แต่แปลงทุก stylesheet ตอนถูกตั้งค่า
  - border-radius ทุกค่า -> 0           (มุมเหลี่ยมแบบพิกเซล)
  - border 1px -> 2px + สีขอบแบบ bevel  (บน/ซ้ายสว่าง ล่าง/ขวาเข้ม เหมือนปุ่มเกมพิกเซล)
  - font-family -> Pixelify Sans        (ภาษาไทยไม่มีในฟอนต์นี้ Qt จะใช้ Leelawadee UI แทนให้เอง)
  - ฟอนต์ monospace -> VT323            (ขยายขนาดให้ใกล้เดิม เพราะ VT323 ตัวเล็กกว่า)

ส่วนที่วาดเองด้วย QPainter ให้ใช้ draw_pixel_rect() แทน drawRoundedRect()
ปิดธีมได้ด้วยตัวแปรแวดล้อม AIR_PIXEL_UI=0
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

PIXEL_FAMILY = "Pixelify Sans"
MONO_FAMILY = "VT323"
FALLBACK = ("Leelawadee UI", "Tahoma")  # ภาษาไทย
UI_FAMILIES = (PIXEL_FAMILY,) + FALLBACK
MONO_FAMILIES = (MONO_FAMILY, "Consolas")
MONO_SCALE = 1.3
BORDER_PX = 2  # ความหนาเส้นขอบ = 1 "พิกเซลศิลป์"

_MONO_HINTS = ("consolas", "menlo", "cascadia", "monospace", "courier")
_enabled = False
_cache: dict[str, str] = {}

# ตัวบ่งชี้ checkbox/radio, slider, scrollbar แบบพิกเซล (ใส่ระดับแอป ทุกหน้าต่างได้ด้วย)
APP_QSS = """
QCheckBox::indicator, QRadioButton::indicator, QGroupBox::indicator {
    width: 12px; height: 12px; background: #0e1722;
    border: 2px solid #4b6680; border-color: #6d8aa5 #2c3d4f #2c3d4f #6d8aa5;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: #93efd9 #357366 #357366 #93efd9; }
QCheckBox::indicator:checked, QRadioButton::indicator:checked, QGroupBox::indicator:checked {
    background: #67ddc3; border-color: #b4f5e6 #2f7f70 #2f7f70 #b4f5e6;
}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled { background: #18222f; border-color: #293748; }
QSlider::handle:horizontal { width: 10px; margin: -6px 0; background: #8ee9d6;
    border: 2px solid #0b2826; }
QSlider::groove:horizontal { height: 4px; background: #2a3c50; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: #3d5670; }
QToolTip { border: 2px solid #536c83; }
"""


def enabled() -> bool:
    return _enabled


# ---------------------------------------------------------------- stylesheet transform
def _bevel(color: str) -> str | None:
    c = QtGui.QColor(color.strip())
    if not c.isValid() or c.alpha() == 0:
        return None
    hi, lo = c.lighter(135), c.darker(165)
    if c.lightness() > 200:  # ขอบสีอ่อนมาก: lighter ไม่เห็นผล ให้ตัวขอบเองเป็นแสง
        hi = c
    fmt = lambda q: q.name(QtGui.QColor.HexArgb if q.alpha() < 255 else QtGui.QColor.HexRgb)
    # top right bottom left
    return f"{fmt(hi)} {fmt(lo)} {fmt(lo)} {fmt(hi)}"


_BORDER = re.compile(r"(?<![-\w])border\s*:\s*(\d+(?:\.\d+)?)px\s+(solid|dashed|dotted)\s+([^;}\n]+?)\s*(;|(?=})|$)")
_BORDER_COLOR = re.compile(r"(?<![-\w])border-color\s*:\s*([^;}\n]+?)\s*(;|(?=})|$)")
_RADIUS = re.compile(r"(border(?:-(?:top|bottom)-(?:left|right))?-radius\s*:\s*)[^;}\n]+")
_FAMILY = re.compile(r"font-family\s*:\s*([^;}\n]+)")
_SIZE = re.compile(r"font-size\s*:\s*(\d+(?:\.\d+)?)(px|pt)")


def _border(m: re.Match) -> str:
    width, style, color, end = float(m.group(1)), m.group(2), m.group(3), m.group(4)
    width = max(BORDER_PX, int(round(width)))
    out = f"border: {width}px {style} {color}{end or ';'}"
    bevel = _bevel(color) if style == "solid" and " " not in color.strip() else None
    if bevel:
        out += f" border-color: {bevel};"
    return out


def _border_color(m: re.Match) -> str:
    value, end = m.group(1), m.group(2)
    bevel = _bevel(value) if " " not in value.strip() else None
    return f"border-color: {bevel or value}{end or ';'}"


def _quote(families) -> str:
    return ", ".join(f'"{f}"' for f in families)


def _block(body: str) -> str:
    body = _RADIUS.sub(lambda m: m.group(1) + "0px", body)
    body = _BORDER.sub(_border, body)
    body = _BORDER_COLOR.sub(_border_color, body)
    fam = _FAMILY.search(body)
    if fam:
        value = fam.group(1).lower()
        mono = any(h in value for h in _MONO_HINTS) or MONO_FAMILY.lower() in value
        body = _FAMILY.sub("font-family: " + _quote(MONO_FAMILIES if mono else UI_FAMILIES), body)
        if mono and MONO_FAMILY.lower() not in value:  # VT323 ตัวเล็กกว่าฟอนต์ monospace ทั่วไป
            body = _SIZE.sub(lambda m: f"font-size: {round(float(m.group(1)) * MONO_SCALE)}{m.group(2)}", body)
    return body


def pixelize(qss: str) -> str:
    """แปลง stylesheet ให้เป็นสไตล์พิกเซล (ผลลัพธ์ถูกแคชไว้ เพราะบางจุดตั้ง stylesheet ซ้ำทุกเฟรม)"""
    if not qss:
        return qss
    out = _cache.get(qss)
    if out is None:
        if "{" in qss:
            out = re.sub(r"\{([^{}]*)\}", lambda m: "{" + _block(m.group(1)) + "}", qss)
        else:  # stylesheet แบบ inline ของ widget เดียว ("color: red; border: ...")
            out = _block(qss)
        if len(_cache) > 4000:
            _cache.clear()
        _cache[qss] = out
    return out


# ---------------------------------------------------------------- painting helpers
def pixel_rect_path(rect, radius: float = 6.0) -> QtGui.QPainterPath:
    """สี่เหลี่ยมมุมขั้นบันได (แทนมุมโค้ง) — radius ใหญ่ = บันไดหลายขั้น"""
    r = QtCore.QRectF(rect)
    path = QtGui.QPainterPath()
    if not _enabled:
        path.addRoundedRect(r, radius, radius)
        return path
    path.addPolygon(QtGui.QPolygonF(_staircase(r, _steps(radius), 2.0)))
    path.closeSubpath()
    return path


def _steps(radius: float) -> int:
    return 0 if radius < 2 else (1 if radius < 7 else 2)


def _staircase(r: QtCore.QRectF, steps: int, u: float):
    """จุดของรูปหลายเหลี่ยมมุมขั้นบันได steps ขั้น ขั้นละ u พิกเซล"""
    l, t, rt, b = r.left(), r.top(), r.right(), r.bottom()
    P = QtCore.QPointF
    pts = []
    # ขอบบน: จากมุมซ้ายบนไปขวา
    for i in range(steps, 0, -1):          # ซ้ายบน (ไล่จากล่างขึ้นบน)
        pts += [P(l + (steps - i) * u, t + i * u), P(l + (steps - i + 1) * u, t + i * u)]
    pts.append(P(l + steps * u, t))
    pts.append(P(rt - steps * u, t))
    for i in range(1, steps + 1):          # ขวาบน
        pts += [P(rt - (steps - i + 1) * u, t + i * u), P(rt - (steps - i) * u, t + i * u)]
    pts.append(P(rt, t + steps * u))
    pts.append(P(rt, b - steps * u))
    for i in range(steps, 0, -1):          # ขวาล่าง
        pts += [P(rt - (steps - i) * u, b - i * u), P(rt - (steps - i + 1) * u, b - i * u)]
    pts.append(P(rt - steps * u, b))
    pts.append(P(l + steps * u, b))
    for i in range(1, steps + 1):          # ซ้ายล่าง
        pts += [P(l + (steps - i + 1) * u, b - i * u), P(l + (steps - i) * u, b - i * u)]
    pts.append(P(l, b - steps * u))
    pts.append(P(l, t + steps * u))
    return pts


def draw_pixel_rect(painter: QtGui.QPainter, rect, radius: float = 6.0) -> None:
    """ใช้แทน painter.drawRoundedRect(rect, r, r): ธีมพิกเซลเปิดอยู่ = มุมขั้นบันได ขอบคม (ไม่ antialias)"""
    if not _enabled:
        painter.drawRoundedRect(QtCore.QRectF(rect), radius, radius)
        return
    painter.save()
    painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
    r = QtCore.QRectF(rect)
    r = QtCore.QRectF(round(r.left()), round(r.top()), round(r.width()), round(r.height()))
    steps = _steps(radius)
    pen = painter.pen()
    if pen.style() != QtCore.Qt.NoPen:  # เส้นขอบหนา 2px ให้เข้ากับธีม
        pen.setWidthF(max(2.0, pen.widthF()))
        pen.setJoinStyle(QtCore.Qt.MiterJoin)
        painter.setPen(pen)
        r = r.adjusted(1, 1, -1, -1)
    painter.drawPolygon(QtGui.QPolygonF(_staircase(r, steps, 2.0)))
    painter.restore()


# ---------------------------------------------------------------- install
def _load_fonts(root: Path) -> list[str]:
    loaded = []
    for name in ("PixelifySans.ttf", "VT323-Regular.ttf"):
        for folder in (root / "assets" / "fonts", root / "fonts"):
            path = folder / name
            if path.is_file():
                fid = QtGui.QFontDatabase.addApplicationFont(str(path))
                if fid >= 0:
                    loaded += QtGui.QFontDatabase.applicationFontFamilies(fid)
                break
    return loaded


def _no_ligatures(font: QtGui.QFont) -> None:
    """ลิเกเจอร์ fi/fl ของ Pixelify Sans แสดงผิด (find -> "And") จึงปิดไว้"""
    try:
        for tag in ("liga", "clig", "dlig"):
            font.setFeature(QtGui.QFont.Tag(tag), 0)
    except AttributeError:  # Qt < 6.7
        font.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 0.01)  # ระยะห่างไม่เป็นศูนย์ = Qt ปิดลิเกเจอร์เอง


def install(app: QtWidgets.QApplication, project_root) -> bool:
    """เปิดธีมพิกเซล: โหลดฟอนต์, ตั้งฟอนต์หลัก, แปลงทุก stylesheet ที่ถูกตั้งหลังจากนี้"""
    global _enabled
    if _enabled or os.environ.get("AIR_PIXEL_UI", "1").strip() in ("0", "false", "off"):
        return _enabled
    families = _load_fonts(Path(project_root))
    if PIXEL_FAMILY not in families and PIXEL_FAMILY not in QtGui.QFontDatabase.families():
        print(f"[pixel_theme] font {PIXEL_FAMILY!r} not found - pixel theme off")
        return False

    font = QtGui.QFont()
    font.setFamilies(list(UI_FAMILIES))
    font.setPointSizeF(10)
    font.setHintingPreference(QtGui.QFont.PreferFullHinting)
    _no_ligatures(font)
    app.setFont(font)

    original_widget = QtWidgets.QWidget.setStyleSheet
    original_app = QtWidgets.QApplication.setStyleSheet

    def widget_set_style_sheet(self, qss):
        original_widget(self, pixelize(qss))

    def app_set_style_sheet(self, qss):
        original_app(self, APP_QSS + pixelize(qss))

    try:
        QtWidgets.QWidget.setStyleSheet = widget_set_style_sheet
        QtWidgets.QApplication.setStyleSheet = app_set_style_sheet
    except (TypeError, AttributeError) as exc:  # PySide รุ่นที่ไม่ยอมให้แทนเมธอด: ใช้แค่ฟอนต์พิกเซล
        print(f"[pixel_theme] stylesheet hook unavailable ({exc}) - pixel font only")
        return False
    _enabled = True
    app.setStyleSheet(app.styleSheet())  # ใส่ APP_QSS
    return True
