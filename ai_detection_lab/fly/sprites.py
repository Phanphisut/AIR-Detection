"""
sprites.py
==========
ตัวละครพิกเซลขยับได้ บนการ์ดหน้า Connect (Real drone / Simulator / Webcam only)

วาดเป็นภาพพิกเซลเล็ก (56x36) ทีละเฟรมไว้ล่วงหน้า แล้วขยายแบบ nearest-neighbour (พิกเซลคมไม่เบลอ)
สไตล์: ขอบเข้ม 1 พิกเซล, แสงด้านบน/เงาด้านล่าง, ตาโตมีจุดแสง
"""
from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

W, H = 56, 36
OUTLINE = QtGui.QColor("#0a0f16")


class Canvas:
    def __init__(self):
        self.px = [[None] * W for _ in range(H)]
        self.solid = [[False] * W for _ in range(H)]  # ส่วนที่มีขอบ (เงาพื้นไม่มีขอบ)

    def dot(self, x, y, color, solid=True):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < W and 0 <= y < H:
            self.px[y][x] = QtGui.QColor(color)
            self.solid[y][x] = solid

    def rect(self, x, y, w, h, color, solid=True):
        for yy in range(int(y), int(y + h)):
            for xx in range(int(x), int(x + w)):
                self.dot(xx, yy, color, solid)

    def ellipse(self, cx, cy, rx, ry, color, shade=True, solid=True):
        base = QtGui.QColor(color)
        for yy in range(int(cy - ry - 1), int(cy + ry + 2)):
            for xx in range(int(cx - rx - 1), int(cx + rx + 2)):
                dx, dy = (xx + .5 - cx) / max(rx, .5), (yy + .5 - cy) / max(ry, .5)
                if dx * dx + dy * dy <= 1.0:
                    c = base
                    if shade and dy < -0.45 and dx < 0.5:
                        c = base.lighter(118)
                    elif shade and dy > 0.5:
                        c = base.darker(125)
                    self.dot(xx, yy, c, solid)

    def outline(self, color=OUTLINE):
        add = []
        for y in range(H):
            for x in range(W):
                if self.solid[y][x]:
                    continue
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and self.solid[ny][nx]:
                        add.append((x, y))
                        break
        for x, y in add:
            self.px[y][x] = QtGui.QColor(color)

    def image(self) -> QtGui.QImage:
        img = QtGui.QImage(W, H, QtGui.QImage.Format_ARGB32)
        img.fill(QtCore.Qt.transparent)
        for y in range(H):
            for x in range(W):
                if self.px[y][x] is not None:
                    img.setPixelColor(x, y, self.px[y][x])
        return img


def _eye(c: Canvas, cx, cy, r, look=0, blink=False):
    """ตากลมโต: ขอบขาว + ตาดำ + จุดแสง (หลับตา = เส้นเดียว)"""
    if blink:
        c.rect(cx - r, cy, 2 * r + 1, 1, OUTLINE)
        return
    c.ellipse(cx + .5, cy + .5, r + .6, r + .6, "#f4fbff", shade=False)
    c.ellipse(cx + .5 + look, cy + .5, r - .7, r - .7, "#14213a", shade=False)
    c.dot(cx - 1 + look, cy - 1, "#ffffff")


def _shadow(c: Canvas, cx, y, half):
    c.rect(cx - half, y, 2 * half + 1, 1, QtGui.QColor(0, 0, 0, 90), solid=False)
    c.rect(cx - half + 2, y + 1, 2 * half - 3, 1, QtGui.QColor(0, 0, 0, 60), solid=False)


# ---------------------------------------------------------------- frames
def drone_frames(n=16):
    """โดรนตัวกลม ใบพัดหมุน ลอยขึ้นลง กะพริบตา"""
    frames = []
    for i in range(n):
        c = Canvas()
        bob = round(math.sin(i / n * 2 * math.pi) * 1.5)
        cx, cy = 28, 17 + bob
        # แขน + มอเตอร์
        c.rect(9, cy - 5, 38, 2, "#3b4b5e")
        for mx in (10, 45):
            c.rect(mx - 1, cy - 7, 3, 3, "#56687d")
            blade = (11, 5, 1, 5)[i % 4]  # กว้าง -> แคบ -> ขีด -> แคบ = หมุน
            c.rect(mx + 1 - blade // 2 - 1, cy - 8, blade, 1, "#cfe3ef")
        # ขาลงจอด
        for lx in (20, 35):
            c.rect(lx, cy + 6, 1, 3, "#3b4b5e")
        c.rect(17, cy + 9, 7, 1, "#3b4b5e")
        c.rect(32, cy + 9, 7, 1, "#3b4b5e")
        # ลำตัว
        c.ellipse(cx, cy + 1, 11.5, 7.5, "#4fb7e8")
        c.rect(cx - 6, cy + 5, 12, 2, "#e9f6fb")  # ท้องขาว
        blink = i in (11, 12)
        _eye(c, cx - 5, cy, 3, blink=blink)
        _eye(c, cx + 5, cy, 3, blink=blink)
        c.rect(cx, cy + 4, 2, 1, "#14213a")  # ปาก
        # ไฟสถานะ
        if i % 8 < 4:
            c.dot(cx, cy - 6, "#7dffb0")
        c.outline()
        _shadow(c, 28, 33, 9 - bob)
        frames.append(c.image())
    return frames


def sim_frames(n=24):
    """จอคอมตัวอ้วน ข้างในมีโดรนจิ๋วบินวนเลข 8 ทิ้งรอย"""
    frames = []
    path = [(28 + 10 * math.sin(t / n * 2 * math.pi), 14 + 4 * math.sin(t / n * 4 * math.pi)) for t in range(n)]
    for i in range(n):
        c = Canvas()
        # ขาตั้ง
        c.rect(25, 28, 7, 3, "#5b6b7f")
        c.rect(19, 31, 19, 2, "#6f8196")
        # ตัวจอ
        c.rect(11, 3, 35, 26, "#8fa3b8")
        c.rect(11, 3, 35, 1, "#b7c8d8")
        c.rect(11, 27, 35, 2, "#6a7c90")
        c.rect(14, 6, 29, 19, "#0f2b2c", solid=True)
        # ตาราง + scanline
        for gx in range(16, 43, 6):
            c.rect(gx, 6, 1, 19, "#16403e")
        for gy in range(9, 25, 5):
            c.rect(14, gy, 29, 1, "#16403e")
        c.rect(14, 6 + (i * 2) % 19, 29, 1, "#1d5450")
        # รอยบิน
        for k in range(1, 6):
            tx, ty = path[(i - k * 2) % n]
            c.dot(tx, ty, ("#67ddc3", "#4fb3a0", "#3b8a7c", "#2b665c", "#214f48")[k - 1])
        # โดรนจิ๋ว
        x, y = path[i]
        x, y = int(round(x)), int(round(y))
        c.rect(x - 2, y, 5, 1, "#e8fbff")
        c.dot(x - 2, y - 1, "#bfe9f5" if i % 2 else "#ffffff")
        c.dot(x + 2, y - 1, "#bfe9f5" if i % 2 else "#ffffff")
        c.dot(x, y + 1, "#67ddc3")
        # ไฟเปิดเครื่อง
        c.dot(42, 27, "#7dffb0" if i % 12 < 9 else "#2f6b4c")
        c.outline()
        _shadow(c, 28, 34, 11)
        frames.append(c.image())
    return frames


def webcam_frames(n=32):
    """เว็บแคมหัวกลม เลนส์คือตา มองซ้าย-ขวา กะพริบตา ไฟ REC กะพริบ"""
    looks = [0] * 6 + [-1] * 2 + [-2] * 6 + [-1] * 2 + [0] * 4 + [1] * 2 + [2] * 6 + [1] * 4
    frames = []
    for i in range(n):
        c = Canvas()
        cx, cy = 28, 15
        # ขาตั้ง
        c.rect(26, 26, 5, 4, "#5b6b7f")
        c.ellipse(28.5, 31.5, 9, 2.5, "#6f8196")
        # หัว + หูเล็ก
        c.rect(17, 3, 4, 4, "#bfe9f5")
        c.rect(36, 3, 4, 4, "#bfe9f5")
        c.ellipse(cx + .5, cy + .5, 13, 11.5, "#bfe9f5")
        c.rect(cx - 8, cy + 8, 17, 2, "#e9f6fb")  # คางขาว
        # เลนส์ใหญ่ = ตา
        blink = i in (20, 21)
        look = looks[i % len(looks)]
        if blink:
            c.rect(cx - 5, cy + 1, 12, 1, OUTLINE)
        else:
            c.ellipse(cx + .5, cy + .5, 7, 7, "#22324a", shade=False)
            c.ellipse(cx + .5, cy + .5, 5.6, 5.6, "#f4fbff", shade=False)
            c.ellipse(cx + .5 + look, cy + .5, 3.6, 3.6, "#14213a", shade=False)
            c.dot(cx - 1 + look, cy - 2, "#ffffff")
            c.dot(cx + look, cy - 2, "#ffffff")
            c.dot(cx - 1 + look, cy - 1, "#ffffff")
        # ไฟ REC
        if i % 16 < 9:
            c.rect(37, 8, 2, 2, "#ff5a6e")
        else:
            c.rect(37, 8, 2, 2, "#6b2a35")
        c.outline()
        _shadow(c, 28, 34, 10)
        frames.append(c.image())
    return frames


FRAMES = {"drone": drone_frames, "sim": sim_frames, "webcam": webcam_frames}
_cache: dict[str, list] = {}


class PixelSprite(QtWidgets.QWidget):
    """แสดงตัวละครพิกเซลแบบเคลื่อนไหว ขยายเป็นจำนวนเท่าเต็ม (พิกเซลคม)"""

    def __init__(self, kind: str, fps: int = 10, parent=None):
        super().__init__(parent)
        if kind not in _cache:
            _cache[kind] = FRAMES[kind]()
        self.frames = _cache[kind]
        self.index = 0
        self.lively = False
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)  # คลิกทะลุไปที่การ์ด
        self.setMinimumHeight(H * 2)
        self.setFixedHeight(H * 3 + 6)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._interval = int(1000 / fps)

    def set_lively(self, flag: bool):
        """การ์ดที่ถูกเลือก: ขยับเร็วขึ้น"""
        self.lively = flag
        self.timer.setInterval(self._interval // 2 if flag else self._interval)

    def _tick(self):
        self.index = (self.index + 1) % len(self.frames)
        self.update()

    def showEvent(self, event):
        self.timer.start(self._interval // 2 if self.lively else self._interval)
        super().showEvent(event)

    def hideEvent(self, event):  # ไม่เห็น = ไม่ต้องวาด
        self.timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        scale = max(1, min(self.width() // W, self.height() // H))
        w, h = W * scale, H * scale
        target = QtCore.QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)
        p.drawImage(target, self.frames[self.index])
