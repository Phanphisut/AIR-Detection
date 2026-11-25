"""
drone_backdrop.py
=================
พื้นหลังเคลื่อนไหวจางๆ ธีมโดรน อยู่หลังทุกหน้าของ AIR Detection

  - ตารางจุดพิกเซล (เหมือนแผนที่/เรดาร์)
  - โดรนพิกเซล (ตัวเดียวกับหน้า Connect) หลายขนาดลอยผ่านจอช้าๆ
  - โดรนบินสำรวจแบบซิกแซ็ก (เหมือนภารกิจ Grid) ตามเส้นประ ทิ้งรอย
  - วงเรดาร์กระจายออกเป็นระยะ

ทุกอย่างวาดด้วยความทึบต่ำ ให้เห็นแค่ในที่ว่าง (การ์ด/วิดีโอทึบอยู่ด้านหน้า)
หยุดวาดเองเมื่อหน้าต่างถูกย่อ  ปิดได้ด้วย AIR_BACKDROP=0
"""
from __future__ import annotations

import math
import os
import random

from PySide6 import QtCore, QtGui, QtWidgets

BG = QtGui.QColor("#0b1118")
INK = QtGui.QColor("#6ff0d8")
FPS = 15


def _flip(img: QtGui.QImage) -> QtGui.QImage:
    if hasattr(img, "flipped"):  # Qt 6.9+
        return img.flipped(QtCore.Qt.Horizontal)
    return img.mirrored(True, False)


class _Flyer:
    """โดรนลอยข้ามจอ: ทิศ ความเร็ว ขนาด ต่างกันไป"""

    def __init__(self, rng: random.Random, w, h, scale, first=False):
        self.scale = scale
        self.dir = rng.choice((-1, 1))
        self.speed = rng.uniform(10, 22) / scale * 2  # ตัวเล็ก (ไกล) ช้ากว่า
        self.y = rng.uniform(0.08, 0.92)
        self.phase = rng.uniform(0, math.tau)
        span = w + 56 * scale * 2
        self.x = rng.uniform(0, span) if first else (0 if self.dir > 0 else span)
        self.frame0 = rng.randrange(16)
        self.alpha = rng.uniform(0.07, 0.13) * (0.7 + scale / 8)


class DroneBackdrop(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)
        self.enabled = os.environ.get("AIR_BACKDROP", "1").strip() not in ("0", "false", "off")
        self.t = 0.0
        self.rng = random.Random(7)
        self.flyers: list[_Flyer] = []
        self.pings: list[list[float]] = []  # [x, y, age]
        self._grid = None
        self._frames = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        if self.enabled:
            self.timer.start(1000 // FPS)

    # ---------------------------------------------------------- state
    def _ensure(self):
        if self._frames is None:
            from ai_detection_lab.fly.sprites import FRAMES, _cache

            if "drone" not in _cache:
                _cache["drone"] = FRAMES["drone"]()
            self._frames = _cache["drone"]
            self._mirrored = [_flip(img) for img in self._frames]
        if not self.flyers and self.width() > 0:
            for scale in (2, 2, 3, 3, 4, 5):
                self.flyers.append(_Flyer(self.rng, self.width(), self.height(), scale, first=True))

    def _tick(self):
        if not self.isVisible() or self.window().isMinimized():
            return
        dt = 1 / FPS
        self.t += dt
        w = self.width()
        for i, f in enumerate(self.flyers):
            f.x += f.dir * f.speed * dt * 3
            span = w + 56 * f.scale * 2
            if (f.dir > 0 and f.x > span) or (f.dir < 0 and f.x < 0):
                self.flyers[i] = _Flyer(self.rng, w, self.height(), f.scale)
        for p in self.pings:
            p[2] += dt
        self.pings = [p for p in self.pings if p[2] < 4.0]
        if self.rng.random() < dt / 2.2:  # ราวๆ ทุก 2 วินาที
            self.pings.append([self.rng.uniform(.1, .9), self.rng.uniform(.15, .9), 0.0])
        self.update()

    # ---------------------------------------------------------- painting
    def _grid_pixmap(self):
        size = self.size()
        if self._grid is not None and self._grid.size() == size:
            return self._grid
        pm = QtGui.QPixmap(size)
        pm.fill(BG)
        p = QtGui.QPainter(pm)
        dot = QtGui.QColor(INK)
        dot.setAlphaF(0.06)
        major = QtGui.QColor(INK)
        major.setAlphaF(0.10)
        step = 24
        for y in range(step, size.height(), step):
            for x in range(step, size.width(), step):
                big = (x // step) % 5 == 0 and (y // step) % 5 == 0
                p.fillRect(x - 1, y - 1, 3 if big else 2, 3 if big else 2, major if big else dot)
        p.end()
        self._grid = pm
        return pm

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        if not self.enabled:
            p.fillRect(self.rect(), BG)
            return
        self._ensure()
        p.drawPixmap(0, 0, self._grid_pixmap())
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        self._paint_pings(p)
        self._paint_survey(p)
        self._paint_flyers(p)

    def _paint_pings(self, p):
        w, h = self.width(), self.height()
        for x, y, age in self.pings:
            for k in range(3):  # 3 วงซ้อน ห่างกันนิดหน่อย
                a = age - k * 0.35
                if a <= 0:
                    continue
                r = int(a * 60) // 4 * 4  # รัศมีขยับทีละ 4px = ดูเป็นพิกเซล
                c = QtGui.QColor(INK)
                c.setAlphaF(max(0.0, 0.13 * (1 - a / 3.6)))
                p.setPen(QtGui.QPen(c, 2))
                p.setBrush(QtCore.Qt.NoBrush)
                p.drawEllipse(QtCore.QPoint(int(x * w), int(y * h)), r, r)

    def _survey_route(self):
        """เส้นทางซิกแซ็กมุมขวาล่าง (เหมือนภารกิจบินสำรวจ Grid)"""
        w, h = self.width(), self.height()
        rw, rh = min(420, int(w * 0.32)), min(260, int(h * 0.3))
        x0, y0 = w - rw - 60, h - rh - 50
        lanes = 5
        pts = []
        for i in range(lanes):
            y = y0 + rh * i // (lanes - 1)
            xs = (x0, x0 + rw) if i % 2 == 0 else (x0 + rw, x0)
            pts += [QtCore.QPointF(xs[0], y), QtCore.QPointF(xs[1], y)]
        return pts

    def _paint_survey(self, p):
        pts = self._survey_route()
        if len(pts) < 2:
            return
        segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        lengths = [math.hypot(b.x() - a.x(), b.y() - a.y()) for a, b in segs]
        total = sum(lengths)
        # เส้นประ (สี่เหลี่ยมพิกเซลทุก 14px)
        c = QtGui.QColor(INK)
        c.setAlphaF(0.07)
        for (a, b), L in zip(segs, lengths):
            n = max(1, int(L // 14))
            for k in range(n + 1):
                q = a + (b - a) * (k / n)
                p.fillRect(int(q.x()) - 1, int(q.y()) - 1, 3, 3, c)
        for q in pts:  # จุดเลี้ยว
            p.fillRect(int(q.x()) - 3, int(q.y()) - 3, 6, 6, c)

        def at(dist):
            dist %= total
            for (a, b), L in zip(segs, lengths):
                if dist <= L:
                    return a + (b - a) * (dist / L)
                dist -= L
            return pts[-1]

        head = self.t * 55
        trail = QtGui.QColor(INK)
        for k in range(1, 14):  # รอยที่บินผ่าน
            q = at(head - k * 10)
            trail.setAlphaF(0.16 * (1 - k / 14))
            p.fillRect(int(q.x()) - 2, int(q.y()) - 2, 4, 4, trail)
        q = at(head)
        img = self._frames[int(self.t * 12) % len(self._frames)]
        p.setOpacity(0.22)
        x, y = int(q.x()) // 2 * 2, int(q.y()) // 2 * 2
        p.drawImage(QtCore.QRect(x - 56, y - 44, 112, 72), img)  # ขยาย 2 เท่า กลางตัวอยู่บนเส้นทาง
        # กรวยกล้องสแกนใต้โดรน
        cone = QtGui.QColor(INK)
        cone.setAlphaF(0.35)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(cone)
        for i in range(4):
            p.drawRect(int(q.x()) - 4 - i * 4, int(q.y()) + 10 + i * 4, 8 + i * 8, 2)
        p.setOpacity(1.0)

    def _paint_flyers(self, p):
        h = self.height()
        for f in self.flyers:
            img = self._frames[(int(self.t * 10) + f.frame0) % len(self._frames)]
            s = f.scale
            x = int(f.x) - 56 * s
            y = int(f.y * h + math.sin(self.t * 1.3 + f.phase) * 6 * s)
            x, y = x // s * s, y // s * s  # ยึดตามกริดพิกเซลของขนาดนั้น
            p.setOpacity(f.alpha)
            if f.dir < 0:
                img = self._mirrored[(int(self.t * 10) + f.frame0) % len(self._frames)]
            p.drawImage(QtCore.QRect(x, y, 56 * s, 36 * s), img)
        p.setOpacity(1.0)

    def resizeEvent(self, event):
        self._grid = None
        super().resizeEvent(event)
