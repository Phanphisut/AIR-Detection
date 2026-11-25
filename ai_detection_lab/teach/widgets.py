"""ชิ้นส่วนหน้าจอของ "สอน AI": ภาพพร้อมกรอบ, แถบเวลาสี"""

from __future__ import annotations

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

MIN_BOX_FRACTION = 0.02     # กรอบเล็กกว่านี้น่าจะเผลอคลิก ไม่นับ
GRAB_PIXELS = 9             # คลิกห่างมุม/ขอบกรอบไม่เกินเท่านี้ถือว่าจับกรอบ
HANDLE_PIXELS = 5           # ขนาดจุดจับครึ่งหนึ่ง
RESIZE_CURSORS = {
    "tl": QtCore.Qt.SizeFDiagCursor, "br": QtCore.Qt.SizeFDiagCursor,
    "tr": QtCore.Qt.SizeBDiagCursor, "bl": QtCore.Qt.SizeBDiagCursor,
    "l": QtCore.Qt.SizeHorCursor, "r": QtCore.Qt.SizeHorCursor,
    "t": QtCore.Qt.SizeVerCursor, "b": QtCore.Qt.SizeVerCursor,
    "move": QtCore.Qt.SizeAllCursor,
}

STATE_COLORS = {
    "drawn": "#3ec98a",
    "none": "#5b6b7a",
    "unset": "#26323e",
    "correction": "#3ec98a",
    "found": "#3ec98a",
    "miss": "#e5534b",
    "wrong": "#e9a23b",
    "empty": "#5b6b7a",
}


def bgr_to_pixmap(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    image = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(image)


class FrameView(QtWidgets.QWidget):
    """แสดงรูป + กรอบ คลิกที่กรอบได้ หรือเมื่อ draw_enabled: ลากพื้นที่ว่างเพื่อวาดใหม่ ลากมุม/ขอบเพื่อปรับขนาด ลากกลางกรอบเพื่อย้าย"""
    box_drawn = QtCore.Signal(list)     # [x1, y1, x2, y2] แบบ 0-1 ทุกครั้งที่วาดหรือปรับกรอบเสร็จ
    box_clicked = QtCore.Signal(int)    # ลำดับกรอบที่ถูกคลิก

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 260)
        self.setCursor(QtCore.Qt.ArrowCursor)
        self._pixmap = None
        self._boxes = []        # dict(box, color, label, dashed)
        self._start = None
        self._now = None
        self._drag = None       # dict(part, start, box) ตอนกำลังปรับกรอบเดิม
        self.draw_enabled = False
        self.setMouseTracking(True)  # เปลี่ยนรูปเมาส์เมื่อชี้โดนมุม/ขอบ
        self.placeholder = ""

    def set_image(self, bgr):
        self._pixmap = None if bgr is None else bgr_to_pixmap(bgr)
        self.update()

    def set_boxes(self, boxes):
        self._boxes = boxes
        self.update()

    def set_draw_enabled(self, flag):
        self.draw_enabled = flag
        self.setCursor(QtCore.Qt.CrossCursor if flag else QtCore.Qt.ArrowCursor)

    def _image_rect(self):
        if self._pixmap is None:
            return QtCore.QRectF()
        size = self._pixmap.size().scaled(self.size(), QtCore.Qt.KeepAspectRatio)
        return QtCore.QRectF((self.width() - size.width()) / 2, (self.height() - size.height()) / 2,
                             size.width(), size.height())

    def _to_norm(self, pos):
        r = self._image_rect()
        return (min(1.0, max(0.0, (pos.x() - r.x()) / r.width())),
                min(1.0, max(0.0, (pos.y() - r.y()) / r.height())))

    def _box_rect(self, box):
        r = self._image_rect()
        return QtCore.QRectF(r.x() + box[0] * r.width(), r.y() + box[1] * r.height(),
                             (box[2] - box[0]) * r.width(), (box[3] - box[1]) * r.height())

    def _grab_part(self, pos):
        """ส่วนของกรอบแรกที่เมาส์ชี้: tl/tr/bl/br, l/r/t/b, move หรือ None"""
        if not self.draw_enabled or not self._boxes or self._pixmap is None:
            return None
        rect = self._box_rect(self._boxes[0]["box"])
        x, y = pos.x(), pos.y()
        if not rect.adjusted(-GRAB_PIXELS, -GRAB_PIXELS, GRAB_PIXELS, GRAB_PIXELS).contains(pos):
            return None
        v = "t" if abs(y - rect.top()) <= GRAB_PIXELS else "b" if abs(y - rect.bottom()) <= GRAB_PIXELS else ""
        h = "l" if abs(x - rect.left()) <= GRAB_PIXELS else "r" if abs(x - rect.right()) <= GRAB_PIXELS else ""
        return (v + h) or "move"

    def _apply_drag(self, pos):
        x, y = self._to_norm(pos)
        dx, dy = x - self._drag["start"][0], y - self._drag["start"][1]
        x1, y1, x2, y2 = self._drag["box"]
        part = self._drag["part"]
        if part == "move":
            dx = max(-x1, min(1.0 - x2, dx))
            dy = max(-y1, min(1.0 - y2, dy))
            box = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        else:
            if "l" in part:
                x1 += dx
            if "r" in part:
                x2 += dx
            if "t" in part:
                y1 += dy
            if "b" in part:
                y2 += dy
            # ลากเลยขอบอีกด้านได้ กรอบกลับด้านแทนที่จะค้าง
            box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            box = [max(0.0, box[0]), max(0.0, box[1]), min(1.0, box[2]), min(1.0, box[3])]
        self._boxes[0]["box"] = box
        self.update()

    def mousePressEvent(self, event):
        if self._pixmap is None or event.button() != QtCore.Qt.LeftButton:
            return
        if self.draw_enabled:
            part = self._grab_part(event.position())
            if part is not None:
                self._drag = dict(part=part, start=self._to_norm(event.position()),
                                  box=list(self._boxes[0]["box"]))
            else:
                self._start = self._now = self._to_norm(event.position())
        else:
            x, y = self._to_norm(event.position())
            hits = [i for i, b in enumerate(self._boxes)
                    if b.get("clickable") and b["box"][0] <= x <= b["box"][2] and b["box"][1] <= y <= b["box"][3]]
            if hits:  # กรอบซ้อนกัน: เอากรอบเล็กสุดที่คลิกโดน
                self.box_clicked.emit(min(hits, key=lambda i: (self._boxes[i]["box"][2] - self._boxes[i]["box"][0])
                                          * (self._boxes[i]["box"][3] - self._boxes[i]["box"][1])))

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            self._apply_drag(event.position())
        elif self._start is not None:
            self._now = self._to_norm(event.position())
            self.update()
        elif self.draw_enabled:
            part = self._grab_part(event.position())
            self.setCursor(RESIZE_CURSORS.get(part, QtCore.Qt.CrossCursor))

    def mouseReleaseEvent(self, event):
        if self._drag is not None:
            original, self._drag = self._drag["box"], None
            box = self._boxes[0]["box"]
            if box[2] - box[0] < MIN_BOX_FRACTION or box[3] - box[1] < MIN_BOX_FRACTION:
                self._boxes[0]["box"] = original  # บีบจนเล็กเกินไป กลับค่าเดิม
                self.update()
            elif box != original:
                self.box_drawn.emit(list(box))
            return
        if self._start is None:
            return
        x1, x2 = sorted((self._start[0], self._now[0]))
        y1, y2 = sorted((self._start[1], self._now[1]))
        self._start = self._now = None
        self.update()
        if x2 - x1 >= MIN_BOX_FRACTION and y2 - y1 >= MIN_BOX_FRACTION:
            self.box_drawn.emit([x1, y1, x2, y2])

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#080d13"))
        if self._pixmap is None:
            p.setPen(QtGui.QColor("#8ea0b2"))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, self.placeholder)
            return
        r = self._image_rect()
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        p.drawPixmap(r.toRect(), self._pixmap)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        font = p.font()
        font.setFamilies(["Leelawadee UI", "Leelawadee", "Tahoma"])
        font.setPixelSize(15)
        font.setBold(True)
        p.setFont(font)
        drawing_new = self._start is not None
        for item in self._boxes:
            if drawing_new and self.draw_enabled:
                break  # กำลังลากกรอบใหม่แทนกรอบเดิม
            rect = self._box_rect(item["box"])
            pen = QtGui.QPen(QtGui.QColor(item.get("color", "#6ff0d8")), 3)
            if item.get("dashed"):
                pen.setStyle(QtCore.Qt.DashLine)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(rect)
            label = item.get("label")
            if label:
                width = p.fontMetrics().horizontalAdvance(label) + 12
                top = max(r.y(), rect.y() - 24)
                tag = QtCore.QRectF(rect.x(), top, width, 22)
                p.fillRect(tag, QtGui.QColor(item.get("color", "#6ff0d8")))
                p.setPen(QtGui.QColor("#07221d"))
                p.drawText(tag, QtCore.Qt.AlignCenter, label)
            if self.draw_enabled:
                p.setPen(QtGui.QPen(QtGui.QColor("#07221d"), 1))
                p.setBrush(QtGui.QColor(item.get("color", "#6ff0d8")))
                cx, cy = rect.center().x(), rect.center().y()
                for hx, hy in ((rect.left(), rect.top()), (cx, rect.top()), (rect.right(), rect.top()),
                               (rect.left(), cy), (rect.right(), cy),
                               (rect.left(), rect.bottom()), (cx, rect.bottom()), (rect.right(), rect.bottom())):
                    p.drawRect(QtCore.QRectF(hx - HANDLE_PIXELS, hy - HANDLE_PIXELS,
                                             HANDLE_PIXELS * 2, HANDLE_PIXELS * 2))
        if self._start is not None:
            x1, x2 = sorted((self._start[0], self._now[0]))
            y1, y2 = sorted((self._start[1], self._now[1]))
            p.setPen(QtGui.QPen(QtGui.QColor("#ffe27a"), 3, QtCore.Qt.DashLine))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(self._box_rect((x1, y1, x2, y2)))


class TimelineBar(QtWidgets.QWidget):
    """แถบเวลา: 1 ช่องต่อ 1 รูป สีบอกสถานะ คลิก/ลากเพื่อเลื่อนไปรูปนั้น"""
    picked = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._cells = []
        self._current = -1

    def set_cells(self, cells):
        self._cells = list(cells)
        self.update()

    def set_current(self, index):
        self._current = index
        self.update()

    def _pick(self, x):
        if self._cells:
            self.picked.emit(max(0, min(len(self._cells) - 1, int(x / max(1, self.width()) * len(self._cells)))))

    def mousePressEvent(self, event):
        self._pick(event.position().x())

    def mouseMoveEvent(self, event):
        self._pick(event.position().x())

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#141d27"))
        n = len(self._cells)
        if not n:
            return
        cell = self.width() / n
        for i, state in enumerate(self._cells):
            x1, x2 = int(i * cell), int((i + 1) * cell)
            p.fillRect(x1, 4, max(1, x2 - x1), self.height() - 8, QtGui.QColor(STATE_COLORS.get(state, "#26323e")))
        if 0 <= self._current < n:
            x = int((self._current + 0.5) * cell)
            p.fillRect(x - 2, 0, 4, self.height(), QtGui.QColor("#f4f8fb"))
