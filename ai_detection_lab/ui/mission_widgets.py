"""Small vector-drawn widgets; no image assets or AI imports at startup."""
from PySide6 import QtCore, QtGui, QtWidgets, QtSvg
from .pixel_theme import draw_pixel_rect

PATHS = {
    "dashboard": '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    "route": '<path d="M5 19V5h14v7H9v7h10"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/>',
    "train": '<path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
    "model": '<path d="m12 3 9 5v9l-9 5-9-5V8l9-5Zm0 10v9M3 8l9 5 9-5M7.5 5.5l9 5"/>',
    "report": '<path d="M14 3H5v18h14V8l-5-5Zm0 0v5h5M8 17v-4m4 4V10m4 7v-6"/>',
    "code": '<path d="m8 7-5 5 5 5m8-10 5 5-5 5m-3-13-2 20"/>',
    "camera": '<rect x="3" y="6" width="14" height="13" rx="3"/><path d="m17 10 4-3v11l-4-3M7 6l2-3h4l2 3"/>',
    "expand": '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m8 0h5v-5"/>',
    "play": '<path d="m8 4 12 8-12 8V4Z"/>',
    "stop": '<rect x="5" y="5" width="14" height="14" rx="2"/>',
    "shield": '<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Zm0 5v5m0 3v1"/>',
    "photo": '<rect x="3" y="5" width="18" height="16" rx="2"/><circle cx="12" cy="13" r="4"/><path d="M8 5V3h8v2"/>',
    "drone": '<path d="m8 8 8 8M8 16l8-8"/><circle cx="6" cy="6" r="4"/><circle cx="18" cy="6" r="4"/><circle cx="6" cy="18" r="4"/><circle cx="18" cy="18" r="4"/>',
    "folder": '<path d="M3 7V4h6l3 3h9v13H3V7Z"/>',
    "keyboard": '<rect x="2" y="5" width="20" height="14" rx="2"/><path d="M6 9h1m4 0h1m4 0h1M6 12h1m4 0h1m4 0h1M7 16h10"/>',
}


def icon(name, color="#9bb3c9"):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">'
           f'<g fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
           f'{PATHS[name]}</g></svg>')
    pix = QtGui.QPixmap(48, 48)
    pix.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pix)
    QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode())).render(painter)
    painter.end()
    pix.setDevicePixelRatio(2)
    return QtGui.QIcon(pix)


class ElidedLabel(QtWidgets.QLabel):
    """Keep transient log/model text from widening the whole dashboard."""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMinimumWidth(0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(self.palette().color(QtGui.QPalette.WindowText))
        painter.drawText(self.rect(), self.alignment(), self.fontMetrics().elidedText(
            self.text(), QtCore.Qt.ElideRight, self.width()))


class VideoSurface(QtWidgets.QLabel):
    """Actual video or an explicit offline state, with a stable size hint."""
    def __init__(self, title, hint, parent=None):
        super().__init__(parent)
        self.empty_title, self.empty_hint = title, hint
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(260, 110)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.setText(title)

    def sizeHint(self):
        return QtCore.QSize(640, 360)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor("#09121b"))
        pix = self.pixmap()
        if pix is not None and not pix.isNull():
            size = pix.size().scaled(self.size(), QtCore.Qt.KeepAspectRatio)
            area = QtCore.QRect(QtCore.QPoint(0, 0), size)
            area.moveCenter(self.rect().center())
            p.drawPixmap(area, pix)
            return
        center = QtCore.QPointF(self.width() / 2, self.height() / 2)
        glow = QtGui.QRadialGradient(center, max(self.width(), self.height()) * .65)
        glow.setColorAt(0, QtGui.QColor("#153135"))
        glow.setColorAt(1, QtGui.QColor("#09121b"))
        p.fillRect(self.rect(), glow)
        p.setPen(QtGui.QPen(QtGui.QColor(83, 119, 131, 22), 1))
        for x in range(0, self.width(), 32):
            p.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 32):
            p.drawLine(0, y, self.width(), y)
        cx, cy = int(center.x()), int(center.y()) - 36
        if self.height() < 200:
            p.setPen(QtGui.QColor("#d1e9e7"))
            font = self.font()
            font.setPixelSize(16)
            p.setFont(font)
            p.drawText(self.rect().adjusted(14, 10, -14, -10),
                       QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, self.text())
            return
        p.setBrush(QtGui.QColor("#19383d"))
        p.setPen(QtGui.QPen(QtGui.QColor("#30585c"), 1))
        draw_pixel_rect(p, QtCore.QRectF(cx - 32, cy - 32, 64, 64), 18)
        icon("camera", "#7fdccc").paint(p, QtCore.QRect(cx - 15, cy - 15, 30, 30))
        p.setPen(QtGui.QColor("#e2f1f1"))
        font = self.font()
        font.setPixelSize(19)
        font.setWeight(QtGui.QFont.DemiBold)
        p.setFont(font)
        p.drawText(QtCore.QRect(12, cy + 49, self.width() - 24, 32), QtCore.Qt.AlignCenter, self.text())
        font.setPixelSize(12)
        font.setWeight(QtGui.QFont.Normal)
        p.setFont(font)
        p.setPen(QtGui.QColor("#8da9b5"))
        p.drawText(QtCore.QRect(24, cy + 85, self.width() - 48, 48),
                   QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop | QtCore.Qt.TextWordWrap, self.empty_hint)
        p.setPen(QtGui.QPen(QtGui.QColor("#3b6670"), 1.5))
        m, n = 22, 16
        for x, dx in ((m, 1), (self.width() - m, -1)):
            for y, dy in ((m, 1), (self.height() - m, -1)):
                p.drawLine(x, y, x + dx * n, y)
                p.drawLine(x, y, x, y + dy * n)
