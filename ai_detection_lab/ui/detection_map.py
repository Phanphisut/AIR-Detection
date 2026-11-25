from __future__ import annotations

import math
from collections import Counter

from PySide6 import QtCore, QtGui, QtWidgets

from .. import config
from ..grid.planner import generate_waypoint_objects
from ..i18n import tr

# Verdict -> marker colour. PENDING means the AI flagged it but no human has
# pressed CONFIRM/REJECT yet, which this project treats as "no ground truth
# exists for this one" rather than as a soft yes (see detection_manager.py's
# design note).
VERDICT_COLORS = {
    "CONFIRMED": "#3ddc97",
    "REJECTED": "#ff5f6d",
    "PENDING": "#ffd166",
}


def build_finds(records: list, decisions: list) -> list[dict]:
    """Collapses a mission's capture log into one entry per target found.

    `records` is detections.json (one row per captured frame, several rows
    can belong to the same target) and `decisions` is decisions.json (the
    operator's CONFIRM/REJECT calls). Both are read straight off disk and
    may be from an older/interrupted mission, so every field is treated as
    untrusted: a malformed row is skipped, never fatal.
    """
    verdicts: dict[object, str] = {}
    for row in decisions or []:
        if not isinstance(row, dict):
            continue
        target_id = row.get("target_id")
        decision = row.get("decision")
        if target_id is not None and decision in ("CONFIRMED", "REJECTED"):
            verdicts[target_id] = decision  # a later decision overrides an earlier one

    grouped: dict[object, dict] = {}
    for index, row in enumerate(records or []):
        if not isinstance(row, dict):
            continue
        survey = row.get("survey_position") or {}
        if not isinstance(survey, dict):
            survey = {}

        target_id = survey.get("target_id")
        key = target_id if target_id is not None else f"capture-{row.get('id', index)}"

        detections = [d for d in (row.get("detections") or []) if isinstance(d, dict)]
        confidences = [
            float(d["confidence"])
            for d in detections
            if isinstance(d.get("confidence"), (int, float))
        ]
        class_names = [str(d["class_name"]) for d in detections if d.get("class_name")]

        find = grouped.get(key)
        if find is None:
            find = {
                "target_id": target_id,
                "key": key,
                "x_cm": _as_number(survey.get("x_cm")),
                "y_cm": _as_number(survey.get("y_cm")),
                "waypoint": survey.get("waypoint"),
                "status": survey.get("status") or "CANDIDATE",
                "image": row.get("image"),
                "captures": 0,
                "best_confidence": 0.0,
                "_classes": Counter(),
            }
            grouped[key] = find

        find["captures"] += 1
        if confidences:
            find["best_confidence"] = max(find["best_confidence"], max(confidences))
        find["_classes"].update(class_names)
        if survey.get("status"):
            find["status"] = survey["status"]  # keep the most recent status seen

    finds = []
    for find in grouped.values():
        classes = find.pop("_classes")
        find["class_name"] = classes.most_common(1)[0][0] if classes else "unknown"
        find["verdict"] = verdicts.get(find["target_id"], "PENDING")
        finds.append(find)

    finds.sort(key=lambda f: (f["target_id"] is None, f["target_id"] if f["target_id"] is not None else 0))
    return finds


def _as_number(value) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def verdict_tally(finds: list[dict]) -> dict:
    tally = {"CONFIRMED": 0, "REJECTED": 0, "PENDING": 0}
    for find in finds:
        tally[find.get("verdict", "PENDING")] += 1
    return tally


class DetectionMapWidget(QtWidgets.QWidget):
    """Plots where on the survey field each target was found.

    The coordinates come from the drone's own survey position at capture
    time (detections.json -> survey_position), which is the only position
    this system knows - the camera looks straight down, so it is the
    drone's spot on the field, not an object-level fix. Several targets
    found from the same spot therefore share one coordinate, and are
    fanned out around it so each stays clickable/readable.
    """

    def __init__(self, finds: list[dict], parent=None):
        super().__init__(parent)
        self.finds = list(finds)
        self.setMinimumHeight(420)
        self.setStyleSheet("background:#071018;border:1px solid #263442;")

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#071018"))

        margin_x, margin_top, margin_bottom = 54, 42, 46
        size = min(self.width() - margin_x * 2, self.height() - margin_top - margin_bottom)
        if size <= 40:
            return
        area = QtCore.QRectF(margin_x, margin_top, size, size)

        max_x = max(1, config.FIELD_WIDTH_CM)
        max_y = max(1, config.FIELD_HEIGHT_CM)

        def pos(x, y):
            # Matches _GridPreviewWidget: x grows EAST/right, y grows
            # SOUTH/down, so north is up and the origin sits top-left.
            return QtCore.QPointF(
                area.left() + (x / max_x) * area.width(),
                area.top() + (y / max_y) * area.height(),
            )

        waypoints = generate_waypoint_objects()

        painter.setPen(QtGui.QPen(QtGui.QColor("#22313d"), 1))
        for x in sorted({w.x_cm for w in waypoints}):
            painter.drawLine(pos(x, 0), pos(x, max_y))
        for y in sorted({w.y_cm for w in waypoints}):
            painter.drawLine(pos(0, y), pos(max_x, y))

        painter.setPen(QtGui.QPen(QtGui.QColor("#7f8c9a"), 2))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(area)

        # Planned route, kept dim - it is context for the finds, not the subject.
        painter.setPen(QtGui.QPen(QtGui.QColor("#2f4a52"), 2))
        for a, b in zip(waypoints, waypoints[1:]):
            painter.drawLine(pos(a.x_cm, a.y_cm), pos(b.x_cm, b.y_cm))

        for wp in waypoints:
            point = pos(wp.x_cm, wp.y_cm)
            painter.setBrush(QtGui.QColor("#16323a"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#3d5f68"), 1))
            painter.drawEllipse(point, 3.5, 3.5)

        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#d8dee9"))
        painter.drawText(14, 24, f"{tr('N')} ↑")
        # The origin is the field's TOP-left (y grows south/down), so label
        # it there - putting "0,0" under the bottom-left corner both names
        # the wrong corner and collides with any find sitting on it.
        painter.drawText(QtCore.QPointF(area.left(), area.top() - 8), "0,0 cm")
        painter.drawText(QtCore.QPointF(area.right() - 74, area.bottom() + 18), f"{max_x},{max_y} cm")

        if not self.finds:
            painter.setPen(QtGui.QColor("#7f8c9a"))
            painter.drawText(area, QtCore.Qt.AlignCenter, tr("No targets found in this mission"))
            self._draw_legend(painter, area)
            painter.end()
            return

        for point, find in self._placed_finds(pos):
            color = QtGui.QColor(VERDICT_COLORS.get(find["verdict"], VERDICT_COLORS["PENDING"]))
            halo = QtGui.QColor(color)
            halo.setAlpha(46)
            radius = 14 + float(find.get("best_confidence", 0.0)) * 16

            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(halo)
            painter.drawEllipse(point, radius, radius)

            painter.setBrush(color)
            painter.setPen(QtGui.QPen(QtGui.QColor("#0b1620"), 2))
            painter.drawEllipse(point, 9, 9)

            label = str(find.get("target_id") if find.get("target_id") is not None else "?")
            font.setPointSize(8)
            painter.setFont(font)
            painter.setPen(QtGui.QColor("#0b1620"))
            painter.drawText(
                QtCore.QRectF(point.x() - 9, point.y() - 9, 18, 18),
                QtCore.Qt.AlignCenter,
                label,
            )

        self._draw_legend(painter, area)
        painter.end()

    def _placed_finds(self, pos):
        """Fans out finds that share one coordinate so none hides another."""
        by_spot: dict[tuple, list[dict]] = {}
        for find in self.finds:
            by_spot.setdefault((find["x_cm"], find["y_cm"]), []).append(find)

        placed = []
        for (x_cm, y_cm), group in by_spot.items():
            center = pos(x_cm, y_cm)
            if len(group) == 1:
                placed.append((center, group[0]))
                continue
            spread = 17.0
            for i, find in enumerate(group):
                angle = (2 * math.pi * i) / len(group) - math.pi / 2
                offset = QtCore.QPointF(math.cos(angle) * spread, math.sin(angle) * spread)
                placed.append((center + offset, find))
        return placed

    def _draw_legend(self, painter, area):
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)

        tally = verdict_tally(self.finds)
        entries = [
            ("CONFIRMED", f"{tr('Confirmed')} {tally['CONFIRMED']}"),
            ("REJECTED", f"{tr('Rejected')} {tally['REJECTED']}"),
            ("PENDING", f"{tr('Not yet reviewed')} {tally['PENDING']}"),
        ]

        x = area.left()
        y = area.bottom() + 38
        for verdict, text in entries:
            painter.setBrush(QtGui.QColor(VERDICT_COLORS[verdict]))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(QtCore.QPointF(x + 5, y - 4), 5, 5)
            painter.setPen(QtGui.QColor("#d8dee9"))
            painter.drawText(QtCore.QPointF(x + 16, y), text)
            x += QtGui.QFontMetrics(font).horizontalAdvance(text) + 44
