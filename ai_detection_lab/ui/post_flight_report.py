from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from .pixel_theme import draw_pixel_rect

from .detection_map import DetectionMapWidget, VERDICT_COLORS, build_finds
from ..i18n import tr


class PostFlightReportDialog(QtWidgets.QDialog):

    def __init__(self, capture_manager, parent=None):
        super().__init__(parent)

        self.capture = capture_manager
        self.setWindowTitle(tr("Post Flight Report"))

        # Give the dialog a real window frame — minimize + maximize + the
        # normal resize border/corner-grip — instead of the bare fixed-size
        # popup a plain QDialog gets by default on some platforms.
        self.setWindowFlags(
            QtCore.Qt.Window
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
            | QtCore.Qt.WindowCloseButtonHint
        )
        self.setSizeGripEnabled(True)
        self.setMinimumSize(900, 620)
        self.resize(1180, 780)

        from .panel_theme import DIALOG_STYLE
        self.setStyleSheet(DIALOG_STYLE)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(14)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(12)

        title_box = QtWidgets.QVBoxLayout()
        title_box.setSpacing(2)

        title = QtWidgets.QLabel(tr("POST FLIGHT REPORT"))
        title.setStyleSheet("font-size: 25px; font-weight: 800; color: #f2f7fa;")
        subtitle = QtWidgets.QLabel(tr("Mission detection summary and captured evidence"))
        subtitle.setStyleSheet("font-size: 12px; color: #8293a1;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        header.addLayout(title_box)
        header.addStretch()

        records = self._load_records()

        mission_badge = QtWidgets.QLabel(f"{len(records)} {tr('CAPTURED FRAMES')}")
        mission_badge.setAlignment(QtCore.Qt.AlignCenter)
        mission_badge.setStyleSheet("""
            background: #13251f;
            color: #8ce8c0;
            border: 1px solid #285746;
            border-radius: 15px;
            padding: 7px 12px;
            font-size: 11px;
            font-weight: 700;
        """)
        header.addWidget(mission_badge)
        root.addLayout(header)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)

        # Summary is kept compact so it does not consume the whole report.
        summary_widget = self._build_summary(records)
        content_layout.addWidget(summary_widget)

        # Where things were found. Every capture already carries the drone's
        # survey position, so this needs no new data collection - it just
        # draws what detections.json has always held.
        finds = build_finds(records, self._load_decisions())
        content_layout.addWidget(self._build_detection_map(finds))

        gallery = self._build_evidence_gallery(records)

        scroll_all = QtWidgets.QScrollArea()
        scroll_all.setWidgetResizable(True)
        scroll_all.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_all.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_all.setWidget(content)
        report_tabs = QtWidgets.QTabWidget()
        report_tabs.addTab(scroll_all, tr("Mission overview"))
        report_tabs.addTab(gallery, tr("Captured evidence") + f" ({len(records)})")
        root.addWidget(report_tabs, 1)

        btns = QtWidgets.QHBoxLayout()
        btns.setSpacing(10)

        open_btn = QtWidgets.QPushButton(tr("📁  Open Capture Folder"))
        open_btn.clicked.connect(self._open_folder)

        close_btn = QtWidgets.QPushButton(tr("Close"))
        close_btn.clicked.connect(self.accept)

        btns.addWidget(open_btn)
        btns.addStretch()
        btns.addWidget(close_btn)
        root.addLayout(btns)

    def _load_records(self):

        if not self.capture.records.exists():
            return []

        try:
            return json.loads(
                self.capture.records.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return []

    def _load_decisions(self):
        """The operator's CONFIRM/REJECT log. Absent for a mission where
        nobody reviewed anything, which is normal, not an error."""
        decisions_path = getattr(self.capture, "decisions", None)
        if decisions_path is None or not decisions_path.exists():
            return []
        try:
            return json.loads(decisions_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _build_detection_map(self, finds):
        frame = QtWidgets.QFrame()
        frame.setObjectName("reportCard")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        heading = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(tr("DETECTION MAP"))
        title.setObjectName("reportSection")
        heading.addWidget(title)
        heading.addStretch()
        count = QtWidgets.QLabel(f"{len(finds)} {tr('TARGET') if len(finds) == 1 else tr('TARGETS')}")
        count.setObjectName("reportMuted")
        heading.addWidget(count)
        layout.addLayout(heading)

        caption = QtWidgets.QLabel(tr(
            "Where each target was found on the survey field — coordinates come "
            "from the drone's own position at capture time (camera facing "
            "straight down); circle size follows that target's highest "
            "confidence score."
        ))
        caption.setObjectName("reportMuted")
        caption.setWordWrap(True)
        layout.addWidget(caption)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(14)
        body.addWidget(DetectionMapWidget(finds), 3)

        side = QtWidgets.QVBoxLayout()
        side.setSpacing(8)
        if finds:
            for find in finds[:6]:
                side.addWidget(self._build_find_row(find))
            if len(finds) > 6:
                more = QtWidgets.QLabel(f"…{tr('and')} {len(finds) - 6} {tr('more target(s)')}")
                more.setObjectName("reportMuted")
                side.addWidget(more)
        else:
            empty = QtWidgets.QLabel(tr(
                "No targets recorded for this mission yet.\n"
                "Try flying again with the AI Camera running the whole flight."
            ))
            empty.setObjectName("reportMuted")
            empty.setWordWrap(True)
            side.addWidget(empty)
        side.addStretch()
        body.addLayout(side, 2)

        layout.addLayout(body)
        return frame

    def _build_find_row(self, find):
        color = VERDICT_COLORS.get(find.get("verdict", "PENDING"), VERDICT_COLORS["PENDING"])
        row = QtWidgets.QFrame()
        row.setObjectName("findRow")
        # Scoped to QFrame#findRow on purpose: an unscoped "border:" here
        # cascades into every child QLabel, which draws a box around each
        # line of text instead of one box around the row.
        row.setStyleSheet(
            "QFrame#findRow {"
            "background:#0d1a24; border:1px solid #243441;"
            f"border-left:3px solid {color}; border-radius:6px;"
            "}"
        )
        layout = QtWidgets.QVBoxLayout(row)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(1)

        target_id = find.get("target_id")
        headline = QtWidgets.QLabel(
            f"#{target_id if target_id is not None else '?'}  {find.get('class_name') or tr('unknown')}"
            f"   {float(find.get('best_confidence', 0.0)):.0%}"
        )
        headline.setStyleSheet("color:#f2f7fa; font-size:13px; font-weight:700;")
        layout.addWidget(headline)

        where = QtWidgets.QLabel(
            f"X={int(find.get('x_cm', 0))}  Y={int(find.get('y_cm', 0))} cm"
            + (f"  •  WP {find['waypoint']}" if find.get("waypoint") else "")
            + f"  •  {find.get('captures', 0)} {tr('image(s)')}"
        )
        where.setObjectName("reportMuted")
        layout.addWidget(where)

        verdict = QtWidgets.QLabel(tr(find.get("verdict", "PENDING")))
        verdict.setStyleSheet(f"color:{color}; font-size:11px; font-weight:700;")
        layout.addWidget(verdict)
        return row

    def _build_summary(self, records):
        frame = QtWidgets.QFrame()
        frame.setObjectName("reportCard")
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        counter = Counter()
        total_detections = 0
        best_conf = 0.0

        for rec in records:
            for det in rec.get("detections", []):
                total_detections += 1
                counter[det.get("class_name") or tr("unknown")] += 1
                if det.get("confidence") is not None:
                    best_conf = max(best_conf, float(det["confidence"]))

        def metric(title, value, note):
            box = QtWidgets.QFrame()
            box.setObjectName("reportMetric")
            box.setStyleSheet("QFrame#reportMetric {background:#101925; border:1px solid #29394b; border-radius:9px;}")
            v = QtWidgets.QVBoxLayout(box)
            v.setContentsMargins(14, 10, 14, 10)
            v.setSpacing(2)

            t = QtWidgets.QLabel(title.upper())
            t.setObjectName("reportSection")
            val = QtWidgets.QLabel(value)
            val.setObjectName("reportValue")
            n = QtWidgets.QLabel(note)
            n.setObjectName("reportMuted")
            v.addWidget(t)
            v.addWidget(val)
            v.addWidget(n)
            return box

        layout.addWidget(metric(tr("Images"), str(len(records)), tr("frames captured")))
        layout.addWidget(metric(tr("Detections"), str(total_detections), tr("AI detections recorded")))
        layout.addWidget(metric(tr("Best Confidence"), f"{best_conf:.1%}" if best_conf else "--", tr("highest model confidence")))

        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(2)
        classes = ", ".join(f"{name} ×{count}" for name, count in counter.most_common()) or tr("No detections")
        classes_label = QtWidgets.QLabel(classes)
        classes_label.setWordWrap(True)
        classes_label.setStyleSheet("font-size: 12px; color: #a9b8c3; padding: 4px 6px;")
        right_col.addWidget(classes_label)

        # Confidence is the AI model's own certainty score for a detection -
        # not a verified-correct score. The only verified/ground-truth
        # signal in this app is the operator's CONFIRM/REJECT decision made
        # live during the flight (see the Current Target panel on the main
        # dashboard), which this report does not second-guess.
        note = QtWidgets.QLabel(tr(
            "Confidence is the AI's own certainty for each detection, not a verified-correct "
            "score - use the CONFIRM/REJECT decisions made during the flight for verified accuracy."
        ))
        note.setWordWrap(True)
        note.setObjectName("reportMuted")
        note.setStyleSheet("padding: 0 6px;")
        right_col.addWidget(note)
        right_col.addStretch()

        layout.addLayout(right_col, 1)

        return frame

    @staticmethod
    def _stamp_confidence(pixmap: QtGui.QPixmap, confidence: float) -> QtGui.QPixmap:
        """Burn a small confidence badge into the top-left corner of a
        thumbnail so the score is visible on the image itself, not just
        in the caption underneath it."""
        stamped = QtGui.QPixmap(pixmap)
        painter = QtGui.QPainter(stamped)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        text = f"{confidence:.0%}"
        font = painter.font()
        font.setBold(True)
        font.setPointSize(11)
        painter.setFont(font)

        metrics = QtGui.QFontMetrics(font)
        pad = 6
        badge_w = metrics.horizontalAdvance(text) + pad * 2
        badge_h = metrics.height() + pad

        painter.setBrush(QtGui.QColor(10, 20, 15, 210))
        painter.setPen(QtCore.Qt.NoPen)
        draw_pixel_rect(painter, QtCore.QRectF(6, 6, badge_w, badge_h), 5)

        painter.setPen(QtGui.QColor("#8ce8c0"))
        painter.drawText(6 + pad, 6 + badge_h - pad // 2, text)

        painter.end()
        return stamped

    def _build_evidence_gallery(self, records):
        """Every captured image that has at least one detection, ranked by
        its strongest detection's confidence (highest first) - the score
        is burned into each thumbnail so it reads at a glance. This is the
        one evidence view in the report: for judging a mission (most
        objects found, most confident catches) you want to see everything
        that was captured, not just a top-3 highlight reel or a single
        image picked by an arbitrary confidence target.
        """
        frame = QtWidgets.QFrame()
        frame.setObjectName("reportCard")
        outer = QtWidgets.QVBoxLayout(frame)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(8)

        best = []
        for rec in records:
            detections = rec.get("detections", [])
            if not detections:
                continue
            top_det = max(detections, key=lambda d: float(d.get("confidence", 0) or 0))
            best.append((float(top_det.get("confidence", 0) or 0), top_det, rec))
        best.sort(key=lambda x: x[0], reverse=True)

        title = QtWidgets.QLabel(f"{tr('CAPTURED EVIDENCE')}  •  {len(best)} {tr('image(s) with a detection')}")
        title.setObjectName("reportSection")
        outer.addWidget(title)

        if not best:
            empty = QtWidgets.QLabel(tr("No detections recorded this mission."))
            empty.setObjectName("reportMuted")
            outer.addWidget(empty)
            return frame

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        grid_content = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_content)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setSpacing(12)

        columns = 3
        for i, (conf, det, rec) in enumerate(best):
            card = QtWidgets.QFrame()
            card.setStyleSheet("background: #0d151c; border: 1px solid #1f303c; border-radius: 9px;")
            v = QtWidgets.QVBoxLayout(card)
            v.setContentsMargins(7, 7, 7, 7)
            v.setSpacing(4)

            image_path = self.capture.mission_dir / rec["image"]
            pix = QtGui.QPixmap(str(image_path)).scaled(
                270, 155, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            pix = self._stamp_confidence(pix, conf)

            img = QtWidgets.QLabel()
            img.setPixmap(pix)
            img.setAlignment(QtCore.Qt.AlignCenter)
            v.addWidget(img)

            info = QtWidgets.QLabel(f'{det.get("class_name") or tr("unknown")}   •   {conf:.1%}')
            info.setAlignment(QtCore.Qt.AlignCenter)
            info.setStyleSheet("font-size: 12px; font-weight: 700; color: #d8e2ea;")
            v.addWidget(info)

            grid.addWidget(card, i // columns, i % columns)

        scroll.setWidget(grid_content)
        outer.addWidget(scroll)
        return frame

    def _open_folder(self):

        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(
                str(self.capture.mission_dir)
            )
        )
