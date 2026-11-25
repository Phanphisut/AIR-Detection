"""
Video-clip AI test popup for AI Detection Lab.

Companion to the webcam test on the main dashboard (see WebcamAIThread in
main_window.py): instead of a live camera, this plays back an imported
video file (.mp4/.avi/.mov/.mkv) frame by frame through every YOLO model
currently in config.MODEL_DIR, drawing the exact same label+confidence
overlay the webcam test and the AI camera draw. It exists specifically for
right after training finishes on the "Train & Export" tab of
LabelTrainDialog: a student who just trained a model on frames pulled out
of one clip can immediately hand it a DIFFERENT clip and watch, frame by
frame, whether it actually recognizes the object - no webcam or drone
needed, and no separate "export/import a report" step.

Reuses ai/yolo_detector.py's YOLODetector unchanged (hot-discovers .pt
files, applies each model's own calibrated confidence threshold) so a clip
test can never show something the webcam test or AI camera wouldn't - one
detection code path for all three.

Looks and behaves like a real video player: a scrubbable timeline lets the
student drag to any point in the clip (even while paused, single-stepping
that one frame through the detector so the preview stays live), Play/Pause
and Stop sit next to it as transport controls, and a "live labels" chip
row underneath echoes whatever the detector found in the current frame in
big, readable text - the same information the tiny on-frame overlay boxes
carry, just legible at a glance instead of squinted at on a scaled-down
video.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from ..i18n import tr
from .code_editor import DIALOG_STYLESHEET, EN_US_LOCALE

VIDEO_IMPORT_FILTER_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")

# Consistent per-class chip colors for the live-labels row - keyed by a
# cheap hash of the class name so "cone" is always the same color across
# frames/clips instead of flickering between colors as dict ordering shifts.
_LABEL_PALETTE = ("#34d399", "#fbbf24", "#60a5fa", "#f472b6", "#a78bfa", "#fb923c", "#4ade80", "#22d3ee")


def _class_color(name: str) -> str:
    return _LABEL_PALETTE[sum(ord(c) for c in name) % len(_LABEL_PALETTE)]


def _format_time(frame_index: int, fps: float) -> str:
    total_seconds = int(max(0, frame_index) / fps) if fps > 0 else 0
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


class VideoClipTestThread(threading.Thread):
    """Plays a video file through YOLODetector, exactly like
    WebcamAIThread in main_window.py plays a live camera through it - same
    lazy import (ultralytics/torch cost real seconds, so this only happens
    once a clip is actually chosen), same "no model yet -> raw playback
    with a one-time warning, never a hard failure" fallback.

    Supports a real player's transport controls on top of that: pause()/
    resume() park the loop in a 50ms poll instead of tearing the thread
    down, and seek() queues a frame index that the loop applies via
    cv2.VideoCapture.set() on its next iteration - checked BEFORE the
    pause check so dragging the scrub bar while paused still decodes and
    shows that one frame instead of leaving the last frame frozen onscreen.

    Tallies per-class detection counts/confidence as it plays so the
    dialog can show a plain-language summary ("found 'cone' in 41/60
    frames, avg confidence 0.87") once the clip ends, instead of leaving
    the student to eyeball a scrolling video for the whole clip.
    """

    def __init__(self, video_path: Path, on_frame, on_status, on_error, on_finished, on_opened=None):
        super().__init__(daemon=True, name="AI-LAB-VIDEO-TEST")
        self.video_path = video_path
        self.on_frame = on_frame
        self.on_status = on_status
        self.on_error = on_error
        self.on_finished = on_finished
        self.on_opened = on_opened
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._seek_lock = threading.Lock()
        self._seek_to = None

    def stop(self):
        self._stop_event.set()

    def pause(self):
        self._pause_event.set()

    def resume(self):
        self._pause_event.clear()

    def seek(self, frame_index: int):
        with self._seek_lock:
            self._seek_to = max(0, frame_index)

    def run(self):
        summary = {
            "frames": 0,
            "frames_with_detection": 0,
            # Total boxes seen for this class across the whole clip - a
            # single frame with 6 overlapping "Car" boxes counts as 6 here.
            "class_counts": {},
            "class_conf_sum": {},
            # Frames that contained AT LEAST ONE box of this class - counted
            # once per frame no matter how many boxes of it that frame had,
            # so "class_frame_counts[name] / frames" is always a real 0-100%
            # ("found in N/total frames"), unlike class_counts above which
            # can run well past the frame count on a crowded frame.
            "class_frame_counts": {},
        }

        detector = None
        try:
            from ..ai.yolo_detector import YOLODetector

            detector = YOLODetector()
        except Exception as exc:
            self.on_status(
                f"⚠️ No YOLO model loaded ({exc}) — playing the clip without AI "
                "detection. Train or import a model first (step 3 / step 1)."
            )

        cap = None
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                self.on_error(f"❌ Could not open video: {self.video_path.name}")
                self.on_finished(None)
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if fps <= 0:
                fps = 30.0
            frame_delay = 1.0 / fps
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

            if self.on_opened:
                self.on_opened(total_frames, fps)

            mode = "AI detection ON" if detector is not None else "raw playback, no AI model"
            self.on_status(f"🎬 Testing clip: {self.video_path.name} ({mode})")

            # Our own counter, not cv2's - CAP_PROP_POS_FRAMES is confirmed
            # non-monotonic on some container/codec combos (observed jumping
            # backward mid-playback with no seek involved on a plain mp4v
            # clip), which would make the scrub bar visibly jitter backward.
            # A seek re-bases this counter to the target frame; otherwise it
            # just counts reads, which cv2 always advances by exactly one.
            frame_index = -1
            while not self._stop_event.is_set():
                with self._seek_lock:
                    target = self._seek_to
                    self._seek_to = None
                if target is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(target))
                    frame_index = target - 1
                elif self._pause_event.is_set():
                    time.sleep(0.05)
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_index += 1
                pos = frame_index

                if detector is not None:
                    try:
                        results = detector.predict(frame)
                        detections = detector.detections(results)
                        annotated = detector.annotate(results, frame)
                    except Exception as exc:
                        self.on_error(f"❌ Clip test: inference error: {exc}")
                        annotated = frame
                        detections = []
                else:
                    annotated = frame
                    detections = []

                summary["frames"] += 1
                if detections:
                    summary["frames_with_detection"] += 1
                classes_this_frame = set()
                for det in detections:
                    name = det["class_name"]
                    summary["class_counts"][name] = summary["class_counts"].get(name, 0) + 1
                    summary["class_conf_sum"][name] = (
                        summary["class_conf_sum"].get(name, 0.0) + det["confidence"]
                    )
                    classes_this_frame.add(name)
                for name in classes_this_frame:
                    summary["class_frame_counts"][name] = summary["class_frame_counts"].get(name, 0) + 1

                self.on_frame(annotated, detections, pos, total_frames)
                # Best-effort real-time pacing so the clip is watchable, not
                # a blur - if inference itself is slower than frame_delay
                # this just plays back slower than real time, never crashes.
                time.sleep(frame_delay)

            if self._stop_event.is_set():
                self.on_status(f"⏹ Clip test stopped early - {summary['frames']} frame(s) played")
            else:
                self.on_status(f"✅ Clip test finished - {summary['frames']} frame(s) played")
            self.on_finished(summary)
        except Exception as exc:
            self.on_error(f"❌ Clip test error: {exc}")
            self.on_finished(summary if summary["frames"] else None)
        finally:
            if cap is not None:
                cap.release()


class VideoClipTestDialog(QtWidgets.QDialog):
    """Non-modal popup: import a video clip, watch it play back with the
    project's trained model(s) labeling every frame live, then read a
    plain-language summary of what was found.

    Kept alive and reused (shown via .show()/.raise_() rather than
    recreated) by whoever opens it - same "never destroyed, just
    hidden/reshown" pattern as LabelTrainDialog uses
    for their own popups, so switching clips repeatedly doesn't leak
    threads or windows.
    """

    frame_signal = QtCore.Signal(object, object, int, int)
    status_signal = QtCore.Signal(str)
    finished_signal = QtCore.Signal(object)
    opened_signal = QtCore.Signal(int, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("AI Detection Lab — Test a Video Clip"))
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.setWindowFlag(QtCore.Qt.WindowMinMaxButtonsHint, True)
        self.setStyleSheet(DIALOG_STYLESHEET + self._EXTRA_STYLESHEET)
        self.resize(920, 760)
        self.setMinimumSize(600, 480)

        self._thread: VideoClipTestThread | None = None
        self._fps = 30.0
        self._total_frames = 0
        self._paused = False
        self._slider_dragging = False

        self._build_ui()

        self.import_btn.clicked.connect(self._import_and_test)
        self.play_pause_btn.clicked.connect(self._toggle_play_pause)
        self.stop_btn.clicked.connect(self._stop)
        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
        self.seek_slider.sliderMoved.connect(self._on_slider_moved)
        self.frame_signal.connect(self._on_frame)
        self.status_signal.connect(self._on_status)
        self.finished_signal.connect(self._on_finished)
        self.opened_signal.connect(self._on_opened)

    _EXTRA_STYLESHEET = """
        QLabel#playerHeader { font-weight: 700; color: #dfe7ef; font-size: 10pt; }
        QLabel#timeLabel { color: #8ea0b2; font-family: "Cascadia Mono", "Consolas"; font-size: 9pt; }
        QLabel#labelsRow {
            background: #0e151c;
            border: 1px solid #263544;
            border-radius: 8px;
            padding: 8px 10px;
        }
        QPushButton#transportButton {
            background: #17222d;
            border: 1px solid #2b3b4a;
            border-radius: 20px;
            min-width: 40px;
            min-height: 40px;
            max-width: 40px;
            max-height: 40px;
            font-size: 13pt;
            padding: 0;
        }
        QPushButton#transportButton:hover { background: #1d2c39; border-color: #466078; }
        QPushButton#transportButton:disabled { background: #10171f; color: #596978; border-color: #1a2530; }
        QSlider::groove:horizontal {
            height: 6px;
            background: #182430;
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background: #34d399;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            width: 14px;
            height: 14px;
            margin: -5px 0;
            background: #e8eef5;
            border-radius: 7px;
        }
        QSlider::handle:horizontal:disabled { background: #465666; }
    """

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        from .panel_theme import prepare
        prepare(layout)

        title = QtWidgets.QLabel(tr("Model playground"))
        title.setObjectName("dlgTitle")
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel(tr(
            "Import any video clip and watch every trained model in this "
            "project's models folder label it live - the exact same overlay "
            "the webcam test and AI camera draw, just played back from a "
            "file instead of a live camera."
        ))
        subtitle.setObjectName("dlgSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # --- Player header: shows the loaded clip's filename, like a real
        # video app's "now playing" strip. Hidden until a clip is loaded.
        self.player_header = QtWidgets.QLabel("")
        self.player_header.setObjectName("playerHeader")
        self.player_header.setVisible(False)
        layout.addWidget(self.player_header)

        # --- Video / drop-zone area
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setWordWrap(True)
        self.video_label.setMinimumHeight(200)
        self._set_placeholder_text()
        layout.addWidget(self.video_label, 1)

        # --- Scrub bar: current time / draggable timeline / total time
        scrub_row = QtWidgets.QHBoxLayout()
        scrub_row.setSpacing(8)
        self.current_time_label = QtWidgets.QLabel("00:00")
        self.current_time_label.setObjectName("timeLabel")
        scrub_row.addWidget(self.current_time_label)
        self.seek_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setEnabled(False)
        scrub_row.addWidget(self.seek_slider, 1)
        self.total_time_label = QtWidgets.QLabel("00:00")
        self.total_time_label.setObjectName("timeLabel")
        scrub_row.addWidget(self.total_time_label)
        layout.addLayout(scrub_row)

        # --- Transport controls: Import / Play-Pause / Stop
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(10)
        self.import_btn = QtWidgets.QPushButton(tr("📂  Import a Video Clip to Test"))
        self.import_btn.setObjectName("primaryButton")
        btn_row.addWidget(self.import_btn)
        self.play_pause_btn = QtWidgets.QPushButton("⏸")
        self.play_pause_btn.setObjectName("transportButton")
        self.play_pause_btn.setToolTip(tr("Pause"))
        self.play_pause_btn.setEnabled(False)
        btn_row.addWidget(self.play_pause_btn)
        self.stop_btn = QtWidgets.QPushButton("⏹")
        self.stop_btn.setObjectName("transportButton")
        self.stop_btn.setToolTip(tr("Stop"))
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # --- Live labels: legible chip row echoing what THIS frame's
        # detector found, separate from the small on-frame overlay boxes.
        labels_title = QtWidgets.QLabel(tr("🏷️ LIVE LABELS"))
        labels_title.setObjectName("sectionTitle")
        layout.addWidget(labels_title)
        self.labels_row = QtWidgets.QLabel(tr("Nothing detected yet"))
        self.labels_row.setObjectName("labelsRow")
        self.labels_row.setWordWrap(True)
        self.labels_row.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(self.labels_row)

        self.progress = QtWidgets.QProgressBar()
        # Forces "13%" instead of Thai/other-locale digits ("๑๓%") on a
        # system whose regional format isn't English - confirmed against a
        # real Thai-locale Windows machine, same fix label_train_dialog.py
        # already applies to its epoch/interval spin boxes via this same
        # EN_US_LOCALE constant.
        self.progress.setLocale(EN_US_LOCALE)
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status_label = QtWidgets.QLabel(tr("⚪ No clip loaded yet"))
        self.status_label.setObjectName("muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

    def _set_placeholder_text(self):
        self.video_label.setText(
            "<div style='color:#5a6c7d;'>"
            "<div style='font-size:34pt;'>🎞️</div>"
            "<div style='font-size:11pt; font-weight:700; color:#8ea0b2; margin-top:6px;'>"
            + tr("Import a clip to begin") + "</div>"
            "<div style='font-size:9pt; margin-top:4px;'>"
            + tr("MP4, AVI, MOV or MKV") + "</div>"
            "</div>"
        )
        self.video_label.setStyleSheet(
            "background:#070c11; border:2px dashed #263544; border-radius:10px; color:#738596;"
        )

    def _import_and_test(self):
        if self._thread is not None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("Import a video clip to test"), "",
            f"{tr('Video Files')} (*.mp4 *.avi *.mov *.mkv);;{tr('All Files')} (*)",
        )
        if not path:
            return
        self._start(Path(path))

    def _start(self, video_path: Path):
        self.import_btn.setEnabled(False)
        self.play_pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.summary_label.setText("")
        self.labels_row.setText(tr("Nothing detected yet"))
        self.status_label.setText(f"⏳ {tr('Opening')} {video_path.name}...")

        self.player_header.setText(f"🎬 {video_path.name}")
        self.player_header.setVisible(True)
        self.video_label.setStyleSheet(
            "background:#070c11; border:1px solid #263544; border-radius:10px; color:#738596;"
        )

        self._paused = False
        self._slider_dragging = False
        self.play_pause_btn.setText("⏸")
        self.play_pause_btn.setToolTip(tr("Pause"))

        self._thread = VideoClipTestThread(
            video_path=video_path,
            on_frame=lambda frame, dets, pos, total: self.frame_signal.emit(frame, dets, pos, total),
            on_status=lambda msg: self.status_signal.emit(msg),
            on_error=lambda msg: self.status_signal.emit(msg),
            on_finished=lambda summary: self.finished_signal.emit(summary),
            on_opened=lambda total, fps: self.opened_signal.emit(total, fps),
        )
        self._thread.start()

    def _toggle_play_pause(self):
        if self._thread is None:
            return
        self._paused = not self._paused
        if self._paused:
            self._thread.pause()
            self.play_pause_btn.setText("▶")
            self.play_pause_btn.setToolTip(tr("Resume"))
        else:
            self._thread.resume()
            self.play_pause_btn.setText("⏸")
            self.play_pause_btn.setToolTip(tr("Pause"))

    def _stop(self):
        if self._thread is not None:
            self._thread.stop()
        self.play_pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    def _on_slider_pressed(self):
        self._slider_dragging = True

    def _on_slider_moved(self, value: int):
        # Live time readout while dragging, without spamming a seek() per
        # pixel of drag - the actual seek is queued once, on release.
        self.current_time_label.setText(_format_time(value, self._fps))

    def _on_slider_released(self):
        self._slider_dragging = False
        if self._thread is not None:
            self._thread.seek(self.seek_slider.value())

    @QtCore.Slot(int, float)
    def _on_opened(self, total_frames: int, fps: float):
        self._fps = fps if fps > 0 else 30.0
        self._total_frames = total_frames
        self.seek_slider.setEnabled(total_frames > 0)
        self.seek_slider.setRange(0, max(0, total_frames - 1))
        self.total_time_label.setText(_format_time(total_frames, self._fps))
        self.play_pause_btn.setEnabled(True)

    @QtCore.Slot(object, object, int, int)
    def _on_frame(self, frame, detections, pos, total):
        if frame is None:
            return
        count = len(detections)
        det_word = tr("detection") if count == 1 else tr("detections")
        if total > 0:
            self.progress.setValue(min(100, int(pos * 100 / total)))
            self.status_label.setText(
                f"🟢 {tr('Playing')} — {tr('frame')} {pos}/{total} ({count} {det_word})"
            )
        else:
            self.status_label.setText(f"🟢 {tr('Playing')} — {tr('frame')} {pos} ({count} {det_word})")

        if not self._slider_dragging:
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(pos)
            self.seek_slider.blockSignals(False)
            self.current_time_label.setText(_format_time(pos, self._fps))

        if detections:
            chips = []
            for det in detections:
                name = det["class_name"]
                conf = det["confidence"]
                color = _class_color(name)
                chips.append(
                    f"<span style='background:{color}22; color:{color}; border:1px solid {color}; "
                    f"border-radius:10px; padding:2px 9px; margin-right:6px;'>{name} {conf:.2f}</span>"
                )
            self.labels_row.setText(" ".join(chips))
        else:
            self.labels_row.setText(f"<span style='color:#5a6c7d;'>{tr('No objects detected in this frame')}</span>")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        )

    @QtCore.Slot(str)
    def _on_status(self, msg: str):
        self.status_label.setText(msg)

    @QtCore.Slot(object)
    def _on_finished(self, summary):
        self._thread = None
        self.import_btn.setEnabled(True)
        self.play_pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.progress.setVisible(False)
        self._paused = False
        self.play_pause_btn.setText("⏸")

        if not summary or not summary.get("frames"):
            self.summary_label.setText(tr("No frames were played."))
            return

        total_frames = summary["frames"]
        with_det = summary["frames_with_detection"]
        counts = summary["class_counts"]
        conf_sums = summary["class_conf_sum"]

        if not counts:
            self.summary_label.setText(
                f"📋 {tr('Result')}: " + tr("no objects were detected in any frame of this clip.")
            )
            return

        frame_counts = summary["class_frame_counts"]
        lines = [
            f"📋 {tr('Result')} — {with_det}/{total_frames} " + tr("frames had a detection") + ":"
        ]
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            avg_conf = conf_sums[name] / count
            # % of frames the class appeared in AT LEAST ONCE, not the raw
            # detection count - a frame with several overlapping boxes of
            # the same class (a crowd of "Car" boxes on one bus photo, say)
            # would otherwise push this well past 100%.
            found_in = frame_counts.get(name, 0)
            pct = 100.0 * found_in / total_frames
            lines.append(
                f"   • {name}: " + tr("found in") + f" {found_in}/{total_frames} "
                f"{tr('frames')} ({pct:.0f}%), {count} {tr('detections')} {tr('total')}, "
                f"{tr('avg confidence')} {avg_conf:.2f}"
            )
        self.summary_label.setText("\n".join(lines))

    def closeEvent(self, event):
        # The dialog itself is kept alive by whoever opens it (re-shown via
        # .show(), never destroyed) - a late signal emission from the
        # thread after this still lands on a live, just-hidden widget.
        if self._thread is not None:
            self._thread.stop()
        event.accept()
