from __future__ import annotations
import threading
import time

from pyhulax.video import StreamState

from .controller import HulaDroneController
from .. import config


class CameraWorker(threading.Thread):
    """Owns the Hula RTP frame loop for AI Lab only."""

    def __init__(self, drone: HulaDroneController, on_frame, on_status, on_ai_frame=None):
        super().__init__(daemon=True, name="AI-LAB-CAMERA")
        self.drone = drone
        self.on_frame = on_frame
        self.on_status = on_status
        self.on_ai_frame = on_ai_frame
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        try:
            # The working game initializes the SDK connection before starting
            # its camera thread. Do the same here, without importing game state.
            if not self.drone.connect():
                self.on_status("❌ Drone connection failed")
                return
            self.on_status("✅ Drone connected")

            # Camera angle is intentionally set only after the SDK is connected.
            # set_camera_down() is safe to call every time a camera session
            # starts — HulaDroneController now skips re-issuing the angle
            # command if it's already down for the current connected
            # session, and resets that guard automatically on reconnect.
            if not self.drone.set_camera_down():
                self.on_status("❌ Camera down 90° failed")
                return
            self.on_status("📷 Camera set to DOWN 90°")

            # Match the working game RTP startup order.
            if not self.drone.start_stream():
                self.on_status("❌ RTP stream failed")
                return
            self.on_status("🎥 RTP stream active — waiting for first frame")

            waiting_since = time.time()
            # pyhulax (and the Simulation Mode stream) create a NEW frame
            # object per decoded frame, so an identity check tells us
            # whether anything new arrived. Without it this loop re-sent
            # the SAME frame ~100 times/sec (it only sleeps FRAME_SLEEP_SEC),
            # flooding the GUI thread with redundant full-size redraws -
            # a big source of stutter.
            last_sent = None

            while not self.stop_event.is_set():
                stream = self.drone.video_stream
                latest = stream.latest_frame if stream is not None else None
                if stream is None and not getattr(self.drone, "stream_started", False):
                    # Video was switched off from outside this thread
                    # (Emergency Stop / stop_stream()). Nothing will ever
                    # arrive again - end now instead of hanging for
                    # FIRST_FRAME_TIMEOUT_SEC, so the app can offer a
                    # clean reconnect straight away.
                    self.on_status("🔒 Video stream stopped (emergency stop / landing)")
                    return
                if latest is None:
                    # Surface a PyAV-side failure (e.g. RTP port never
                    # received a packet — a common symptom when the
                    # previous drone's camera session didn't fully tear
                    # down its socket before this one started) instead of
                    # looping forever with no feedback.
                    # getattr: the Simulation Mode stream (webcam) has no
                    # .state/.last_error - it used to crash here with
                    # AttributeError whenever its first frame wasn't ready yet.
                    if stream is not None and getattr(stream, "state", None) == StreamState.ERROR:
                        self.on_status(
                            f"❌ RTP stream error: {getattr(stream, 'last_error', None) or 'unknown error'}"
                        )
                        return
                    if time.time() - waiting_since > config.FIRST_FRAME_TIMEOUT_SEC:
                        self.on_status(
                            "❌ No video frame received within "
                            f"{config.FIRST_FRAME_TIMEOUT_SEC:.0f}s — RTP stream never "
                            "started. Stop and reconnect this drone."
                        )
                        return
                    time.sleep(config.NO_FRAME_SLEEP_SEC)
                    continue

                # pyhulax's Frame.image is already BGR (OpenCV format) - no
                # colorspace conversion needed here, unlike the old pyhula
                # get_image_array() which returned RGB.
                if latest is last_sent:
                    time.sleep(config.FRAME_SLEEP_SEC)
                    continue
                last_sent = latest

                frame_bgr = latest.image

                self.on_status_once("VIDEO STREAM ACTIVE — First frame received")

                # Live video is delivered immediately. AI receives the same
                # frame through a separate latest-frame worker so YOLO can
                # never block the camera loop.
                self.on_frame(frame_bgr)
                if self.on_ai_frame is not None:
                    self.on_ai_frame(frame_bgr)

                time.sleep(config.FRAME_SLEEP_SEC)

        except Exception as exc:
            self.on_status(f"⚠️ Camera error: {exc}")
        finally:
            self.drone.stop_stream()
            self.on_status("🔒 Camera stopped")

    _first_frame_reported = False

    def on_status_once(self, message: str):
        if not self._first_frame_reported:
            self._first_frame_reported = True
            self.on_status(message)