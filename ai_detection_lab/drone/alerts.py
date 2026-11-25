"""v9.0.0: alerts for basic drone problems - connection lost, battery low,
battery too low to fly - plus the 3-second alarm sound that goes with them.

AlertMonitor is plain logic (no Qt, no drone): main_window feeds it one
reading per second and shows whatever it returns, so the rules can be tested
without hardware (checks/verify_alerts.py).

Why the link check uses telemetry age and not a failed battery read: pyhulax's
get_battery() returns the LAST value it ever received, so after the drone
drops out it keeps answering the old percentage forever - the v6 "3 failed
battery reads" check could never fire on a real drone.
"""
from __future__ import annotations

import io
import math
import os
import struct
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass

from .. import config


@dataclass(frozen=True)
class Alert:
    kind: str          # link_lost / link_back / battery_low / battery_too_low / battery_too_low_flying
    title: str
    message: str
    sound: bool = True
    severity: str = "warn"   # warn / danger / info


def warn_pct() -> int:
    return int(getattr(config, "BATTERY_WARN_PCT", 30))


def takeoff_min_pct() -> int:
    return int(getattr(config, "BATTERY_TAKEOFF_MIN_PCT", 20))


def takeoff_block_reason(battery_pct) -> str:
    """'' if this battery level may take off, else the reason shown to the user."""
    if battery_pct is None:
        return ""
    if int(battery_pct) <= takeoff_min_pct():
        return (f"Battery too low to fly ({int(battery_pct)}%, needs more than {takeoff_min_pct()}%). "
                "Charge or change the battery first.")
    return ""


class AlertMonitor:
    """Turns readings into alerts, each one ONCE until the situation clears."""

    # Battery has to come back this far above the warning level (a fresh
    # battery) before the same warning may sound again - stops 29/30/29 % jitter
    # from beeping every second.
    RESET_MARGIN = 5

    def __init__(self):
        self.reset()

    def reset(self):
        self.link_lost = False
        self.battery_level = 0      # 0 ok, 1 warned (<= warn), 2 too low (<= takeoff min)

    def check(self, connected: bool, battery, telemetry_age, flying: bool) -> list[Alert]:
        if not connected:
            # The user disconnected (or never connected): nothing to alarm about.
            self.reset()
            return []
        alerts = []
        lost_after = float(getattr(config, "DRONE_LINK_LOST_SEC", 5.0))
        if telemetry_age is not None and telemetry_age > lost_after:
            if not self.link_lost:
                self.link_lost = True
                alerts.append(Alert(
                    "link_lost", "Lost the drone connection",
                    f"No signal from the drone for {telemetry_age:.0f} seconds\n"
                    + ("The drone was flying — the app sent an emergency landing. If it doesn't land, move closer or check the Wi-Fi"
                       if flying else
                       "Check the drone is on, the battery isn't empty, and the computer is on the drone's Wi-Fi\nThe app is reconnecting by itself — no need to restart it"),
                    severity="danger"))
            return alerts      # the battery value is stale while the link is down
        if self.link_lost:
            self.link_lost = False
            alerts.append(Alert("link_back", "The drone is connected again",
                                "Getting a signal from the drone again", sound=False, severity="info"))
        if battery is None:
            return alerts
        pct = int(battery)
        if pct <= takeoff_min_pct():
            if self.battery_level < 2:
                self.battery_level = 2
                if flying:
                    alerts.append(Alert(
                        "battery_too_low_flying", "Battery too low to fly",
                        f"Battery {pct}% — land now", severity="danger"))
                else:
                    alerts.append(Alert(
                        "battery_too_low", "Battery too low to fly",
                        f"Battery {pct}% (needs more than {takeoff_min_pct()}%) — the app won't take off\n"
                        "Charge or change the battery first", severity="danger"))
        elif pct <= warn_pct():
            if self.battery_level < 1:
                self.battery_level = 1
                alerts.append(Alert(
                    "battery_low", "Battery almost empty",
                    f"Battery {pct}% — " + ("get ready to land" if flying else "charge it before the next flight")))
        elif pct >= warn_pct() + self.RESET_MARGIN:
            self.battery_level = 0
        return alerts


# --- alarm sound -------------------------------------------------------------

ALARM_SEC = 3.0


def alarm_wav(seconds: float = ALARM_SEC, rate: int = 22050) -> bytes:
    """Fast beeping (2.7 kHz, 0.12 s on / 0.08 s off) for `seconds`, then silence."""
    on, period, freq = 0.12, 0.20, 2700.0
    frames = bytearray()
    for i in range(int(seconds * rate)):
        t = i / rate
        phase = t % period
        if phase < on:
            # 5 ms fade in/out so each beep does not click
            env = min(1.0, phase / 0.005, (on - phase) / 0.005)
            frames += struct.pack("<h", int(0.6 * 32767 * env * math.sin(2 * math.pi * freq * t)))
        else:
            frames += b"\x00\x00"
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


_wav_path = None


def _alarm_file() -> str:
    global _wav_path
    if _wav_path is None or not os.path.exists(_wav_path):
        fd, path = tempfile.mkstemp(prefix="aidl_alarm_", suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(alarm_wav())
        _wav_path = path
    return _wav_path


def play_alarm() -> bool:
    """Start the 3 s alarm without blocking; it stops by itself. False = no sound
    player on this computer (the on-screen alert still shows)."""
    if not getattr(config, "ALERT_SOUND_ENABLED", True):
        return False
    try:
        path = _alarm_file()
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            return True
        if sys.platform == "darwin":
            subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        for player in (["paplay", path], ["aplay", "-q", path]):
            try:
                subprocess.Popen(player, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except FileNotFoundError:
                continue
    except Exception as exc:
        print(f"⚠️ AI LAB: alarm sound failed: {exc}")
    return False
