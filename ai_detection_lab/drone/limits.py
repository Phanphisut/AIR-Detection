"""The Hula firmware / pyhulax limits for single commands, in ONE place.

These are pyhulax's own parameter checks (fylo/taskprocessor.py). A value
outside them is NOT sent to the drone at all - pyhulax prints "parameter
error" and the call comes back as a timeout, which looked to the user like
"the command randomly fails". Both the real controller and the Simulation
Mode controller validate against these, so Simulation fails the same way a
real drone would.

  move forward/back/left/right : whole number of cm, 5..500
  up / down                    : whole number of cm, 5..500
  turn left / right            : whole number of degrees, 1..360
                                  (a float such as 90.0 is rejected, and for
                                  turn-right it even crashes MAVLink packing)
"""
from __future__ import annotations

MOVE_MIN_CM = 5
MOVE_MAX_CM = 500
ROTATE_MIN_DEG = 1
ROTATE_MAX_DEG = 360


class CommandValueError(ValueError):
    """A distance/angle the drone would reject. The message is for students."""


def move_cm(value) -> int:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        raise CommandValueError(f"Distance must be a number: {value!r}")
    if v < MOVE_MIN_CM or v > MOVE_MAX_CM:
        raise CommandValueError(
            f"Got {v} cm "
            f"(distance must be {MOVE_MIN_CM}-{MOVE_MAX_CM} cm)"
        )
    return v


def rotate_deg(value) -> int:
    """Absolute size of a turn, as the whole number the firmware accepts."""
    try:
        v = int(round(abs(float(value))))
    except (TypeError, ValueError):
        raise CommandValueError(f"Angle must be a number: {value!r}")
    if v < ROTATE_MIN_DEG or v > ROTATE_MAX_DEG:
        raise CommandValueError(
            f"Got {v}° "
            f"(turn must be {ROTATE_MIN_DEG}-{ROTATE_MAX_DEG}°)"
        )
    return v
