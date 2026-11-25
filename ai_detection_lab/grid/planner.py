from __future__ import annotations
from dataclasses import dataclass
from .. import config


@dataclass(frozen=True)
class GridWaypoint:
    index: int
    x_cm: int
    y_cm: int
    heading: str
    turn: str = "NONE"


def _axis_points(length_cm: int, spacing_cm: int) -> list[int]:
    pts = list(range(0, length_cm + 1, spacing_cm))
    if not pts or pts[-1] != length_cm:
        pts.append(length_cm)
    return pts


def generate_zigzag_waypoints() -> list[tuple[int, int]]:
    """Survey points only. The flight executor turns the drone and flies forward."""
    cols = _axis_points(config.FIELD_WIDTH_CM, config.GRID_SPACING_CM)
    rows = _axis_points(config.FIELD_HEIGHT_CM, config.GRID_SPACING_CM)
    points: list[tuple[int, int]] = []
    for row, y in enumerate(rows):
        xs = cols if row % 2 == 0 else list(reversed(cols))
        points.extend((x, y) for x in xs)
    return points


def generate_waypoint_objects() -> list[GridWaypoint]:
    points = generate_zigzag_waypoints()
    out: list[GridWaypoint] = []
    heading = config.INITIAL_HEADING

    for i, (x, y) in enumerate(points):
        if i == 0:
            out.append(GridWaypoint(i + 1, x, y, heading, "NONE"))
            continue

        px, py = points[i - 1]
        dx, dy = x - px, y - py

        if dx > 0:
            desired = "EAST"
        elif dx < 0:
            desired = "WEST"
        elif dy > 0:
            desired = "SOUTH"
        else:
            desired = "NORTH"

        turn = "NONE" if desired == heading else _turn_instruction(heading, desired)
        out.append(GridWaypoint(i + 1, x, y, desired, turn))
        heading = desired

    return out


def _turn_instruction(current: str, desired: str) -> str:
    order = ["EAST", "SOUTH", "WEST", "NORTH"]
    ci = order.index(current)
    di = order.index(desired)
    diff = (di - ci) % 4
    if diff == 1:
        return "RIGHT_90"
    if diff == 3:
        return "LEFT_90"
    if diff == 2:
        return "TURN_180"
    return "NONE"
