"""🗺️ Survey Route in 3D: the autonomous grid's planned route over the field,
and - while AutonomousGridMission flies - the real drone moving along it.

Same QPainter 3D view as the Flight Plan simulation (ui/flight_sim_view.py).
It only DRAWS: nothing here sends a flight command.

Coordinates: the grid planner uses x = east (cm), y = south (cm) with the
origin at the field's north-west corner. The 3D view uses y = north, so a
waypoint (x, y) is drawn at (x, FIELD_HEIGHT_CM - y).
"""
from __future__ import annotations

from dataclasses import replace

from PySide6 import QtCore, QtWidgets

from .. import config
from ..i18n import tr
from ..missions.challenges import Area, Checkpoint, Pad
from ..missions.flight_sim import Pose, SimTrack
from .flight_sim_view import FlightSim3DView

# heading name -> yaw of the 3D drone (counter-clockwise, 0 = north / +y)
_YAW = {"NORTH": 0.0, "WEST": 90.0, "SOUTH": 180.0, "EAST": -90.0}
_ANIM_SEC = 0.6


def _to_3d(x_cm, y_cm, z=0.0):
    return float(x_cm), float(config.FIELD_HEIGHT_CM - y_cm), float(z)


class SurveyRoute3DWidget(QtWidgets.QWidget):
    """Drop-in for the 2D grid preview: same set_current_waypoint() /
    clear_current_waypoint() calls from MainWindow, plus set_finished()."""

    def __init__(self, waypoints, parent=None):
        super().__init__(parent)
        self.waypoints = list(waypoints)
        self.altitude = float(config.SURVEY_ALTITUDE_CM)
        self.track = SimTrack()
        self.track.mode = "route"
        self.view = FlightSim3DView(self.track)
        self.view.show_takeoff_arrow = False
        self.view.show_start_pad = False   # START is drawn as its own pad at waypoint 1
        self.view.show_position = False    # field X/Y is in the status label instead
        self.view.follow = False
        # a 20 cm drone drawn x2.4 hides a 1 m field: scale it with the field
        span = max(config.FIELD_WIDTH_CM, config.FIELD_HEIGHT_CM)
        self.view.drone_scale = max(0.9, min(2.4, span / 220.0))
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view, 1)
        self.current_wp_index = None
        self._anim_from = None
        self._anim_to = None
        self._anim_t = 1.0
        self._anim = QtCore.QTimer(self)
        self._anim.timeout.connect(self._step)
        self._build_world()
        self.clear_current_waypoint()

    # -- static scene --------------------------------------------------------
    def _build_world(self):
        w, h = config.FIELD_WIDTH_CM, config.FIELD_HEIGHT_CM
        world = [Area(0, 0, w, h, f"{w} × {h} cm", north=True)]
        if self.waypoints:
            sx, sy, _ = _to_3d(self.waypoints[0].x_cm, self.waypoints[0].y_cm)
            world.append(Pad(sx, sy, 22, "START"))
        many = len(self.waypoints) > 40
        radius = max(5.0, min(14.0, config.GRID_SPACING_CM * 0.14))
        for wp in self.waypoints:
            x, y, z = _to_3d(wp.x_cm, wp.y_cm, self.altitude)
            world.append(Checkpoint(x, y, z, radius, "" if many else str(wp.index)))
        self.track.set_world(world)
        plan = []
        if self.waypoints:
            first, last = self.waypoints[0], self.waypoints[-1]
            plan.append(_to_3d(first.x_cm, first.y_cm, 0))
            plan += [_to_3d(wp.x_cm, wp.y_cm, self.altitude) for wp in self.waypoints]
            plan.append(_to_3d(last.x_cm, last.y_cm, 0))
        self.track.set_plan(plan)
        self.view.frame_world(world)
        # look at the field from its south-west corner, a little higher
        self.view.azimuth = -125.0
        self.view.elevation = 38.0
        self.view.distance = max(260.0, 1.45 * max(w, h))

    def _start_pose(self):
        if not self.waypoints:
            return Pose()
        x, y, _ = _to_3d(self.waypoints[0].x_cm, self.waypoints[0].y_cm)
        return Pose(x=x, y=y, z=0.0, yaw=_YAW.get(config.INITIAL_HEADING, 0.0), camera=90)

    # -- live flight (called by MainWindow on the GUI thread) -----------------
    def set_current_waypoint(self, index, x, y, heading):
        first = self.current_wp_index is None
        self.current_wp_index = index
        px, py, pz = _to_3d(x, y, self.altitude)
        target = replace(self.track.snapshot().pose, x=px, y=py, z=pz,
                         yaw=_YAW.get(str(heading).upper(), self.track.snapshot().pose.yaw),
                         flying=True, camera=90)
        if first:
            self.track.reset(keep_plan=True, start=replace(self._start_pose(), flying=True))
            self.track.mode = "route_live"
        self.track.set_status(label=f"WP {index}/{len(self.waypoints)}  •  X={x} Y={y} cm  •  {heading}")
        self._animate_to(target)

    def clear_current_waypoint(self):
        """Back to the pre-flight preview: drone parked on START."""
        self._anim.stop()
        self.current_wp_index = None
        self.track.reset(keep_plan=True, start=self._start_pose())
        self.track.mode = "route"
        self.track.set_status(label=tr("Preview only — not flying"))

    def set_finished(self, success: bool):
        """Mission ended: the drone lands where it got to."""
        if self.current_wp_index is None:
            return
        pose = self._anim_to or self.track.snapshot().pose
        self._animate_to(replace(pose, z=0.0, flying=False))
        self.track.set_status(label=(tr("Mission complete") if success else tr("Mission stopped"))
                              + f"  •  WP {self.current_wp_index}/{len(self.waypoints)}")

    # -- smooth move between the waypoint reports ------------------------------
    def _animate_to(self, target):
        if self._anim.isActive() and self._anim_to is not None:
            # A new report before the last move finished drawing: finish it
            # first, so the flown line still passes through every waypoint.
            self.track.set_pose(self._anim_to)
        self._anim_from = self.track.snapshot().pose
        self._anim_to = target
        self._anim_t = 0.0
        if not self._anim.isActive():
            self._anim.start(33)

    def _step(self):
        self._anim_t = min(1.0, self._anim_t + 0.033 / _ANIM_SEC)
        a, b, t = self._anim_from, self._anim_to, self._anim_t
        t = t * t * (3 - 2 * t)
        dyaw = (b.yaw - a.yaw + 180.0) % 360.0 - 180.0   # turn the short way round
        self.track.set_pose(replace(b, x=a.x + (b.x - a.x) * t, y=a.y + (b.y - a.y) * t,
                                    z=a.z + (b.z - a.z) * t, yaw=a.yaw + dyaw * t,
                                    flying=a.flying or b.flying))
        if self._anim_t >= 1.0:
            self._anim.stop()
            self.track.set_pose(b)


class SurveyRouteViews(QtWidgets.QTabWidget):
    """🧊 3D (default) and the original 🗺️ 2D map, kept in step."""

    def __init__(self, waypoints, map_2d, parent=None):
        super().__init__(parent)
        self.view_3d = SurveyRoute3DWidget(waypoints)
        self.map_2d = map_2d
        self.addTab(self.view_3d, tr("🧊 3D view"))
        self.addTab(map_2d, tr("🗺️ 2D map (top)"))

    @property
    def current_wp_index(self):
        return self.view_3d.current_wp_index

    def set_current_waypoint(self, index, x, y, heading):
        self.view_3d.set_current_waypoint(index, x, y, heading)
        self.map_2d.set_current_waypoint(index, x, y, heading)

    def clear_current_waypoint(self):
        self.view_3d.clear_current_waypoint()
        self.map_2d.clear_current_waypoint()

    def set_finished(self, success: bool):
        self.view_3d.set_finished(success)
