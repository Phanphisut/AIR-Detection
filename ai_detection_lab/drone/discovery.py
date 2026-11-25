"""Find Hula drones on the network without connecting to any of them.

Drones send BROADCAST_PLANE_STATUS (MAVLink 232) to UDP 8668 every 8-15 s
without being asked (seen on a real F09-lite, 2026-10-07), plus REPORT_STATS
(207) while a client is connected. The packet's source address is the drone's
IP. bind_client 255 = nobody controls it; otherwise it is the last number of
the controlling computer's IP.

Ported from the iPhone app (AI_Detection_Lab_iOS/Sources/HulaKit/Discovery.swift),
which was tested against the real drone.
"""
from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass, field

STATUS_PORT = 8668
# crc_extra per message id (pyhulax fylo/mavlink.py)
_CRC_EXTRA = {207: 165, 232: 32}


def _crc(data: bytes, extra: int) -> int:
    crc = 0xFFFF
    for b in bytes(data) + bytes([extra]):
        tmp = b ^ (crc & 0xFF)
        tmp = (tmp ^ (tmp << 4)) & 0xFF
        crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    return crc


def parse_frames(data: bytes):
    """MAVLink v1 frames (0xFE) with a valid CRC -> [(msg_id, payload)]."""
    frames, i = [], 0
    while i + 8 <= len(data):
        if data[i] != 0xFE:
            i += 1
            continue
        length = data[i + 1]
        end = i + 6 + length + 2
        if end > len(data):
            break
        msg_id = data[i + 5]
        extra = _CRC_EXTRA.get(msg_id)
        if extra is not None:
            crc = struct.unpack_from("<H", data, end - 2)[0]
            if crc == _crc(data[i + 1:end - 2], extra):
                frames.append((msg_id, data[i + 6:end - 2]))
                i = end
                continue
        i += 1
    return frames


@dataclass
class FoundDrone:
    ip: str
    drone_id: int | None = None
    serial: str = ""
    bind_client: int | None = None
    battery: int | None = None
    last_seen: float = field(default_factory=time.monotonic)

    def availability(self, my_bind: int | None = None) -> str:
        """'free', 'mine', 'taken' or 'unknown'."""
        if self.bind_client is None:
            return "unknown"
        if self.bind_client in (0, 255):
            return "free"
        if my_bind is not None and self.bind_client == my_bind:
            return "mine"
        return "taken"

    def describe(self, my_bind: int | None = None) -> str:
        name = f"Drone #{self.drone_id}" if self.drone_id is not None else "drone"
        if self.serial:
            name += f" · {self.serial}"
        state = {"free": "free", "mine": "this computer", "taken": f"in use (.{self.bind_client})",
                 "unknown": "?"}[self.availability(my_bind)]
        battery = f" · battery {self.battery}%" if self.battery is not None else ""
        return f"{name} @ {self.ip}{battery} — {state}"


def ingest(found: dict, data: bytes, ip: str) -> None:
    """Update `found` (ip -> FoundDrone) from one datagram."""
    frames = parse_frames(data)
    if not frames:
        return
    drone = found.setdefault(ip, FoundDrone(ip=ip))
    drone.last_seen = time.monotonic()
    for msg_id, payload in frames:
        if msg_id == 232 and len(payload) >= 36:
            sn, _ip, plane_id, _wifi_mode, bind, _power = struct.unpack_from("<16s16sBBBB", payload)
            drone.serial = sn.split(b"\0", 1)[0].decode("utf-8", "replace").strip() or drone.serial
            if plane_id:
                drone.drone_id = plane_id
            drone.bind_client = bind
        elif msg_id == 207:
            stats = report_stats(payload)
            if stats:
                if stats.get("drone_id"):
                    drone.drone_id = stats["drone_id"]
                drone.battery = stats.get("battery_volumn")


def report_stats(payload: bytes) -> dict | None:
    """REPORT_STATS (207) payload -> {field: value}, decoded with pyhulax's own
    message definition so the layout can never drift from what the app uses."""
    try:
        from pyhulax.fylo import mavlink as m
        cls = m.MAVLink_report_stats_message
        values = list(cls.unpacker.unpack(payload[:cls.unpacker.size].ljust(cls.unpacker.size, b"\0")))
    except Exception:
        return None
    out, i = {}, 0
    for name, n in zip(cls.ordered_fieldnames, cls.array_lengths):
        if n:
            out[name], i = values[i:i + n], i + n
        else:
            out[name], i = values[i], i + 1
    return out


def discover(seconds: float = 16.0, port: int = STATUS_PORT, stop_when=None) -> list[FoundDrone]:
    """Listen on UDP `port` for `seconds` (drones announce every 8-15 s).
    stop_when(found_dict) -> True ends early."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.5)
    found: dict[str, FoundDrone] = {}
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            try:
                data, (ip, _port) = sock.recvfrom(4096)
            except socket.timeout:
                continue
            ingest(found, data, ip)
            if stop_when is not None and stop_when(found):
                break
    finally:
        sock.close()
    return sorted(found.values(), key=lambda d: (d.drone_id or 999, d.ip))


def my_bind_client(drone_ip: str) -> int | None:
    """Last number of this computer's IP on the drone's network (what the drone
    reports as bind_client when we control it)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((drone_ip, 9))
        ip = s.getsockname()[0]
        s.close()
        return int(ip.rsplit(".", 1)[1])
    except (OSError, ValueError, IndexError):
        return None
