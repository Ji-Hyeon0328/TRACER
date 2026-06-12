from __future__ import annotations

import json
import socket
import time
from typing import Optional


class LowLevelUdpClient:
    """Tiny UDP client used inside Isaac Lab conda process.

    Direction:
      Isaac Lab -> ROS2 sidecar:
        send robot_state to UDP port 50100

      ROS2 sidecar -> Isaac Lab:
        receive low_level_cmd from UDP port 50101
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        state_send_port: int = 50100,
        cmd_recv_port: int = 50101,
    ):
        self.host = host
        self.state_addr = (host, int(state_send_port))

        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.cmd_sock.bind((host, int(cmd_recv_port)))
        self.cmd_sock.setblocking(False)

        self.latest_cmd: Optional[list[float]] = None
        self.latest_cmd_stamp: float = 0.0

    def send_robot_state(self, data: list[float]):
        payload = {
            "stamp": time.time(),
            "data": data,
        }
        raw = json.dumps(payload).encode("utf-8")
        self.state_sock.sendto(raw, self.state_addr)

    def poll_low_level_cmd(self):
        while True:
            try:
                raw, _ = self.cmd_sock.recvfrom(65535)
            except BlockingIOError:
                break

            try:
                payload = json.loads(raw.decode("utf-8"))
                data = list(payload.get("data", []))
            except Exception:
                continue

            if len(data) >= 61:
                self.latest_cmd = data[:61]
                self.latest_cmd_stamp = time.time()

    def get_latest_cmd(self, timeout_s: float = 0.2) -> Optional[list[float]]:
        self.poll_low_level_cmd()
        if self.latest_cmd is None:
            return None
        if time.time() - self.latest_cmd_stamp > float(timeout_s):
            return None
        return self.latest_cmd
