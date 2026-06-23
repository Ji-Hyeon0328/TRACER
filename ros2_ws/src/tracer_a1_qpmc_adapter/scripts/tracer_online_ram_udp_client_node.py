#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import time
from collections import deque
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


ROOT = Path(os.environ.get("TRACER_ROOT", str(Path.home() / "Tracer/TRACER")))

SERVER_HOST = os.environ.get("TRACER_RAM_UDP_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("TRACER_RAM_UDP_PORT", "50210"))

RAM_HZ = float(os.environ.get("TRACER_RAM_CLIENT_HZ", "5.0"))
WINDOW = int(os.environ.get("TRACER_RAM_WINDOW", "30"))
INPUT_DIM = int(os.environ.get("TRACER_RAM_INPUT_DIM", "57"))

PROPRIO_DIM = int(os.environ.get("TRACER_RAM_PROPRIO_DIM", "45"))
MPC_DIM = int(os.environ.get("TRACER_RAM_MPC_DIM", "6"))
ODOM_DIM = int(os.environ.get("TRACER_RAM_ODOM_DIM", "6"))

REQUIRE_ODOM = os.environ.get("TRACER_RAM_REQUIRE_ODOM", "0") == "1"

# Freshness gates.
# Physics-paused Gazebo does not publish fresh proprio/odom.
# Without this gate, the RAM client can repeatedly infer from stale cached state.
PROPRIO_STALE_SEC = float(os.environ.get("TRACER_RAM_PROPRIO_STALE_SEC", "1.0"))
ODOM_STALE_SEC = float(os.environ.get("TRACER_RAM_ODOM_STALE_SEC", "1.0"))
MPC_STALE_SEC = float(os.environ.get("TRACER_RAM_MPC_STALE_SEC", "2.0"))

UDP_TIMEOUT_SEC = float(os.environ.get("TRACER_RAM_UDP_TIMEOUT_SEC", "0.25"))


class OnlineRamUdpClient(Node):
    def __init__(self) -> None:
        super().__init__("tracer_online_ram_udp_client")

        self.proprio: Optional[list[float]] = None
        self.mpc: Optional[list[float]] = None
        self.odom: Optional[list[float]] = None

        self.proprio_wall = 0.0
        self.mpc_wall = 0.0
        self.odom_wall = 0.0

        self.proprio_seq = 0
        self.last_used_proprio_seq = -1

        self.window = deque(maxlen=WINDOW)

        self.last_log_time = 0.0
        self.request_count = 0
        self.success_count = 0

        self.control_risk_ema = None
        self.control_risk_alpha = float(os.environ.get("TRACER_RAM_RISK_EMA_ALPHA", "0.25"))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(UDP_TIMEOUT_SEC)

        self.pub_risk = self.create_publisher(Float64MultiArray, "/tracer/ram_risk", 10)
        self.pub_latent = self.create_publisher(Float64MultiArray, "/tracer/ram_latent", 10)

        self.sub_proprio = self.create_subscription(
            Float64MultiArray,
            "/tracer/proprio_vector",
            self.on_proprio,
            10,
        )
        self.sub_mpc = self.create_subscription(
            Float64MultiArray,
            "/tracer/mpc_reference",
            self.on_mpc,
            10,
        )
        self.sub_odom = self.create_subscription(
            Float64MultiArray,
            "/tracer/robot_odom_flat",
            self.on_odom,
            10,
        )

        period = 1.0 / max(RAM_HZ, 1e-6)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(
            f"RAM UDP client started: server={SERVER_HOST}:{SERVER_PORT}, "
            f"hz={RAM_HZ}, window={WINDOW}, input_dim={INPUT_DIM}, "
            f"require_odom={REQUIRE_ODOM}, "
            f"freshness proprio/mpc/odom={PROPRIO_STALE_SEC}/{MPC_STALE_SEC}/{ODOM_STALE_SEC}s"
        )

    def on_proprio(self, msg: Float64MultiArray) -> None:
        data = list(msg.data)
        if len(data) != PROPRIO_DIM:
            self.get_logger().warn(f"ignore /tracer/proprio_vector len={len(data)} expected={PROPRIO_DIM}")
            return
        self.proprio = data
        self.proprio_wall = time.time()
        self.proprio_seq += 1

    def on_mpc(self, msg: Float64MultiArray) -> None:
        data = list(msg.data)
        if len(data) != MPC_DIM:
            self.get_logger().warn(f"ignore /tracer/mpc_reference len={len(data)} expected={MPC_DIM}")
            return
        self.mpc = data
        self.mpc_wall = time.time()

    def on_odom(self, msg: Float64MultiArray) -> None:
        data = list(msg.data)
        if len(data) != ODOM_DIM:
            self.get_logger().warn(f"ignore /tracer/robot_odom_flat len={len(data)} expected={ODOM_DIM}")
            return
        self.odom = data
        self.odom_wall = time.time()

    def log_throttled(self, text: str, interval: float = 2.0) -> None:
        now = time.time()
        if now - self.last_log_time > interval:
            self.get_logger().warn(text)
            self.last_log_time = now

    def clear_window_if_needed(self, reason: str) -> None:
        if self.window:
            self.window.clear()
            self.control_risk_ema = None
            self.get_logger().warn(f"cleared RAM window: {reason}")

    def on_timer(self) -> None:
        now = time.time()

        missing = []
        if self.proprio is None:
            missing.append("/tracer/proprio_vector")
        if self.mpc is None:
            missing.append("/tracer/mpc_reference")
        if REQUIRE_ODOM and self.odom is None:
            missing.append("/tracer/robot_odom_flat")

        if missing:
            self.log_throttled("waiting for " + ", ".join(missing))
            return

        if now - self.proprio_wall > PROPRIO_STALE_SEC:
            self.clear_window_if_needed("stale /tracer/proprio_vector")
            self.log_throttled(
                f"waiting for fresh /tracer/proprio_vector "
                f"age={now - self.proprio_wall:.2f}s > {PROPRIO_STALE_SEC:.2f}s"
            )
            return

        if now - self.mpc_wall > MPC_STALE_SEC:
            self.clear_window_if_needed("stale /tracer/mpc_reference")
            self.log_throttled(
                f"waiting for fresh /tracer/mpc_reference "
                f"age={now - self.mpc_wall:.2f}s > {MPC_STALE_SEC:.2f}s"
            )
            return

        if REQUIRE_ODOM:
            if now - self.odom_wall > ODOM_STALE_SEC:
                self.clear_window_if_needed("stale /tracer/robot_odom_flat")
                self.log_throttled(
                    f"waiting for fresh /tracer/robot_odom_flat "
                    f"age={now - self.odom_wall:.2f}s > {ODOM_STALE_SEC:.2f}s"
                )
                return
            odom = list(self.odom)
        else:
            if self.odom is not None and now - self.odom_wall <= ODOM_STALE_SEC:
                odom = list(self.odom)
            else:
                odom = [0.0] * ODOM_DIM

        # Append only when a new proprio message arrived.
        # This prevents repeatedly filling the RAM window with duplicate cached states.
        if self.proprio_seq == self.last_used_proprio_seq:
            return
        self.last_used_proprio_seq = self.proprio_seq

        # The first MPC reference entry is a publisher-side counter.
        # It grows unbounded and is not a physical command, so remove it for RAM input only.
        mpc_for_ram = list(self.mpc)
        if len(mpc_for_ram) >= 1:
            mpc_for_ram[0] = 0.0

        x = list(self.proprio) + mpc_for_ram + odom

        if len(x) != INPUT_DIM:
            self.get_logger().warn(f"bad RAM input dim: {len(x)} expected={INPUT_DIM}")
            return

        self.window.append(x)

        if len(self.window) < WINDOW:
            if len(self.window) == 1 or len(self.window) % 10 == 0:
                self.get_logger().info(f"warming RAM window: {len(self.window)}/{WINDOW}")
            return

        payload = {
            "window": list(self.window),
            "client_time": now,
        }

        try:
            raw = json.dumps(payload).encode("utf-8")
            self.sock.sendto(raw, (SERVER_HOST, SERVER_PORT))
            data, _addr = self.sock.recvfrom(65535)
            result = json.loads(data.decode("utf-8"))
        except Exception as e:
            self.get_logger().warn(f"RAM UDP request failed: {e}")
            return

        if not result.get("ok", False):
            self.get_logger().warn(f"RAM server returned error: {result}")
            return

        head_probs = result.get("head_probs", {}) or {}

        future_risk = float(head_probs.get("future_risk", 0.0))
        future_slip = float(head_probs.get("future_slip", 0.0))
        future_invalid = float(head_probs.get("future_invalid", 0.0))
        run_valid = float(head_probs.get("run_valid", 0.0))
        run_fallen = float(head_probs.get("run_fallen", 0.0))
        recovery_needed = float(head_probs.get("recovery_needed", 0.0))

        sigma_mean = float(result.get("sigma_mean", 0.0))
        rho_norm = float(result.get("rho_norm", 0.0))
        style_score_pred = float(result.get("style_score_pred", 0.0))

        # Control-safe risk intentionally ignores future_slip/future_invalid/run_valid for now.
        # Those heads are diagnostics but too noisy for direct runtime gating in RAM V0.
        control_risk_raw = (
            0.50 * future_risk
            + 0.30 * run_fallen
            + 0.20 * recovery_needed
        )
        if self.control_risk_ema is None:
            self.control_risk_ema = control_risk_raw
        else:
            a = self.control_risk_alpha
            self.control_risk_ema = a * control_risk_raw + (1.0 - a) * self.control_risk_ema

        input_debug = result.get("input_debug", {}) or {}
        z_abs_mean = float(input_debug.get("z_abs_mean", 0.0))
        z_abs_p95 = float(input_debug.get("z_abs_p95", 0.0))
        z_abs_max = float(input_debug.get("z_abs_max", 0.0))
        last_z_abs_max = float(input_debug.get("last_z_abs_max", 0.0))

        risk_msg = Float64MultiArray()
        risk_msg.data = [
            future_risk,
            future_slip,
            future_invalid,
            run_valid,
            run_fallen,
            recovery_needed,
            sigma_mean,
            rho_norm,
            style_score_pred,
            float(control_risk_raw),
            float(self.control_risk_ema),
        ]
        self.pub_risk.publish(risk_msg)

        rho = list(result.get("rho", []))
        sigma = list(result.get("sigma", []))
        latent_msg = Float64MultiArray()
        latent_msg.data = [float(v) for v in rho + sigma]
        self.pub_latent.publish(latent_msg)

        self.success_count += 1

        log_now = time.time()
        if log_now - self.last_log_time > 1.0:
            self.get_logger().info(
                f"RAM risk={future_risk:.3f} "
                f"slip={future_slip:.3f} "
                f"invalid={future_invalid:.3f} "
                f"valid={run_valid:.3f} "
                f"fallen={run_fallen:.3f} "
                f"recovery={recovery_needed:.3f} "
                f"sigma={sigma_mean:.3f} "
                f"rho_norm={rho_norm:.3f} "
                f"score={style_score_pred:.2f} "
                f"ctrl_risk={control_risk_raw:.3f} "
                f"ctrl_ema={float(self.control_risk_ema):.3f} "
                f"zmean={z_abs_mean:.2f} "
                f"z95={z_abs_p95:.2f} "
                f"zmax={z_abs_max:.2f} "
                f"last_zmax={last_z_abs_max:.2f}"
            )
            self.last_log_time = log_now


def main() -> None:
    rclpy.init()
    node = OnlineRamUdpClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
