#!/usr/bin/env python3
from __future__ import annotations

import json
import socket
import time


def request(payload, host="127.0.0.1", port=50430, timeout=2.0):
    data = json.dumps(payload).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        sock.sendto(data, (host, port))
        resp, _ = sock.recvfrom(65535)
        return json.loads(resp.decode("utf-8"))
    finally:
        sock.close()


def main():
    # objective_selector_v2 input_dim = 15
    # [terrain onehot(3), duration, vx/target, enable, h_delta, clr_delta,
    #  gate_level, gate_action, gate_override, ram_fallen, ram_recovery,
    #  success, preference_score_norm]
    objective_x_flat = [
        1.0, 0.0, 0.0,
        1.0,
        0.5,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.8,
    ]

    # RAM shadow v2 input_dim = 570 = 30 * 19.
    # Safe flat window: low speed-risk/intervention, fast gait, high confidence.
    per_step = [
        0.28,   # final_vx
        0.0,    # final_yaw
        0.295,  # final_body_height
        0.030,  # final_clearance
        1.0,    # final_enable
        0.0,    # rule_gms_code fast / 5
        0.0,    # learned_gms_code fast / 5
        0.999,  # learned_gms_prob
        0.0,    # gms_disagree
        0.55,   # learned_beta_motion
        0.25,   # learned_beta_stability
        0.20,   # learned_beta_energy
        0.0,    # learned_ram_intervention_score
        0.0,    # learned_ram_future_override_mean
        1.0/3.0, # rule_ram_level_code stable / 3
        1.0/4.0, # rule_gate_action_code keep / 4
        0.0,    # rule_control_risk
        0.0,    # rule_fallen_prob_weak
        0.1,    # rule_sigma_mean
    ]
    ram_x_flat = per_step * 30

    # GMS classifier v1 input_dim = 22.
    # This is a flat/fast-like proxy.
    gms_x_flat = [
        0.28, 0.0, 0.295, 0.030, 1.0,
        0.55, 0.25, 0.20,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.1, 0.0,
        1.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ]

    payload = {
        "request_id": f"smoke_v3_flat_{int(time.time())}",
        "objective_x": objective_x_flat,
        "ram_x": ram_x_flat,
        "gms_x": gms_x_flat,
    }

    print(json.dumps(request(payload), indent=2))


if __name__ == "__main__":
    main()
