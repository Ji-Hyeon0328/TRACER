#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path


def latest_model(pattern, filename):
    roots = sorted(Path("artifacts").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for r in roots:
        f = r / filename
        if f.is_file():
            return str(f)
    return ""


def run_json(cmd):
    raw = subprocess.check_output(cmd, text=True)
    return json.loads(raw)


def ff(x, default=0.0):
    try:
        v = float(x)
        return v
    except Exception:
        return default


def key_string(world, profile, vx, near, slow, h, clr):
    return (
        f"{world}::{profile}"
        f"::vx={round(ff(vx), 4):.4f}"
        f"::near={round(ff(near), 4):.4f}"
        f"::slow={round(ff(slow), 4):.4f}"
        f"::h={round(ff(h), 4):.4f}"
        f"::clr={round(ff(clr), 4):.4f}"
    )


def load_uncertainty(registry_path, world, profile):
    if not registry_path or not Path(registry_path).is_file():
        return {}

    with open(registry_path, "r") as f:
        return json.load(f)


def lookup_uncertainty(registry, world, profile, p):
    if not registry:
        return {}

    k = key_string(
        world,
        profile,
        p.get("vx_far"),
        p.get("vx_near"),
        p.get("goal_slow_distance"),
        p.get("body_height"),
        p.get("swing_clearance"),
    )

    groups = registry.get("groups", {})
    if k in groups:
        g = dict(groups[k])
        g["matched_key"] = k
        g["match_type"] = "exact"
        return g

    # Fallback: nearest same world/profile action.
    best = None
    best_d2 = None
    for _, g in groups.items():
        kk = g.get("key", {})
        if kk.get("world_name") != world:
            continue
        if kk.get("theta5_profile_name") != profile:
            continue

        d2 = 0.0
        for name, pred_name in [
            ("vx_far", "vx_far"),
            ("vx_near", "vx_near"),
            ("goal_slow_distance", "goal_slow_distance"),
            ("body_height", "body_height"),
            ("swing_clearance", "swing_clearance"),
        ]:
            d = ff(kk.get(name)) - ff(p.get(pred_name))
            d2 += d * d

        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best = dict(g)

    if best is not None:
        best["matched_key"] = best.get("key_string")
        best["match_type"] = "nearest_same_world_profile"
        best["action_distance2"] = best_d2
        return best

    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta5-model", default="")
    ap.add_argument("--objective-ram-model", default="")
    ap.add_argument("--risk-state-json", required=True)
    ap.add_argument("--world-name", default="earth")
    ap.add_argument("--goal-distance-ahead", type=float, default=0.5)
    ap.add_argument("--goal-stop-distance", type=float, default=0.15)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--uncertainty-registry", default="")
    args = ap.parse_args()

    theta5_model = args.theta5_model or latest_model(
        "phase_b_theta5_regressor_v0_*",
        "phase_b_theta5_regressor_v0.json",
    )
    obj_ram_model = args.objective_ram_model
    if not obj_ram_model:
        cfg_model = Path("configs/phase_b_objective_ram_bootstrap_v0/current_model.json")
        if cfg_model.is_file():
            obj_ram_model = str(cfg_model)
        else:
            obj_ram_model = latest_model(
                "phase_b_objective_ram_bootstrap_v0_*",
                "phase_b_objective_ram_bootstrap_v0.json",
            )

    uncertainty_registry = args.uncertainty_registry
    if not uncertainty_registry:
        cfg_unc = Path("configs/phase_b_objective_ram_uncertainty_v0/current_registry.json")
        if cfg_unc.is_file():
            uncertainty_registry = str(cfg_unc)

    if not theta5_model:
        raise RuntimeError("theta5 model not found")
    if not obj_ram_model:
        raise RuntimeError("objective/RAM bootstrap model not found")

    theta5 = run_json([
        sys.executable,
        "scripts/runtime/tracer_predict_phase_b_theta5_hybrid_profile_v0.py",
        "--linear-model", theta5_model,
        "--risk-state-json", args.risk_state_json,
        "--world-name", args.world_name,
        "--goal-distance-ahead", str(args.goal_distance_ahead),
        "--goal-stop-distance", str(args.goal_stop_distance),
    ])

    p = theta5["profile"]

    obj_ram = run_json([
        sys.executable,
        "scripts/runtime/tracer_predict_phase_b_objective_ram_bootstrap_v0.py",
        "--model", obj_ram_model,
        "--world-name", args.world_name,
        "--vx-far", str(p["vx_far"]),
        "--vx-near", str(p["vx_near"]),
        "--goal-slow-distance", str(p["goal_slow_distance"]),
        "--body-height", str(p["body_height"]),
        "--swing-clearance", str(p["swing_clearance"]),
    ])

    objective = obj_ram["objective_prediction"]
    ram = obj_ram["ram_prediction"]

    profile_name = p.get("name") or theta5.get("profile", {}).get("name")
    unc_registry = load_uncertainty(uncertainty_registry, args.world_name, profile_name)
    uncertainty = lookup_uncertainty(unc_registry, args.world_name, profile_name, p)

    future_uncertainty = float(uncertainty.get("future_uncertainty", 0.0) or 0.0)
    semantic_uncertainty = float(uncertainty.get("semantic_uncertainty", 0.0) or 0.0)
    high_variability = bool(uncertainty.get("high_variability", False))

    uncertainty_avg_risk = float(uncertainty.get("avg_future_risk", 0.0) or 0.0)
    uncertainty_majority_semantic = uncertainty.get("majority_semantic")

    calibrated_future_risk = max(float(ram.get("future_risk", 0.0) or 0.0), uncertainty_avg_risk)

    calibrated_semantic = objective.get("semantic")
    if high_variability and uncertainty_majority_semantic:
        # Use the observed majority outcome as the effective semantic when
        # repeated rollouts show high variability. This is especially important
        # for soft terrain where the same theta can occasionally reach but then
        # escape severely.
        calibrated_semantic = uncertainty_majority_semantic

    calibrated_future_invalid = bool(ram.get("future_invalid", False))
    if calibrated_semantic == "forward_walk_unreliable_on_soft_terrain":
        calibrated_future_invalid = True

    calibrated_recovery_needed = bool(ram.get("recovery_needed", False))
    if calibrated_future_risk >= 0.60 or future_uncertainty >= 0.50 or high_variability:
        calibrated_recovery_needed = True

    normal_walk_blocked = (
        calibrated_future_invalid
        or calibrated_recovery_needed
        or calibrated_future_risk >= 0.60
        or future_uncertainty >= 0.50
        or high_variability
    )

    out = {
        # Keep top-level compatibility with theta5 runner.
        "model": theta5_model,
        "model_type": theta5.get("model_type"),
        "source": theta5.get("source"),
        "table_key": theta5.get("table_key"),
        "risk_name": theta5.get("risk_name"),
        "risk_state_json": args.risk_state_json,
        "risk_state": theta5.get("risk_state"),
        "world_name": args.world_name,
        "raw_prediction": theta5.get("raw_prediction", {}),
        "profile": p,
        "guard_reasons": theta5.get("guard_reasons", []),

        # New Objective/RAM layer.
        "theta5_prediction": theta5,
        "objective_ram_prediction": obj_ram,
        "runtime_decision": {
            "semantic": objective.get("semantic"),
            "beta_velocity": objective.get("beta_velocity"),
            "beta_stability": objective.get("beta_stability"),
            "beta_energy": objective.get("beta_energy"),
            "future_risk": ram.get("future_risk"),
            "future_invalid": ram.get("future_invalid"),
            "recovery_needed": ram.get("recovery_needed"),
            "stable_reached_pred": ram.get("stable_reached"),
            "approach_success_pred": ram.get("approach_success"),
            "drift_after_approach_pred": ram.get("drift_after_approach"),
            "yaw_saturated_pred": ram.get("yaw_saturated"),
            "future_uncertainty": future_uncertainty,
            "semantic_uncertainty": semantic_uncertainty,
            "high_variability": high_variability,
            "uncertainty_n": uncertainty.get("n"),
            "uncertainty_majority_semantic": uncertainty.get("majority_semantic"),

            "calibrated_semantic": calibrated_semantic,
            "calibrated_future_risk": calibrated_future_risk,
            "calibrated_future_invalid": calibrated_future_invalid,
            "calibrated_recovery_needed": calibrated_recovery_needed,
            "normal_walk_blocked": normal_walk_blocked,
            "uncertainty_avg_future_risk": uncertainty_avg_risk,
        },
        "uncertainty_registry": uncertainty_registry,
        "uncertainty_prediction": uncertainty,
    }

    text = json.dumps(out, indent=2, sort_keys=True)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            f.write(text + "\n")

    print(text)


if __name__ == "__main__":
    main()
