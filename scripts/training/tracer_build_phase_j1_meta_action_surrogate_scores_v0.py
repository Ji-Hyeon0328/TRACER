#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, pstdev


def ff(x, default=None):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def stats(vals):
    vals = [v for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def fmt(v):
    return "NA" if v is None else f"{float(v):.6f}"


def get_beta(row):
    # Prefer D7 actual beta, then pred beta.
    candidates = [
        ("actual_beta_motion", "actual_beta_stability", "actual_beta_energy"),
        ("pred_beta_motion", "pred_beta_stability", "pred_beta_energy"),
        ("raw_beta_motion", "raw_beta_stability", "raw_beta_energy"),
        ("prior_beta_motion", "prior_beta_stability", "prior_beta_energy"),
    ]
    for cols in candidates:
        vals = [ff(row.get(c)) for c in cols]
        if all(v is not None for v in vals):
            s = sum(vals)
            if s > 1e-12:
                return [v / s for v in vals], cols
    return None, None


def action_components(action, context, beta, row):
    th = action["theta"]
    name = action["name"]

    vx_scale = float(th.get("vx_scale", 1.0))
    clearance_delta = float(th.get("clearance_delta", 0.0))
    body_h_delta = float(th.get("body_h_delta", 0.0))
    stability_bias = float(th.get("stability_bias", 0.0))
    energy_bias = float(th.get("energy_bias", 0.0))
    hold = bool(th.get("hold_override", False))

    x = ff(row.get("x"), 0.0)
    y = ff(row.get("y"), 0.0)
    abs_y = abs(y) if y is not None else 0.0

    ram_slip = ff(row.get("ram_slip_proxy_mean"), 0.0)
    ram_rough = ff(row.get("ram_roughness_proxy_mean"), 0.0)
    ram_sigma = ff(row.get("ram_sigma_mean"), 0.0)

    near_goal = (context == "goal_flat") or (x is not None and x >= 7.90)
    intended = set(action.get("intended_contexts", []))
    intent_match = context in intended

    # Abstract objective preferences in [0,1].
    motion_pref = clamp01((vx_scale - 0.65) / max(1.08 - 0.65, 1e-6))

    stability_pref = clamp01(
        0.45
        + stability_bias
        + 0.35 * max(0.0, 1.0 - vx_scale)
        + 0.35 * max(0.0, clearance_delta) / 0.012
        - 0.10 * max(0.0, vx_scale - 1.0)
    )

    energy_pref = clamp01(
        0.55
        + energy_bias
        + 0.30 * max(0.0, 1.0 - vx_scale)
        - 0.35 * max(0.0, clearance_delta) / 0.012
        - 0.10 * abs(body_h_delta) / 0.010
    )

    penalty = 0.0
    bonus = 0.0

    if intent_match:
        bonus += 0.12
    else:
        penalty += 0.08

    if hold:
        if near_goal:
            motion_pref = 0.20
            stability_pref = 0.85
            energy_pref = 0.75
            bonus += 0.50
        else:
            penalty += 2.00

    if context == "goal_flat" and not hold:
        penalty += 0.25

    if context in ("rough", "downslope"):
        if name in ("fast_motion", "energy_saver"):
            penalty += 0.25
        if stability_bias >= 0.20:
            bonus += 0.10
        if clearance_delta < 0.0:
            penalty += 0.15

    if context == "rough" and "rough" in intended:
        bonus += 0.08

    if context == "downslope" and name in ("downslope_stable", "lateral_recovery_soft"):
        bonus += 0.12

    if context in ("flat", "start_flat") and name in ("downslope_stable", "lateral_recovery_soft"):
        penalty += 0.12

    if abs_y > 0.25:
        if name == "lateral_recovery_soft":
            bonus += 0.25
        if vx_scale > 1.0:
            penalty += 0.20

    # Legacy RAM scale is small, so amplify softly.
    stress = max(
        clamp01(ram_slip / 0.25),
        clamp01(ram_rough / 0.40),
        clamp01(ram_sigma / 0.35),
    )

    if stress > 0.60:
        if stability_bias >= 0.20:
            bonus += 0.12 * stress
        if vx_scale > 1.0:
            penalty += 0.15 * stress

    beta_motion, beta_stability, beta_energy = beta
    objective_score = (
        beta_motion * motion_pref
        + beta_stability * stability_pref
        + beta_energy * energy_pref
    )

    final_score = objective_score + bonus - penalty

    return {
        "motion_pref": motion_pref,
        "stability_pref": stability_pref,
        "energy_pref": energy_pref,
        "objective_score": objective_score,
        "bonus": bonus,
        "penalty": penalty,
        "stress": stress,
        "final_score": final_score,
        "near_goal": near_goal,
        "intent_match": intent_match,
    }


ap = argparse.ArgumentParser()
ap.add_argument("--d7-csv", required=True)
ap.add_argument("--bank-json", default="configs/phase_j/meta_action_bank_v0.json")
ap.add_argument("--out-scores", default="datasets/phase_j/j1_meta_action_surrogate_scores_v0.csv")
ap.add_argument("--out-teacher", default="datasets/phase_j/j1_meta_action_teacher_v0.csv")
ap.add_argument("--out-md", default="reports/phase_j/j1_meta_action_surrogate_scorer_summary_v0.md")
ap.add_argument("--out-json", default="models/phase_j/j1_meta_action_surrogate_scorer_v0.json")
args = ap.parse_args()

d7_path = Path(args.d7_csv)
bank_path = Path(args.bank_json)

if not d7_path.exists():
    raise SystemExit(f"[TRACER] missing D7 csv: {d7_path}")
if not bank_path.exists():
    raise SystemExit(f"[TRACER] missing action bank: {bank_path}")

rows = list(csv.DictReader(open(d7_path)))
bank = json.loads(bank_path.read_text())
actions = bank["actions"]

score_rows = []
teacher_rows = []
beta_source_counter = Counter()

for idx, row in enumerate(rows):
    beta, beta_cols = get_beta(row)
    if beta is None:
        continue
    beta_source_counter[",".join(beta_cols)] += 1

    context = row.get("context", "unknown") or "unknown"
    x = ff(row.get("x"), "")
    y = ff(row.get("y"), "")

    candidates = []
    for action in actions:
        comp = action_components(action, context, beta, row)
        out = {
            "row_idx": idx,
            "t_wall": row.get("t_wall", ""),
            "context": context,
            "x": row.get("x", ""),
            "y": row.get("y", ""),
            "beta_motion": f"{beta[0]:.8f}",
            "beta_stability": f"{beta[1]:.8f}",
            "beta_energy": f"{beta[2]:.8f}",
            "action_id": action["id"],
            "action_name": action["name"],
            "motion_pref": f"{comp['motion_pref']:.8f}",
            "stability_pref": f"{comp['stability_pref']:.8f}",
            "energy_pref": f"{comp['energy_pref']:.8f}",
            "objective_score": f"{comp['objective_score']:.8f}",
            "bonus": f"{comp['bonus']:.8f}",
            "penalty": f"{comp['penalty']:.8f}",
            "stress": f"{comp['stress']:.8f}",
            "final_score": f"{comp['final_score']:.8f}",
            "near_goal": "1" if comp["near_goal"] else "0",
            "intent_match": "1" if comp["intent_match"] else "0",
        }
        score_rows.append(out)
        candidates.append(out)

    candidates.sort(key=lambda r: float(r["final_score"]), reverse=True)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    margin = float(best["final_score"]) - (float(second["final_score"]) if second else 0.0)

    teacher_rows.append({
        "row_idx": idx,
        "t_wall": row.get("t_wall", ""),
        "context": context,
        "x": row.get("x", ""),
        "y": row.get("y", ""),
        "beta_motion": f"{beta[0]:.8f}",
        "beta_stability": f"{beta[1]:.8f}",
        "beta_energy": f"{beta[2]:.8f}",
        "best_action_id": best["action_id"],
        "best_action_name": best["action_name"],
        "best_score": best["final_score"],
        "second_action_id": "" if second is None else second["action_id"],
        "second_action_name": "" if second is None else second["action_name"],
        "top2_margin": f"{margin:.8f}",
    })

out_scores = Path(args.out_scores)
out_scores.parent.mkdir(parents=True, exist_ok=True)
with open(out_scores, "w", newline="") as f:
    fields = [
        "row_idx", "t_wall", "context", "x", "y",
        "beta_motion", "beta_stability", "beta_energy",
        "action_id", "action_name",
        "motion_pref", "stability_pref", "energy_pref",
        "objective_score", "bonus", "penalty", "stress",
        "final_score", "near_goal", "intent_match",
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(score_rows)

out_teacher = Path(args.out_teacher)
out_teacher.parent.mkdir(parents=True, exist_ok=True)
with open(out_teacher, "w", newline="") as f:
    fields = [
        "row_idx", "t_wall", "context", "x", "y",
        "beta_motion", "beta_stability", "beta_energy",
        "best_action_id", "best_action_name", "best_score",
        "second_action_id", "second_action_name", "top2_margin",
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(teacher_rows)

model = {
    "phase": "J1",
    "purpose": "Offline surrogate scorer for J0 meta-action bank",
    "safe_claim": "Pseudo-label generator for contextual-bandit/offline-RL preparation; not active control.",
    "d7_csv": str(d7_path),
    "bank_json": str(bank_path),
    "outputs": {
        "scores": str(out_scores),
        "teacher": str(out_teacher),
    },
    "scoring_policy": {
        "uses": ["D7 beta", "terrain context", "legacy RAM proxy", "x/y drift", "J0 action metadata"],
        "not_uses": ["online exploration", "real action rollout outcome"],
        "note": "Scores are heuristic pseudo-labels and must be validated by shadow/rollout tests."
    }
}

out_json = Path(args.out_json)
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(model, indent=2))

# Summaries
teacher_counts = Counter(r["best_action_name"] for r in teacher_rows)
ctx_counts = defaultdict(Counter)
margins = []
for r in teacher_rows:
    ctx_counts[r["context"]][r["best_action_name"]] += 1
    m = ff(r["top2_margin"])
    if m is not None:
        margins.append(m)

out_md = Path(args.out_md)
out_md.parent.mkdir(parents=True, exist_ok=True)
with open(out_md, "w") as f:
    f.write("# TRACER Phase-J1 Meta-Action Surrogate Scorer v0\n\n")
    f.write("This scores the J0 discrete meta-action bank using D7 beta, context, RAM proxy, and drift-style runtime features.\n\n")
    f.write(f"- D7 csv: `{d7_path}`\n")
    f.write(f"- action bank: `{bank_path}`\n")
    f.write(f"- scores csv: `{out_scores}`\n")
    f.write(f"- teacher csv: `{out_teacher}`\n")
    f.write(f"- scorer json: `{out_json}`\n")
    f.write(f"- D7 rows used: `{len(teacher_rows)}`\n")
    f.write(f"- score rows: `{len(score_rows)}`\n")
    f.write(f"- actions: `{len(actions)}`\n\n")

    f.write("## Beta source columns\n\n")
    f.write("| beta columns | rows |\n")
    f.write("|---|---:|\n")
    for k, v in beta_source_counter.items():
        f.write(f"| {k} | {v} |\n")

    f.write("\n## Teacher action distribution\n\n")
    f.write("| action | rows |\n")
    f.write("|---|---:|\n")
    for k, v in teacher_counts.most_common():
        f.write(f"| {k} | {v} |\n")

    f.write("\n## Teacher action by context\n\n")
    f.write("| context | action | rows |\n")
    f.write("|---|---|---:|\n")
    for ctx in sorted(ctx_counts):
        for action_name, cnt in ctx_counts[ctx].most_common():
            f.write(f"| {ctx} | {action_name} | {cnt} |\n")

    st = stats(margins)
    f.write("\n## Top-2 margin summary\n\n")
    f.write("| n | mean | std | min | max |\n")
    f.write("|---:|---:|---:|---:|---:|\n")
    f.write(f"| {st['n']} | {fmt(st['mean'])} | {fmt(st['std'])} | {fmt(st['min'])} | {fmt(st['max'])} |\n")

    f.write("\n## Safe interpretation\n\n")
    f.write("- J1 produces heuristic pseudo-labels for meta-action selection.\n")
    f.write("- J1 does not modify active control.\n")
    f.write("- The labels should be treated as contextual-bandit / offline-RL initialization, not final RL.\n")
    f.write("- If one action dominates too strongly, J2 should add entropy/balancing before training a policy.\n")

print(f"[TRACER] wrote {out_scores}")
print(f"[TRACER] wrote {out_teacher}")
print(f"[TRACER] wrote {out_json}")
print(f"[TRACER] wrote {out_md}")
print(f"[TRACER] teacher_rows={len(teacher_rows)} score_rows={len(score_rows)}")
print("[TRACER] teacher distribution:")
for k, v in teacher_counts.most_common():
    print(f"  {k}: {v}")
