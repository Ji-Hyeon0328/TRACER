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


def fmt(v):
    return "NA" if v is None else f"{float(v):.6f}"


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


def norm_probs(counts, action_names, alpha):
    total = sum(counts.get(a, 0) for a in action_names)
    denom = total + alpha * len(action_names)
    if denom <= 0:
        return {a: 1.0 / len(action_names) for a in action_names}
    return {a: (counts.get(a, 0) + alpha) / denom for a in action_names}


def top_action_from_probs(probs):
    return max(probs.items(), key=lambda kv: kv[1])[0]


def action_id_by_name(actions):
    return {a["name"]: int(a["id"]) for a in actions}


def predict_action(row, policy):
    context = row.get("context", "unknown") or "unknown"
    x = ff(row.get("x"), None)
    y = ff(row.get("y"), 0.0)
    abs_y = abs(y) if y is not None else 0.0

    guards = policy["guards"]

    # Existing stop/hold guard must dominate.
    if context == "goal_flat":
        return "goal_hold", "goal_context_guard"
    if x is not None and x >= float(guards["goal_x_threshold"]):
        return "goal_hold", "goal_x_guard"

    # Soft lateral guard: conservative only when drift is clearly large.
    if (
        abs_y >= float(guards["lateral_abs_y_threshold"])
        and context in guards["lateral_guard_contexts"]
    ):
        return "lateral_recovery_soft", "lateral_guard"

    ctx_policy = policy["context_policy"].get(context)
    if ctx_policy:
        return ctx_policy["top_action_name"], "context_top1"

    return policy["fallback_action_name"], "fallback"


ap = argparse.ArgumentParser()
ap.add_argument("--teacher-csv", default="datasets/phase_j/j1_meta_action_teacher_v0.csv")
ap.add_argument("--bank-json", default="configs/phase_j/meta_action_bank_v0.json")
ap.add_argument("--out-json", default="models/phase_j/j2_meta_action_tabular_policy_v0.json")
ap.add_argument("--out-eval", default="datasets/phase_j/j2_meta_action_tabular_policy_eval_v0.csv")
ap.add_argument("--out-md", default="reports/phase_j/j2_meta_action_tabular_policy_summary_v0.md")
ap.add_argument("--margin-threshold", type=float, default=0.05)
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--goal-x-threshold", type=float, default=7.90)
ap.add_argument("--lateral-abs-y-threshold", type=float, default=0.45)
args = ap.parse_args()

teacher_path = Path(args.teacher_csv)
bank_path = Path(args.bank_json)

if not teacher_path.exists():
    raise SystemExit(f"[TRACER] missing teacher csv: {teacher_path}")
if not bank_path.exists():
    raise SystemExit(f"[TRACER] missing action bank json: {bank_path}")

rows = list(csv.DictReader(open(teacher_path)))
bank = json.loads(bank_path.read_text())
actions = bank["actions"]
action_names = [a["name"] for a in actions]
id_map = action_id_by_name(actions)

if not rows:
    raise SystemExit("[TRACER] empty teacher csv")

high_conf_rows = [
    r for r in rows
    if ff(r.get("top2_margin"), 0.0) >= args.margin_threshold
]

train_rows = high_conf_rows if high_conf_rows else rows

global_counts = Counter(r["best_action_name"] for r in train_rows)

# Build both high-confidence and all-row context counts.
# Important: some contexts, especially flat, may have low top-2 margins and disappear
# from high_conf_rows. In that case, falling back to a global action is unsafe.
context_counts_high = defaultdict(Counter)
for r in train_rows:
    context_counts_high[r.get("context", "unknown") or "unknown"][r["best_action_name"]] += 1

context_counts_all = defaultdict(Counter)
for r in rows:
    context_counts_all[r.get("context", "unknown") or "unknown"][r["best_action_name"]] += 1

global_probs = norm_probs(global_counts, action_names, args.alpha)
fallback_action_name = top_action_from_probs(global_probs)

context_policy = {}
all_contexts = sorted(set(context_counts_high.keys()) | set(context_counts_all.keys()))
for ctx in all_contexts:
    if sum(context_counts_high[ctx].values()) > 0:
        counts = context_counts_high[ctx]
        source = "high_conf"
    else:
        counts = context_counts_all[ctx]
        source = "all_rows_context_fallback"

    probs = norm_probs(counts, action_names, args.alpha)
    top_name = top_action_from_probs(probs)
    context_policy[ctx] = {
        "rows": sum(counts.values()),
        "source": source,
        "top_action_name": top_name,
        "top_action_id": id_map[top_name],
        "action_probs": [
            {
                "action_id": id_map[name],
                "action_name": name,
                "prob": probs[name],
                "count": counts.get(name, 0),
            }
            for name in action_names
        ],
    }

policy = {
    "phase": "J2",
    "name": "TRACER meta-action tabular policy v0",
    "safe_claim": "Supervised/tabular initialization from J1 pseudo-labels; not active control and not final RL.",
    "teacher_csv": str(teacher_path),
    "bank_json": str(bank_path),
    "training": {
        "rows_total": len(rows),
        "rows_high_conf": len(high_conf_rows),
        "rows_used": len(train_rows),
        "context_fallback_policy": "Use high-confidence rows per context when available; otherwise use all rows for that context.",
        "top2_margin_threshold": args.margin_threshold,
        "dirichlet_alpha": args.alpha,
    },
    "guards": {
        "goal_context": "context == goal_flat -> goal_hold",
        "goal_x_threshold": args.goal_x_threshold,
        "lateral_abs_y_threshold": args.lateral_abs_y_threshold,
        "lateral_guard_contexts": ["rough", "downslope", "unknown"],
        "lateral_guard_action": "lateral_recovery_soft",
    },
    "fallback_action_name": fallback_action_name,
    "fallback_action_id": id_map[fallback_action_name],
    "global_action_probs": [
        {
            "action_id": id_map[name],
            "action_name": name,
            "prob": global_probs[name],
            "count": global_counts.get(name, 0),
        }
        for name in action_names
    ],
    "context_policy": context_policy,
}

# Evaluation against all J1 teacher rows.
eval_rows = []
correct = 0
correct_high_conf = 0
n_high_conf = 0
by_context = defaultdict(lambda: {"n": 0, "correct": 0})
by_reason = Counter()
confusion = Counter()

for r in rows:
    pred_name, reason = predict_action(r, policy)
    true_name = r["best_action_name"]
    ok = pred_name == true_name

    if ok:
        correct += 1

    margin = ff(r.get("top2_margin"), 0.0)
    if margin >= args.margin_threshold:
        n_high_conf += 1
        if ok:
            correct_high_conf += 1

    ctx = r.get("context", "unknown") or "unknown"
    by_context[ctx]["n"] += 1
    by_context[ctx]["correct"] += int(ok)
    by_reason[reason] += 1
    confusion[(ctx, true_name, pred_name)] += 1

    eval_rows.append({
        "row_idx": r.get("row_idx", ""),
        "t_wall": r.get("t_wall", ""),
        "context": ctx,
        "x": r.get("x", ""),
        "y": r.get("y", ""),
        "true_action_id": r.get("best_action_id", ""),
        "true_action_name": true_name,
        "pred_action_id": id_map[pred_name],
        "pred_action_name": pred_name,
        "pred_reason": reason,
        "correct": "1" if ok else "0",
        "top2_margin": r.get("top2_margin", ""),
    })

acc = correct / len(rows)
acc_high = correct_high_conf / n_high_conf if n_high_conf else None

out_json = Path(args.out_json)
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(policy, indent=2))

out_eval = Path(args.out_eval)
out_eval.parent.mkdir(parents=True, exist_ok=True)
with open(out_eval, "w", newline="") as f:
    fields = [
        "row_idx", "t_wall", "context", "x", "y",
        "true_action_id", "true_action_name",
        "pred_action_id", "pred_action_name", "pred_reason",
        "correct", "top2_margin",
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(eval_rows)

out_md = Path(args.out_md)
out_md.parent.mkdir(parents=True, exist_ok=True)

margins = [ff(r.get("top2_margin")) for r in rows]
margins_high = [ff(r.get("top2_margin")) for r in high_conf_rows]
st_all = stats(margins)
st_high = stats(margins_high)

with open(out_md, "w") as f:
    f.write("# TRACER Phase-J2 Meta-Action Tabular Policy v0\n\n")
    f.write("This trains a simple tabular/context-guarded policy from J1 meta-action pseudo-labels.\n\n")
    f.write(f"- teacher csv: `{teacher_path}`\n")
    f.write(f"- action bank: `{bank_path}`\n")
    f.write(f"- policy json: `{out_json}`\n")
    f.write(f"- eval csv: `{out_eval}`\n")
    f.write(f"- total rows: `{len(rows)}`\n")
    f.write(f"- high-confidence rows: `{len(high_conf_rows)}`\n")
    f.write(f"- rows used for policy table: `{len(train_rows)}`\n")
    f.write(f"- margin threshold: `{args.margin_threshold}`\n")
    f.write(f"- alpha: `{args.alpha}`\n\n")

    f.write("## Accuracy against J1 teacher\n\n")
    f.write("| split | rows | accuracy |\n")
    f.write("|---|---:|---:|\n")
    f.write(f"| all | {len(rows)} | {acc:.6f} |\n")
    f.write(f"| high_conf | {n_high_conf} | {fmt(acc_high)} |\n")

    f.write("\n## Margin summary\n\n")
    f.write("| split | n | mean | std | min | max |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    f.write(f"| all | {st_all['n']} | {fmt(st_all['mean'])} | {fmt(st_all['std'])} | {fmt(st_all['min'])} | {fmt(st_all['max'])} |\n")
    f.write(f"| high_conf | {st_high['n']} | {fmt(st_high['mean'])} | {fmt(st_high['std'])} | {fmt(st_high['min'])} | {fmt(st_high['max'])} |\n")

    f.write("\n## Context policy top actions\n\n")
    f.write("| context | rows used | source | top action | top action id |\n")
    f.write("|---|---:|---|---|---:|\n")
    for ctx in sorted(context_policy):
        cp = context_policy[ctx]
        f.write(f"| {ctx} | {cp['rows']} | {cp['source']} | {cp['top_action_name']} | {cp['top_action_id']} |\n")

    f.write("\n## Accuracy by context\n\n")
    f.write("| context | rows | correct | accuracy |\n")
    f.write("|---|---:|---:|---:|\n")
    for ctx in sorted(by_context):
        n = by_context[ctx]["n"]
        c = by_context[ctx]["correct"]
        f.write(f"| {ctx} | {n} | {c} | {c / n if n else 0.0:.6f} |\n")

    f.write("\n## Prediction reasons\n\n")
    f.write("| reason | rows |\n")
    f.write("|---|---:|\n")
    for reason, cnt in by_reason.most_common():
        f.write(f"| {reason} | {cnt} |\n")

    f.write("\n## Main confusions\n\n")
    f.write("| context | true | pred | rows |\n")
    f.write("|---|---|---|---:|\n")
    for (ctx, true_name, pred_name), cnt in confusion.most_common(20):
        if true_name != pred_name:
            f.write(f"| {ctx} | {true_name} | {pred_name} | {cnt} |\n")

    f.write("\n## Safe interpretation\n\n")
    f.write("- J2 is a supervised/tabular initialization from J1 pseudo-labels.\n")
    f.write("- J2 does not modify active control.\n")
    f.write("- This policy can be used as the first shadow meta-action selector.\n")
    f.write("- The next step should apply this policy in shadow mode to produce a_HL without changing `/tracer/mpc_reference`.\n")

print(f"[TRACER] wrote {out_json}")
print(f"[TRACER] wrote {out_eval}")
print(f"[TRACER] wrote {out_md}")
print(f"[TRACER] acc_all={acc:.6f} acc_high_conf={fmt(acc_high)} rows={len(rows)}")
