#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


CONTEXT_PRIOR_BETA = {
    "flat":       (0.45, 0.25, 0.30),
    "start_flat": (0.45, 0.25, 0.30),
    "rough":      (0.25, 0.55, 0.20),
    "upslope":    (0.35, 0.40, 0.25),
    "downslope":  (0.25, 0.60, 0.15),
    "goal_flat":  (0.20, 0.50, 0.30),
    "unknown":    (0.33, 0.34, 0.33),
}


def clamp(x, lo=0.0, hi=1.0):
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(lo, min(hi, x))


def fget(d, key, default=0.0):
    try:
        v = d.get(key, default)
        if v in ("", None):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def bget(d, key, default=False):
    v = d.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "y")
    return bool(default)


def normalize3(a, b, c):
    a = max(1e-6, float(a))
    b = max(1e-6, float(b))
    c = max(1e-6, float(c))
    s = a + b + c
    return a / s, b / s, c / s


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def max_abs(xs):
    xs = list(xs)
    return max((abs(x) for x in xs), default=0.0)


def infer_reset_y_from_name(path):
    name = Path(path).name
    table = {
        "m060": -0.60,
        "m030": -0.30,
        "p000":  0.00,
        "p030":  0.30,
        "p060":  0.60,
    }
    for tag, val in table.items():
        if f"yoffset_{tag}_" in name:
            return val
    return 0.0


def load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def resolve_manifest_path(summary_path, manifest_value):
    p = Path(manifest_value)
    if p.exists():
        return p
    root = Path.cwd()
    p2 = root / manifest_value
    if p2.exists():
        return p2
    p3 = summary_path.parent / manifest_value
    if p3.exists():
        return p3
    raise FileNotFoundError(f"manifest not found: {manifest_value}")


def read_manifest(path):
    rows = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows


def find_policy_input_csv(d5_dir):
    d = Path(d5_dir)
    candidates = [
        d / "policy_inputs_v0.csv",
        d / "phase_d5_policy_inputs_v0.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    globs = sorted(d.glob("*policy*input*.csv"))
    if globs:
        return globs[0]

    raise FileNotFoundError(f"policy input csv not found in {d5_dir}")


def read_policy_rows(path):
    rows = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def rollout_by_trial(summary):
    rollouts = summary.get("rollouts") or summary.get("individual_rollouts") or []
    out = {}
    for r in rollouts:
        try:
            t = int(r.get("trial"))
        except Exception:
            continue
        out[t] = r
    return out


def compute_context_sample(summary_path, summary, manifest_row, rollout_row, context, rows):
    goal_x = fget(manifest_row, "goal_x", 8.0)
    lateral_bound = max(1e-6, fget(manifest_row, "lateral_bound", 2.0))
    hold_vx = fget(manifest_row, "hold_vx", 0.025)
    trial = int(float(manifest_row.get("trial", 0)))

    xs = [fget(r, "x") for r in rows]
    ys = [fget(r, "y") for r in rows]

    ref_vx = [fget(r, "ref_vx") for r in rows]
    ref_yaw = [fget(r, "ref_yaw_rate") for r in rows]
    ref_body_h = [fget(r, "ref_body_h") for r in rows]
    ref_clearance = [fget(r, "ref_clearance") for r in rows]
    ref_enable = [fget(r, "ref_enable") for r in rows]

    beta_m = [fget(r, "beta_motion") for r in rows]
    beta_s = [fget(r, "beta_stability") for r in rows]
    beta_e = [fget(r, "beta_energy") for r in rows]

    ram_slip = [fget(r, "ram_slip_proxy") for r in rows]
    ram_rough = [fget(r, "ram_roughness_proxy") for r in rows]
    ram_sigma = [fget(r, "ram_sigma") for r in rows]

    final_x = fget(rollout_row, "final_x", 0.0)
    final_y = fget(rollout_row, "final_y", 0.0)
    rollout_max_abs_y = fget(rollout_row, "max_abs_y", max_abs(ys))
    rollout_mean_abs_y = fget(rollout_row, "mean_abs_y", mean(abs(y) for y in ys))
    hold_drift = fget(rollout_row, "hold_drift", 0.0)

    success = bget(rollout_row, "success")
    goal = bget(rollout_row, "goal")
    startup_failed = bget(rollout_row, "startup_failed")
    out_lane = bget(rollout_row, "out_lane")

    # Objective scores are bootstrap proxies, not true reward decomposition.
    progress_score = clamp(final_x / max(goal_x, 1e-6))
    goal_score = 1.0 if goal else 0.0
    speed_score = clamp(mean(ref_vx) / 0.21)

    motion_score = clamp(
        0.60 * progress_score +
        0.25 * goal_score +
        0.15 * speed_score
    )

    group_max_abs_y = max_abs(ys)
    group_mean_abs_y = mean(abs(y) for y in ys)
    group_lateral_score = 1.0 - clamp(group_max_abs_y / lateral_bound)
    rollout_lateral_score = 1.0 - clamp(rollout_max_abs_y / lateral_bound)
    hold_score = 1.0 - clamp(hold_drift / 0.25)

    stability_score = clamp(
        0.45 * group_lateral_score +
        0.35 * rollout_lateral_score +
        0.20 * hold_score
    )

    vx_effort = clamp(abs(mean(ref_vx)) / 0.24)
    yaw_effort = clamp(abs(mean(ref_yaw)) / 0.20)
    clearance_effort = clamp((mean(ref_clearance) - 0.035) / max(1e-6, 0.065 - 0.035))

    effort_proxy = clamp(
        0.45 * vx_effort +
        0.25 * yaw_effort +
        0.30 * clearance_effort
    )
    energy_score = 1.0 - effort_proxy

    motion_need = 1.0 - motion_score
    stability_need = 1.0 - stability_score
    energy_need = 1.0 - energy_score

    prior = CONTEXT_PRIOR_BETA.get(context, CONTEXT_PRIOR_BETA["unknown"])

    # Bootstrap beta target:
    # - start from current semantic prior
    # - increase motion weight if progress/speed is poor
    # - increase stability weight if lateral/hold behavior is poor
    # - increase energy weight if ref effort proxy is high
    # This is intentionally conservative and explainable.
    raw_motion = prior[0] + 0.50 * motion_need
    raw_stability = prior[1] + 0.80 * stability_need + 0.25 * clamp(group_mean_abs_y / lateral_bound)
    raw_energy = prior[2] + 0.40 * energy_need

    target_beta_m, target_beta_s, target_beta_e = normalize3(
        raw_motion, raw_stability, raw_energy
    )

    return {
        "source_summary": str(summary_path),
        "source_manifest": str(summary.get("manifest", "")),
        "world": manifest_row.get("world", ""),
        "trial": trial,
        "reset_y": infer_reset_y_from_name(summary_path),
        "context": context,
        "n_rows": len(rows),

        "goal_x": goal_x,
        "lateral_bound": lateral_bound,
        "hold_vx": hold_vx,

        "x_start": min(xs) if xs else 0.0,
        "x_end": max(xs) if xs else 0.0,
        "y_mean": mean(ys),
        "mean_abs_y_context": group_mean_abs_y,
        "max_abs_y_context": group_max_abs_y,

        "ref_vx_mean": mean(ref_vx),
        "ref_yaw_rate_mean": mean(ref_yaw),
        "ref_body_h_mean": mean(ref_body_h),
        "ref_clearance_mean": mean(ref_clearance),
        "ref_enable_mean": mean(ref_enable),

        "input_beta_motion_mean": mean(beta_m),
        "input_beta_stability_mean": mean(beta_s),
        "input_beta_energy_mean": mean(beta_e),

        "ram_slip_proxy_mean": mean(ram_slip),
        "ram_roughness_proxy_mean": mean(ram_rough),
        "ram_sigma_mean": mean(ram_sigma),

        "success": int(success),
        "goal": int(goal),
        "startup_failed": int(startup_failed),
        "out_lane": int(out_lane),
        "final_x": final_x,
        "final_y": final_y,
        "rollout_max_abs_y": rollout_max_abs_y,
        "rollout_mean_abs_y": rollout_mean_abs_y,
        "hold_drift": hold_drift,

        "motion_score": motion_score,
        "stability_score": stability_score,
        "energy_score": energy_score,
        "effort_proxy": effort_proxy,

        "motion_need": motion_need,
        "stability_need": stability_need,
        "energy_need": energy_need,

        "target_beta_motion": target_beta_m,
        "target_beta_stability": target_beta_s,
        "target_beta_energy": target_beta_e,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", nargs="+", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--min-context-rows", type=int, default=20)
    args = ap.parse_args()

    samples = []
    skipped = []

    for sp in args.summary_json:
        summary_path = Path(sp)
        if not summary_path.exists():
            skipped.append((sp, "summary file missing"))
            continue

        summary = load_json(summary_path)
        manifest_value = summary.get("manifest") or summary.get("manifest_path")
        if not manifest_value:
            skipped.append((sp, "no manifest field"))
            continue

        rollouts = rollout_by_trial(summary)
        if not rollouts:
            skipped.append((sp, "no rollout rows in summary json"))
            continue

        try:
            manifest_path = resolve_manifest_path(summary_path, manifest_value)
            manifest_rows = read_manifest(manifest_path)
        except Exception as e:
            skipped.append((sp, f"manifest read failed: {e}"))
            continue

        for mr in manifest_rows:
            try:
                trial = int(float(mr.get("trial", 0)))
            except Exception:
                skipped.append((sp, f"bad trial in manifest row: {mr.get('trial')}"))
                continue

            rr = rollouts.get(trial)
            if rr is None:
                skipped.append((sp, f"no rollout row for trial {trial}"))
                continue

            d5_dir = mr.get("d5_log_dir", "")
            try:
                policy_csv = find_policy_input_csv(d5_dir)
                policy_rows = read_policy_rows(policy_csv)
            except Exception as e:
                skipped.append((sp, f"trial {trial}: policy input read failed: {e}"))
                continue

            by_ctx = defaultdict(list)
            for r in policy_rows:
                ctx = (r.get("context") or "unknown").strip() or "unknown"
                by_ctx[ctx].append(r)

            for ctx, rows in sorted(by_ctx.items()):
                if len(rows) < args.min_context_rows:
                    continue
                sample = compute_context_sample(
                    summary_path=summary_path,
                    summary=summary,
                    manifest_row=mr,
                    rollout_row=rr,
                    context=ctx,
                    rows=rows,
                )
                samples.append(sample)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not samples:
        raise SystemExit("[ERROR] no samples generated")

    fieldnames = list(samples[0].keys())
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow(s)

    ctx_counts = Counter(s["context"] for s in samples)
    source_counts = Counter(Path(s["source_summary"]).name for s in samples)

    target_means = {
        "target_beta_motion_mean": mean(s["target_beta_motion"] for s in samples),
        "target_beta_stability_mean": mean(s["target_beta_stability"] for s in samples),
        "target_beta_energy_mean": mean(s["target_beta_energy"] for s in samples),
        "motion_score_mean": mean(s["motion_score"] for s in samples),
        "stability_score_mean": mean(s["stability_score"] for s in samples),
        "energy_score_mean": mean(s["energy_score"] for s in samples),
    }

    lines = []
    lines.append("# TRACER Phase-D7.0a Objective Selector Bootstrap Dataset Summary v0")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This dataset bootstraps a data-derived Objective Selector from Phase-D6 rollout results.")
    lines.append("")
    lines.append("It is not yet IRL. It converts rollout statistics into interpretable proxy objective scores and heuristic beta targets.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- output_csv: `{out_csv}`")
    lines.append(f"- samples: `{len(samples)}`")
    lines.append(f"- min_context_rows: `{args.min_context_rows}`")
    lines.append("")
    lines.append("## Context counts")
    lines.append("")
    for k, v in sorted(ctx_counts.items()):
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Target beta / score means")
    lines.append("")
    for k, v in target_means.items():
        lines.append(f"- {k}: `{v:.6f}`")
    lines.append("")
    lines.append("## Source summaries")
    lines.append("")
    for k, v in sorted(source_counts.items()):
        lines.append(f"- `{k}`: `{v}` context samples")
    lines.append("")
    lines.append("## Skipped inputs")
    lines.append("")
    if skipped:
        for p, reason in skipped:
            lines.append(f"- `{p}`: {reason}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Input beta/RAM values are still proxy values from the Phase-D5/D6 pipeline.")
    lines.append("- Target beta values are bootstrap labels derived from motion, stability, and energy proxy scores.")
    lines.append("- Goal/hold behavior remains safety-protected by empirical fallback in the current runtime.")
    lines.append("- This dataset is meant to support D7.0b beta-target inspection and D7.0c Objective Selector training.")
    lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))

    print(f"[TRACER] wrote {out_csv}")
    print(f"[TRACER] wrote {out_md}")
    print(f"[TRACER] samples={len(samples)}")
    print("[TRACER] context_counts:")
    for k, v in sorted(ctx_counts.items()):
        print(f"  {k}: {v}")
    print("[TRACER] target/score means:")
    for k, v in target_means.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
