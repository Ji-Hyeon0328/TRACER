#!/usr/bin/env python3
import csv
import math
import re
from pathlib import Path

F18 = Path("datasets/phase_f/f18_robust_grouped_true_metric_beta_targets_v0.csv")
F17 = Path("datasets/phase_f/f17_rollout_outlier_variance_audit_v0.csv")
F14_REPORT = Path("reports/phase_f/f14_grouped_runtime_safe_beta_calibrator_summary_v0.md")
F20_REPORT = Path("reports/phase_f/f20_robust_runtime_safe_beta_calibrator_summary_v0.md")

OUT_CSV = Path("datasets/phase_f/f21_mean_vs_robust_calibrator_comparison_v0.csv")
OUT_MD = Path("reports/phase_f/f21_mean_vs_robust_calibrator_comparison_summary_v0.md")

def ff(x, default=""):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default

def fmt_float(v):
    return "NA" if v == "" else f"{float(v):.6f}"

def fmt_beta(b):
    return "(" + ", ".join(f"{float(x):.4f}" for x in b) + ")"

def pick_existing(row, candidates):
    for c in candidates:
        if c in row and row[c] != "":
            return row[c]
    return ""

def parse_tuple_like(x):
    if x is None:
        return None
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", str(x))
    if len(nums) >= 3:
        return [float(nums[0]), float(nums[1]), float(nums[2])]
    return None

def find_first_existing_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def parse_report_l1s(path):
    """
    Return (train_l1_by_tag, loto_l1_by_tag).

    This parser accepts both console-style lines:
      clean: target=(...) pred=(...) L1=0.123
      clean: pred=(...) target=(...) L1=0.456

    and markdown table rows containing tag + L1 columns.
    """
    train = {}
    loto = {}

    if not path.exists():
        return train, loto

    known_tags = {"clean", "m015", "m030", "m060", "p015", "p030", "p045", "p060"}
    mode = "train"
    last_header = None

    for line in path.read_text(errors="ignore").splitlines():
        raw = line.strip()
        low = raw.lower()

        if "leave-one" in low or "loto" in low or "heldout" in low:
            mode = "loto"
        if "train" in low and ("fit" in low or "training" in low):
            mode = "train"

        # Keep markdown header for table parsing.
        if raw.startswith("|") and "---" not in raw and any(k in low for k in ["tag", "heldout", "base_tag", "l1"]):
            last_header = [c.strip().lower().replace(" ", "_") for c in raw.strip("|").split("|")]
            continue

        # Console-style parser.
        m = re.search(r"^\s*([A-Za-z0-9_]+):.*?L1\s*=\s*([0-9.]+)", raw)
        if m:
            tag = m.group(1)
            if tag in known_tags:
                if mode == "loto":
                    loto[tag] = float(m.group(2))
                else:
                    train[tag] = float(m.group(2))
            continue

        # Markdown table parser.
        if raw.startswith("|") and "---" not in raw and last_header is not None:
            cells = [c.strip() for c in raw.strip("|").split("|")]
            if len(cells) != len(last_header):
                continue

            row = dict(zip(last_header, cells))

            tag = ""
            for k in ["base_tag", "heldout", "tag", "condition", "suite"]:
                if row.get(k, "") in known_tags:
                    tag = row[k]
                    break

            if not tag:
                for c in cells:
                    if c in known_tags:
                        tag = c
                        break

            if not tag:
                continue

            # Prefer explicit LOTO/leave-one columns, otherwise use any L1-like column.
            l1_candidates = []
            for k, v in row.items():
                lk = k.lower()
                if "loto" in lk and "l1" in lk:
                    l1_candidates.append(v)
                elif "leave" in lk and "l1" in lk:
                    l1_candidates.append(v)
                elif lk in {"l1", "loto_l1", "heldout_l1", "mean_l1"}:
                    l1_candidates.append(v)

            if not l1_candidates:
                # Fall back: scan numeric cells near the end.
                l1_candidates = cells[::-1]

            val = None
            for cand in l1_candidates:
                mm = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", cand)
                if mm:
                    try:
                        val = float(mm.group(0))
                        break
                    except Exception:
                        pass

            if val is None:
                continue

            if mode == "loto":
                loto[tag] = val
            else:
                train[tag] = val

    return train, loto

def load_f17_max_delta():
    out = {}

    if not F17.exists():
        return out

    for r in csv.DictReader(open(F17)):
        tag = pick_existing(r, ["base_tag", "group_base_tag", "tag"])
        if not tag:
            continue

        val = pick_existing(r, [
            "max_rollout_loo_beta_delta_l1",
            "f17_max_rollout_loo_beta_delta_l1",
            "rollout_loo_beta_delta_l1",
            "loo_beta_delta_l1",
        ])
        v = ff(val, "")
        if v == "":
            continue

        out[tag] = max(out.get(tag, 0.0), float(v))

    return out

def load_f18_rows():
    if not F18.exists():
        raise SystemExit(f"[TRACER] missing {F18}")

    raw = list(csv.DictReader(open(F18)))
    if not raw:
        raise SystemExit(f"[TRACER] empty {F18}")

    cols = raw[0].keys()

    tag_col = find_first_existing_col(cols, ["group_base_tag", "base_tag", "tag"])
    if tag_col is None:
        raise SystemExit(f"[TRACER] cannot find tag column in F18. columns={list(cols)}")

    mean_tuple_col = find_first_existing_col(cols, ["mean_beta", "group_mean_beta"])
    robust_tuple_col = find_first_existing_col(cols, ["robust_beta", "group_robust_beta"])

    mean_cols = {
        "motion": find_first_existing_col(cols, [
            "group_mean_beta_motion_true_seed",
            "mean_beta_motion_true_seed",
            "beta_motion_mean_true_metric",
            "target_beta_motion_mean_true_metric",
            "target_beta_motion_mean",
            "mean_beta_motion",
        ]),
        "stability": find_first_existing_col(cols, [
            "group_mean_beta_stability_true_seed",
            "mean_beta_stability_true_seed",
            "beta_stability_mean_true_metric",
            "target_beta_stability_mean_true_metric",
            "target_beta_stability_mean",
            "mean_beta_stability",
        ]),
        "energy": find_first_existing_col(cols, [
            "group_mean_beta_energy_true_seed",
            "mean_beta_energy_true_seed",
            "beta_energy_mean_true_metric",
            "target_beta_energy_mean_true_metric",
            "target_beta_energy_mean",
            "mean_beta_energy",
        ]),
    }

    robust_cols = {
        "motion": find_first_existing_col(cols, [
            "group_robust_median_beta_motion_true_seed",
            "robust_median_beta_motion_true_seed",
            "robust_beta_motion_true_metric",
            "target_beta_motion_robust_true_metric",
            "beta_motion_robust_true_metric",
            "robust_beta_motion",
        ]),
        "stability": find_first_existing_col(cols, [
            "group_robust_median_beta_stability_true_seed",
            "robust_median_beta_stability_true_seed",
            "robust_beta_stability_true_metric",
            "target_beta_stability_robust_true_metric",
            "beta_stability_robust_true_metric",
            "robust_beta_stability",
        ]),
        "energy": find_first_existing_col(cols, [
            "group_robust_median_beta_energy_true_seed",
            "robust_median_beta_energy_true_seed",
            "robust_beta_energy_true_metric",
            "target_beta_energy_robust_true_metric",
            "beta_energy_robust_true_metric",
            "robust_beta_energy",
        ]),
    }

    print("[TRACER] F18 detected columns:")
    print(f"  tag_col={tag_col}")
    print(f"  mean_tuple_col={mean_tuple_col}")
    print(f"  robust_tuple_col={robust_tuple_col}")
    print(f"  mean_cols={mean_cols}")
    print(f"  robust_cols={robust_cols}")

    rows = []
    seen = set()

    for r in raw:
        tag = r.get(tag_col, "")
        if not tag or tag in seen:
            continue
        seen.add(tag)

        mean_beta = None
        robust_beta = None

        if mean_tuple_col:
            mean_beta = parse_tuple_like(r.get(mean_tuple_col))
        if robust_tuple_col:
            robust_beta = parse_tuple_like(r.get(robust_tuple_col))

        if mean_beta is None:
            vals = [ff(r.get(mean_cols[k]), "") if mean_cols[k] else "" for k in ["motion", "stability", "energy"]]
            if all(v != "" for v in vals):
                mean_beta = [float(v) for v in vals]

        if robust_beta is None:
            vals = [ff(r.get(robust_cols[k]), "") if robust_cols[k] else "" for k in ["motion", "stability", "energy"]]
            if all(v != "" for v in vals):
                robust_beta = [float(v) for v in vals]

        if mean_beta is None or robust_beta is None:
            continue

        shift = sum(abs(a - b) for a, b in zip(mean_beta, robust_beta))

        rows.append({
            "base_tag": tag,
            "mean_beta": mean_beta,
            "robust_beta": robust_beta,
            "mean_vs_robust_beta_l1": shift,
        })

    if not rows:
        raise SystemExit(
            "[TRACER] F21 produced no rows from F18. "
            f"Check F18 columns: {list(cols)}"
        )

    return rows

f14_train, f14_loto = parse_report_l1s(F14_REPORT)
f20_train, f20_loto = parse_report_l1s(F20_REPORT)
f17_delta = load_f17_max_delta()

rows = []
for r in load_f18_rows():
    tag = r["base_tag"]

    train14 = f14_train.get(tag, "")
    train20 = f20_train.get(tag, "")
    loto14 = f14_loto.get(tag, "")
    loto20 = f20_loto.get(tag, "")

    rows.append({
        "base_tag": tag,
        "mean_beta": fmt_beta(r["mean_beta"]),
        "robust_beta": fmt_beta(r["robust_beta"]),
        "mean_vs_robust_beta_l1": r["mean_vs_robust_beta_l1"],
        "f17_max_rollout_loo_beta_delta_l1": f17_delta.get(tag, ""),
        "f14_train_l1": train14,
        "f20_train_l1": train20,
        "train_l1_improvement_f14_minus_f20": (train14 - train20) if train14 != "" and train20 != "" else "",
        "f14_loto_l1": loto14,
        "f20_loto_l1": loto20,
        "loto_l1_improvement_f14_minus_f20": (loto14 - loto20) if loto14 != "" and loto20 != "" else "",
    })

def sort_key(r):
    v = r["loto_l1_improvement_f14_minus_f20"]
    return float("-inf") if v == "" else v

rows.sort(key=sort_key, reverse=True)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "base_tag",
    "mean_beta",
    "robust_beta",
    "mean_vs_robust_beta_l1",
    "f17_max_rollout_loo_beta_delta_l1",
    "f14_train_l1",
    "f20_train_l1",
    "train_l1_improvement_f14_minus_f20",
    "f14_loto_l1",
    "f20_loto_l1",
    "loto_l1_improvement_f14_minus_f20",
]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

comparable = [r for r in rows if r["loto_l1_improvement_f14_minus_f20"] != ""]
new_conditions = [r for r in rows if r["loto_l1_improvement_f14_minus_f20"] == ""]
improved = [r for r in comparable if r["loto_l1_improvement_f14_minus_f20"] > 0]
worse = [r for r in comparable if r["loto_l1_improvement_f14_minus_f20"] < 0]

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("# TRACER Phase-F21 Mean vs Robust Calibrator Comparison v0\n\n")
    f.write("This compares the earlier mean-target grouped calibrator against the robust-target calibrator. Conditions absent from the older F14 baseline are marked as `NA` rather than treated as zero.\n\n")
    f.write(f"- output csv: `{OUT_CSV}`\n")
    f.write(f"- conditions: `{len(rows)}`\n")
    f.write(f"- comparable conditions: `{len(comparable)}`\n")
    f.write(f"- new conditions without F14 baseline: `{len(new_conditions)}`\n")
    f.write(f"- conditions improved by robust target: `{len(improved)}` / `{len(comparable)}`\n")
    f.write(f"- conditions worsened by robust target: `{len(worse)}` / `{len(comparable)}`\n\n")

    f.write("## Leave-one-tag-out comparison\n\n")
    f.write("| base_tag | mean beta | robust beta | target shift L1 | F14 LOTO L1 | F20 LOTO L1 | improvement | F17 max rollout delta |\n")
    f.write("|---|---|---|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(
            f"| {r['base_tag']} | {r['mean_beta']} | {r['robust_beta']} | "
            f"{float(r['mean_vs_robust_beta_l1']):.6f} | "
            f"{fmt_float(r['f14_loto_l1'])} | "
            f"{fmt_float(r['f20_loto_l1'])} | "
            f"{fmt_float(r['loto_l1_improvement_f14_minus_f20'])} | "
            f"{fmt_float(r['f17_max_rollout_loo_beta_delta_l1'])} |\n"
        )

    f.write("\n## Safe interpretation\n\n")
    f.write("Rows with `NA` in the F14 columns are new anchor conditions and should not be interpreted as degradation or improvement against the older baseline.\n")

print(f"[TRACER] wrote {OUT_CSV}")
print(f"[TRACER] wrote {OUT_MD}")
print("[TRACER] LOTO improvement F14 - F20:")
for r in rows:
    print(
        f"  {r['base_tag']}: "
        f"F14={fmt_float(r['f14_loto_l1'])} "
        f"F20={fmt_float(r['f20_loto_l1'])} "
        f"improvement={fmt_float(r['loto_l1_improvement_f14_minus_f20'])}"
    )
