#!/usr/bin/env python3
import argparse
import json
import traceback
from pathlib import Path


CANDIDATES = [
    # RAM high-priority
    "artifacts/tracer_ram_scalar_v3/model.pt",
    "configs/learned_models/tracer_ram_scalar_v3_model.pt",
    "artifacts/tracer_ram_v2/model.pt",
    "artifacts/tracer_ram_v2c/model.pt",
    "artifacts/phase_a_ram_empirical_v1/phase_a_ram_empirical_v1.pt",
    "artifacts/phase_a_ram_teacher_smoke_v0/phase_a_ram_teacher_smoke_v0.pt",

    # Phase-B RAM proxy/runtime
    "configs/phase_b_objective_ram_bootstrap_v0/current_model.json",
    "configs/phase_b_objective_ram_uncertainty_v0/current_registry.json",

    # Objective selector high-priority
    "artifacts/tracer_objective_irl_v1/model.pt",
    "artifacts/tracer_objective_irl_v0/model.pt",
    "artifacts/phase_a_objective_selector_pref_with_risk_v0/phase_a_objective_selector_pref_v0.pt",
    "artifacts/phase_a_objective_selector_pref_v0/phase_a_objective_selector_pref_v0.pt",
    "artifacts/objective_selector_v2/objective_selector_v2.pt",
    "artifacts/objective_selector_v1/objective_selector_v1.pt",
    "artifacts/objective_preference_v2/objective_preference_v2.pt",
    "artifacts/tracer_highlevel_selector_v0/model.pt",

    # Runtime exported JSONs
    "configs/learned_models/tracer_preference_objective_irl_v1_model.json",
    "configs/learned_models/tracer_preference_objective_irl_v0_model.json",
    "configs/highlevel_policy/tracer_objective_conditioned_selector_v0.json",

    # Companion metadata / reports
    "artifacts/tracer_ram_v2/normalization.json",
    "artifacts/tracer_ram_v0/scaler_metrics_env.json",
    "artifacts/tracer_highlevel_selector_v0/scaler_and_metrics.json",
    "artifacts/tracer_objective_irl_v1/scaler_metrics_env.json",
    "artifacts/tracer_objective_irl_v1/objective_beta_by_terrain_v1.csv",
    "artifacts/phase_a_objective_selector_pref_with_risk_v0/phase_a_objective_selector_pref_v0_meta.json",
    "reports/tracer_preference_objective_irl_v1_summary.json",
    "reports/learned_stack_v3_beta_blend_report.md",
    "reports/learned_stack_v3_beta_blend_robust_eval_report.md",
]


def classify_path(path):
    s = str(path).lower()
    if "ram" in s or "rho" in s or "sigma" in s or "uncertainty" in s:
        return "ram"
    if "objective" in s or "selector" in s or "preference" in s or "irl" in s or "beta" in s:
        return "objective_selector"
    if "highlevel" in s:
        return "highlevel_selector"
    return "unknown"


def summarize_tensor(x):
    try:
        return {
            "type": "tensor",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "requires_grad": bool(getattr(x, "requires_grad", False)),
        }
    except Exception:
        return {"type": "tensor_like"}


def summarize_obj(obj, depth=0, max_depth=3, max_items=80):
    if depth > max_depth:
        return {"type": type(obj).__name__, "truncated": True}

    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return summarize_tensor(obj)
    except Exception:
        pass

    if isinstance(obj, dict):
        out = {
            "type": "dict",
            "num_keys": len(obj),
            "keys": list(obj.keys())[:max_items],
            "items": {}
        }
        for k in list(obj.keys())[:max_items]:
            try:
                out["items"][str(k)] = summarize_obj(obj[k], depth + 1, max_depth, max_items)
            except Exception as e:
                out["items"][str(k)] = {"error": str(e)}
        return out

    if isinstance(obj, (list, tuple)):
        return {
            "type": type(obj).__name__,
            "len": len(obj),
            "items": [summarize_obj(x, depth + 1, max_depth, max_items) for x in obj[:10]]
        }

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return {
            "type": type(obj).__name__,
            "value": obj
        }

    # nn.Module or unknown object
    out = {
        "type": type(obj).__name__,
        "module": getattr(type(obj), "__module__", ""),
    }

    if hasattr(obj, "state_dict"):
        try:
            sd = obj.state_dict()
            out["state_dict"] = summarize_obj(sd, depth + 1, max_depth, max_items)
        except Exception as e:
            out["state_dict_error"] = str(e)

    if hasattr(obj, "__dict__"):
        try:
            keys = list(obj.__dict__.keys())
            out["dict_keys"] = keys[:max_items]
        except Exception:
            pass

    return out


def inspect_pt(path):
    try:
        import torch
        # Local checkpoints from this repo. weights_only may not exist on old torch, so try fallback.
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")

        summary = summarize_obj(obj)
        return {
            "ok": True,
            "loader": "torch.load",
            "summary": summary,
        }
    except Exception as e:
        return {
            "ok": False,
            "loader": "torch.load",
            "error": str(e),
            "traceback": traceback.format_exc()[-4000:]
        }


def inspect_json(path):
    try:
        obj = json.load(open(path))
        return {
            "ok": True,
            "loader": "json",
            "summary": summarize_obj(obj),
        }
    except Exception as e:
        return {
            "ok": False,
            "loader": "json",
            "error": str(e),
        }


def inspect_text(path, max_chars=3000):
    try:
        txt = path.read_text(errors="ignore")
        return {
            "ok": True,
            "loader": "text",
            "num_chars": len(txt),
            "head": txt[:max_chars],
        }
    except Exception as e:
        return {
            "ok": False,
            "loader": "text",
            "error": str(e),
        }


def infer_io_hints(row):
    text = json.dumps(row.get("inspection", {}), default=str).lower()
    hints = {
        "input_dim": None,
        "output_dim": None,
        "feature_names_possible": False,
        "normalization_possible": False,
        "outputs_possible": [],
        "phase_b_connectability": "unknown"
    }

    if "input_dim" in text or "obs_dim" in text or "feature_dim" in text:
        hints["feature_names_possible"] = True
    if "normalization" in text or "scaler" in text or "mean" in text or "std" in text:
        hints["normalization_possible"] = True
    for key in ["beta", "risk", "uncertainty", "rho", "sigma", "reward", "preference", "score", "action"]:
        if key in text:
            hints["outputs_possible"].append(key)

    path = row["path"]
    if "tracer_ram_scalar_v3" in path:
        hints["phase_b_connectability"] = "high_priority_ram_adapter_candidate"
    elif "tracer_objective_irl_v1" in path:
        hints["phase_b_connectability"] = "high_priority_objective_adapter_candidate"
    elif "phase_a_objective_selector_pref_with_risk" in path:
        hints["phase_b_connectability"] = "risk_aware_objective_candidate"
    elif "phase_b_objective_ram" in path:
        hints["phase_b_connectability"] = "phase_b_proxy_immediately_readable"
    elif path.endswith(".json"):
        hints["phase_b_connectability"] = "runtime_json_candidate"
    else:
        hints["phase_b_connectability"] = "needs_manual_schema_match"

    return hints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = []

    for raw in CANDIDATES:
        p = Path(raw)
        row = {
            "path": raw,
            "exists": p.exists(),
            "kind": classify_path(p),
            "suffix": p.suffix,
            "size_bytes": p.stat().st_size if p.exists() else None,
            "inspection": None,
            "io_hints": None,
        }

        if p.exists():
            if p.suffix in [".pt", ".pth", ".ckpt"]:
                row["inspection"] = inspect_pt(p)
            elif p.suffix == ".json":
                row["inspection"] = inspect_json(p)
            else:
                row["inspection"] = inspect_text(p)

            row["io_hints"] = infer_io_hints(row)

        rows.append(row)

    report = {
        "schema": "phase_b_learned_module_schema_inspection_v0",
        "purpose": "Inspect candidate learned Objective Selector/RAM artifacts to decide how to connect them into the Phase-B PPO/Gazebo loop.",
        "rows": rows,
        "recommended_next_step": [
            "Pick one RAM checkpoint with clear input normalization.",
            "Pick one Objective Selector/IRL artifact with clear beta or score output.",
            "Build a Phase-B adapter that maps current episode/window features into those models.",
            "Evaluate learned adapter against raw PPO and controller-only baselines."
        ]
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True, default=str))

    lines = []
    lines.append("# Phase-B Learned Module Schema Inspection")
    lines.append("")
    lines.append("This report inspects candidate learned Objective Selector/RAM artifacts for Phase-B integration.")
    lines.append("")
    lines.append("| kind | exists | path | connectability | outputs_possible | normalization | ok |")
    lines.append("|---|---:|---|---|---|---:|---:|")

    for r in rows:
        h = r.get("io_hints") or {}
        insp = r.get("inspection") or {}
        lines.append(
            f"| {r['kind']} | {r['exists']} | `{r['path']}` | "
            f"{h.get('phase_b_connectability', '')} | "
            f"{','.join(h.get('outputs_possible', []))} | "
            f"{h.get('normalization_possible', '')} | "
            f"{insp.get('ok', '')} |"
        )

    lines.append("")
    lines.append("## Priority Interpretation")
    lines.append("")
    lines.append("- `tracer_ram_scalar_v3` should be checked first as a RAM adapter candidate.")
    lines.append("- `tracer_objective_irl_v1` and `phase_a_objective_selector_pref_with_risk_v0` should be checked first for Objective Selector / β / risk-aware scoring.")
    lines.append("- Phase-B JSON artifacts can be used immediately as proxy/registry inputs, but they are not the same as learned checkpoint inference.")
    lines.append("- The next patch should build a small adapter only after confirming input feature names and normalization.")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "num_rows": len(rows),
        "exists": sum(1 for r in rows if r["exists"]),
        "torch_ok": sum(1 for r in rows if (r.get("inspection") or {}).get("loader") == "torch.load" and (r.get("inspection") or {}).get("ok")),
        "json_ok": sum(1 for r in rows if (r.get("inspection") or {}).get("loader") == "json" and (r.get("inspection") or {}).get("ok")),
    }, indent=2, sort_keys=True))
    print()
    print("[wrote]", args.out_json)
    print("[wrote]", args.out_md)


if __name__ == "__main__":
    main()
