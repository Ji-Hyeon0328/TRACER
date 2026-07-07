#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean


INFO_T_RE = re.compile(r"\[INFO\]\s+\[(?P<t>[0-9.]+)\]")
XY_RE = re.compile(r"seq=(?P<seq>\d+)\s+x=(?P<x>[-+0-9.]+)\s+y=(?P<y>[-+0-9.]+)")
STOP_RE = re.compile(r"stopped=(?P<stopped>True|False)")
LABEL_RE = re.compile(r"label=(?P<label>[a-zA-Z0-9_]+)")
CTX_SWITCH_RE = re.compile(
    r"context switch:\s+x=(?P<x>[-+0-9.]+),\s+y=(?P<y>[-+0-9.]+),\s+segment=(?P<label>[a-zA-Z0-9_]+)"
)


def parse_control_log(path: Path):
    samples = []
    if not path.exists():
        return samples

    for line in path.read_text(errors="replace").splitlines():
        m_t = INFO_T_RE.search(line)
        m_xy = XY_RE.search(line)
        if not (m_t and m_xy):
            continue

        m_stop = STOP_RE.search(line)
        m_label = LABEL_RE.search(line)

        samples.append(
            {
                "t": float(m_t.group("t")),
                "seq": int(m_xy.group("seq")),
                "x": float(m_xy.group("x")),
                "y": float(m_xy.group("y")),
                "stopped": (m_stop.group("stopped") == "True") if m_stop else None,
                "label": m_label.group("label") if m_label else None,
                "raw": line,
            }
        )

    return samples


def parse_context_log(path: Path):
    switches = []
    if not path.exists():
        return switches

    for line in path.read_text(errors="replace").splitlines():
        m_t = INFO_T_RE.search(line)
        m_sw = CTX_SWITCH_RE.search(line)
        if not (m_t and m_sw):
            continue

        switches.append(
            {
                "t": float(m_t.group("t")),
                "x": float(m_sw.group("x")),
                "y": float(m_sw.group("y")),
                "label": m_sw.group("label"),
                "raw": line,
            }
        )

    return switches


def infer_run_type(log_dir: Path) -> tuple[str, Path]:
    fixed = log_dir / "fixed_baseline.log"
    oracle = log_dir / "oracle_schedule.log"
    beta = log_dir / "beta_schedule.log"

    if fixed.exists():
        return "fixed", fixed
    if oracle.exists():
        return "oracle", oracle
    if beta.exists():
        return "beta", beta
    return "unknown", fixed


def summarize_one(log_dir: Path, goal_x: float):
    run_type, control_log = infer_run_type(log_dir)
    context_log = log_dir / "context_provider.log"

    samples = parse_control_log(control_log)
    switches = parse_context_log(context_log)

    summary = {
        "log_dir": str(log_dir),
        "run_type": run_type,
        "control_log": str(control_log),
        "context_log": str(context_log),
        "num_samples": len(samples),
        "num_context_switches": len(switches),
        "goal_x": goal_x,
    }

    if not samples:
        summary.update(
            {
                "success": False,
                "reason": "no control samples parsed",
            }
        )
        return summary

    t0 = samples[0]["t"]
    tf = samples[-1]["t"]
    xs = [s["x"] for s in samples]
    ys = [s["y"] for s in samples]

    goal_samples = [s for s in samples if s["x"] >= goal_x]
    stopped_samples = [s for s in samples if s["stopped"] is True]

    t_goal = goal_samples[0]["t"] if goal_samples else None
    t_stop = stopped_samples[0]["t"] if stopped_samples else None

    segment_times = {}
    for threshold, name in [(2.0, "upslope_start"), (4.0, "rough_start"), (6.0, "downslope_start"), (8.0, "goal_flat_start")]:
        hit = next((s for s in samples if s["x"] >= threshold), None)
        if hit is not None:
            segment_times[name] = {
                "t_rel": hit["t"] - t0,
                "x": hit["x"],
                "y": hit["y"],
                "seq": hit["seq"],
            }

    context_switches = []
    for sw in switches:
        context_switches.append(
            {
                "label": sw["label"],
                "t_rel": sw["t"] - t0,
                "x": sw["x"],
                "y": sw["y"],
            }
        )

    elapsed = max(1e-9, tf - t0)
    summary.update(
        {
            "success": bool(goal_samples),
            "stopped": bool(stopped_samples),
            "t_start": t0,
            "t_end": tf,
            "duration_logged_sec": elapsed,
            "time_to_goal_sec": (t_goal - t0) if t_goal is not None else None,
            "time_to_stop_sec": (t_stop - t0) if t_stop is not None else None,
            "final_x": xs[-1],
            "final_y": ys[-1],
            "max_x": max(xs),
            "min_x": min(xs),
            "max_abs_y": max(abs(y) for y in ys),
            "mean_abs_y": mean(abs(y) for y in ys),
            "progress_rate_logged": (max(xs) - xs[0]) / elapsed,
            "segment_times": segment_times,
            "context_switches": context_switches,
        }
    )

    return summary


def write_markdown(summaries, out_md: Path):
    lines = []
    lines.append("# TRACER Phase-C Mixed Course Log Evaluation")
    lines.append("")
    lines.append("| run | success | stopped | time_to_goal_s | max_abs_y | final_x | final_y | log_dir |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")

    for s in summaries:
        lines.append(
            "| {run_type} | {success} | {stopped} | {ttg} | {maxy:.3f} | {fx:.3f} | {fy:.3f} | `{log_dir}` |".format(
                run_type=s.get("run_type", "unknown"),
                success=str(s.get("success")),
                stopped=str(s.get("stopped")),
                ttg="{:.2f}".format(s["time_to_goal_sec"]) if s.get("time_to_goal_sec") is not None else "NA",
                maxy=float(s.get("max_abs_y", 0.0)),
                fx=float(s.get("final_x", 0.0)),
                fy=float(s.get("final_y", 0.0)),
                log_dir=s.get("log_dir", ""),
            )
        )

    lines.append("")
    for s in summaries:
        lines.append(f"## {s.get('run_type', 'unknown')} — `{s.get('log_dir', '')}`")
        lines.append("")
        lines.append("Segment threshold times:")
        lines.append("")
        lines.append("| segment | t_rel_s | x | y | seq |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, item in s.get("segment_times", {}).items():
            lines.append(
                f"| {name} | {item['t_rel']:.2f} | {item['x']:.3f} | {item['y']:.3f} | {item['seq']} |"
            )

        if s.get("context_switches"):
            lines.append("")
            lines.append("Context switches:")
            lines.append("")
            lines.append("| label | t_rel_s | x | y |")
            lines.append("|---|---:|---:|---:|")
            for sw in s["context_switches"]:
                lines.append(f"| {sw['label']} | {sw['t_rel']:.2f} | {sw['x']:.3f} | {sw['y']:.3f} |")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dirs", nargs="+", required=True)
    ap.add_argument("--goal-x", type=float, default=8.4)
    ap.add_argument("--out-json", default="reports/phase_c_mixed_course_eval_v0.json")
    ap.add_argument("--out-md", default="reports/phase_c_mixed_course_eval_v0.md")
    args = ap.parse_args()

    summaries = [summarize_one(Path(d), goal_x=args.goal_x) for d in args.log_dirs]

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    write_markdown(summaries, out_md)

    print(f"[TRACER] wrote {out_json}")
    print(f"[TRACER] wrote {out_md}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
