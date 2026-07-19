#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def fnum(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def pred(x, w, b):
    return sum(a * wi for a, wi in zip(x, w)) + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-json", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary", required=True)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    model_path = Path(args.model_json)
    out_csv = Path(args.out_csv)
    out_summary = Path(args.out_summary)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    model = json.loads(model_path.read_text())
    features = model["features"]
    targets = model["targets"]
    rows = list(csv.DictReader(dataset.open()))

    out_fields = [
        "idx", "trial", "context", "x", "y",
    ]
    for t in targets:
        out_fields += [t, f"pred_{t}", f"err_{t}"]

    errors = {t: [] for t in targets}

    with out_csv.open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=out_fields)
        wcsv.writeheader()

        for i, r in enumerate(rows):
            xvec = [fnum(r.get(c)) for c in features]
            out = {
                "idx": i,
                "trial": r.get("trial"),
                "context": r.get("context"),
                "x": r.get("x"),
                "y": r.get("y"),
            }

            for t in targets:
                y = fnum(r.get(t))
                m = model["models"][t]
                yp = pred(xvec, m["weights"], m["bias"])
                e = yp - y
                out[t] = y
                out[f"pred_{t}"] = yp
                out[f"err_{t}"] = e
                errors[t].append(e)

            wcsv.writerow(out)

    summary = {}
    for t, es in errors.items():
        mae = sum(abs(e) for e in es) / len(es)
        rmse = math.sqrt(sum(e * e for e in es) / len(es))
        summary[t] = {"mae": mae, "rmse": rmse}

    lines = []
    lines.append("# TRACER Phase-D5 Learned Selector Prediction Eval v0")
    lines.append("")
    lines.append(f"- dataset: `{dataset}`")
    lines.append(f"- model_json: `{model_path}`")
    lines.append(f"- out_csv: `{out_csv}`")
    lines.append("")
    lines.append("| target | rmse | mae |")
    lines.append("|---|---:|---:|")
    for t, m in summary.items():
        lines.append(f"| {t} | {m['rmse']:.8f} | {m['mae']:.8f} |")

    out_summary.write_text("\n".join(lines) + "\n")

    print(f"[TRACER] wrote {out_csv}")
    print(f"[TRACER] wrote {out_summary}")


if __name__ == "__main__":
    main()
