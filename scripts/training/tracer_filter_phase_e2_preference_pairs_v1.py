#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--keep-types", default="stability,balanced")
    args = ap.parse_args()

    keep_types = {x.strip() for x in args.keep_types.split(",") if x.strip()}

    rows = list(csv.DictReader(open(args.in_csv)))
    kept = []
    dropped_same_dims = 0
    dropped_type = 0

    for r in rows:
        if r["pref_type"] not in keep_types:
            dropped_type += 1
            continue

        if r["preferred_dims"] == r["rejected_dims"]:
            dropped_same_dims += 1
            continue

        kept.append(r)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if kept:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
            w.writeheader()
            for r in kept:
                w.writerow(r)
    else:
        out_csv.write_text("")

    c_type = Counter(r["pref_type"] for r in kept)
    c_reset = Counter(r["reset_y"] for r in kept)
    c_dims = Counter((r["preferred_dims"], r["rejected_dims"]) for r in kept)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with open(out_md, "w") as f:
        f.write("# TRACER Phase-E2 Preference Pairs v1 Filtered\n\n")
        f.write("This filters E2 v0 into cross-policy preference pairs for bootstrap policy/objective selection.\n\n")
        f.write(f"- in_csv: `{args.in_csv}`\n")
        f.write(f"- out_csv: `{out_csv}`\n")
        f.write(f"- kept_pairs: `{len(kept)}`\n")
        f.write(f"- dropped_same_dims: `{dropped_same_dims}`\n")
        f.write(f"- dropped_type: `{dropped_type}`\n")
        f.write(f"- keep_types: `{','.join(sorted(keep_types))}`\n\n")

        f.write("## Counts by preference type\n\n")
        for k, v in sorted(c_type.items()):
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Counts by reset_y\n\n")
        for k, v in sorted(c_reset.items()):
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Dimension preference patterns\n\n")
        for (pd, rd), v in c_dims.most_common(20):
            f.write(f"- `{pd}` > `{rd}` : `{v}`\n")

        f.write("\n## Top filtered pairs\n\n")
        f.write("| pref_type | reset_y | preferred | rejected | diff | pref max_y | rej max_y | pref accept | rej accept |\n")
        f.write("|---|---:|---|---|---:|---:|---:|---:|---:|\n")
        for r in sorted(kept, key=lambda x: float(x["score_diff"]), reverse=True)[:20]:
            f.write(
                f"| {r['pref_type']} | {float(r['reset_y']):+.2f} | "
                f"{r['preferred_dims']} | {r['rejected_dims']} | "
                f"{float(r['score_diff']):.3f} | "
                f"{float(r['preferred_path_max_abs_y']):.3f} | "
                f"{float(r['rejected_path_max_abs_y']):.3f} | "
                f"{float(r['preferred_moving_accept_rate']):.3f} | "
                f"{float(r['rejected_moving_accept_rate']):.3f} |\n"
            )

    print(f"[TRACER] wrote {out_csv}")
    print(f"[TRACER] wrote {out_md}")
    print(f"[TRACER] kept_pairs={len(kept)}")
    print(f"[TRACER] dropped_same_dims={dropped_same_dims}")
    print(f"[TRACER] dropped_type={dropped_type}")
    print("[TRACER] counts_by_type:", dict(c_type))
    print("[TRACER] counts_by_reset_y:", dict(c_reset))

if __name__ == "__main__":
    main()
