#!/usr/bin/env python3

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-json", default="configs/phase_b_theta_lite_rl_v0/current_bandit_policy.json")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    p = json.load(open(args.policy_json))

    print(json.dumps({
        "schema": "phase_b_theta_lite_bandit_policy_report_v0",
        "policy_json": args.policy_json,
    }, indent=2, sort_keys=True))

    for world, ws in sorted(p.get("worlds", {}).items()):
        n_total = int(ws.get("n_total", 0))
        if n_total == 0:
            continue

        visited = []
        unvisited = []

        for action_name, st in ws.get("actions", {}).items():
            n = int(st.get("n", 0))
            item = {
                "action": action_name,
                "n": n,
                "mean_reward": float(st.get("mean_reward", 0.0)),
                "best_reward": st.get("best_reward"),
                "last_reward": st.get("last_reward"),
                "semantic_counts": st.get("semantic_counts", {}),
            }
            if n > 0:
                visited.append(item)
            else:
                unvisited.append(item)

        visited.sort(key=lambda x: x["mean_reward"], reverse=True)

        print()
        print("=" * 80)
        print(f"{world} | n_total={n_total} | visited={len(visited)} | unvisited={len(unvisited)}")
        print("- visited ranking")
        for x in visited[:args.top_k]:
            print(
                f"{x['action']:28s} "
                f"n={x['n']:2d} "
                f"mean={x['mean_reward']:8.3f} "
                f"best={x['best_reward']} "
                f"last={x['last_reward']} "
                f"sem={x['semantic_counts']}"
            )

        if unvisited:
            print("- unvisited")
            for x in unvisited:
                print(f"{x['action']:28s} n=0")


if __name__ == "__main__":
    main()
