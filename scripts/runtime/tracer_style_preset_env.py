#!/usr/bin/env python3
import argparse
import shlex
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--style", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    styles = cfg.get("styles", [])

    target = None
    for s in styles:
        if s.get("name") == args.style:
            target = s
            break

    if target is None:
        names = [s.get("name") for s in styles]
        raise SystemExit(f"[ERROR] style not found: {args.style}. Available: {names}")

    cmd = target["command"]
    beta = target.get("beta_hint", [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    beta_csv = ",".join(str(float(x)) for x in beta)

    exports = {
        "TRACER_STYLE_NAME": target["name"],
        "TRACER_STYLE_VX": cmd.get("vx", 0.0),
        "TRACER_STYLE_YAW_RATE": cmd.get("yaw_rate", 0.0),
        "TRACER_STYLE_BODY_HEIGHT": cmd.get("body_height", 0.30),
        "TRACER_STYLE_CLEARANCE": cmd.get("swing_clearance", cmd.get("clearance", 0.035)),
        "TRACER_STYLE_ENABLE": cmd.get("enable", 1.0),
        "TRACER_STYLE_BETA_HINT": beta_csv,
    }

    for k, v in exports.items():
        print(f"export {k}={shlex.quote(str(v))}")


if __name__ == "__main__":
    main()
