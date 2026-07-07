#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


PHYSICS = """\
    <physics name="default_physics" type="ode">
      <max_step_size>0.00025</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>50</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
"""

SURFACE = """\
          <surface>
            <friction>
              <ode>
                <mu>{mu:.3f}</mu>
                <mu2>{mu2:.3f}</mu2>
                <slip1>0.0</slip1>
                <slip2>0.0</slip2>
              </ode>
            </friction>
            <contact>
              <ode>
                <kp>1000000.0</kp>
                <kd>100.0</kd>
                <max_vel>0.01</max_vel>
                <min_depth>0.001</min_depth>
              </ode>
            </contact>
          </surface>
"""


def box_model(
    name: str,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    sx: float,
    sy: float,
    sz: float,
    ambient: str,
    diffuse: str,
    mu: float = 0.95,
    mu2: float = 0.95,
) -> str:
    surface = SURFACE.format(mu=mu, mu2=mu2)
    return f"""\
    <model name="{name}">
      <static>true</static>
      <pose>{x:.4f} {y:.4f} {z:.4f} {roll:.6f} {pitch:.6f} {yaw:.6f}</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>{sx:.4f} {sy:.4f} {sz:.4f}</size>
            </box>
          </geometry>
{surface}
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>{sx:.4f} {sy:.4f} {sz:.4f}</size>
            </box>
          </geometry>
          <material>
            <ambient>{ambient}</ambient>
            <diffuse>{diffuse}</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""


def flat_box_segment(name: str, x0: float, x1: float, top_z: float, width: float, thickness: float, ambient: str, diffuse: str) -> str:
    cx = 0.5 * (x0 + x1)
    length = x1 - x0
    cz = top_z - 0.5 * thickness
    return box_model(
        name=name,
        x=cx,
        y=0.0,
        z=cz,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        sx=length,
        sy=width,
        sz=thickness,
        ambient=ambient,
        diffuse=diffuse,
    )


def slope_box_segment(name: str, x0: float, x1: float, z0: float, z1: float, width: float, thickness: float, ambient: str, diffuse: str) -> str:
    dx = x1 - x0
    dz = z1 - z0
    slope = math.atan2(dz, dx)

    # Gazebo box top surface has dz/dx = -tan(pitch), so pitch = -slope.
    pitch = -slope

    cx = 0.5 * (x0 + x1)
    top_center_z = 0.5 * (z0 + z1)
    local_length = math.hypot(dx, dz)
    cz = top_center_z - 0.5 * thickness * math.cos(abs(pitch))

    return box_model(
        name=name,
        x=cx,
        y=0.0,
        z=cz,
        roll=0.0,
        pitch=pitch,
        yaw=0.0,
        sx=local_length,
        sy=width,
        sz=thickness,
        ambient=ambient,
        diffuse=diffuse,
    )


def rough_patch_models(x0: float, x1: float, base_z: float, width: float, seed: int) -> str:
    rng = random.Random(seed)
    chunks = []

    # Underlay flat support for rough segment.
    chunks.append(
        flat_box_segment(
            "mixed_rough_underlay",
            x0,
            x1,
            base_z,
            width,
            0.10,
            "0.36 0.34 0.32 1",
            "0.46 0.43 0.40 1",
        )
    )

    xs = [x0 + 0.25 + i * 0.35 for i in range(int((x1 - x0 - 0.3) / 0.35))]
    ys = [-0.75, -0.35, 0.0, 0.35, 0.75]

    k = 0
    for x in xs:
        for y in ys:
            sx = rng.uniform(0.18, 0.32)
            sy = rng.uniform(0.16, 0.30)
            sz = rng.uniform(0.012, 0.045)
            yaw = rng.uniform(-0.45, 0.45)
            z = base_z + 0.5 * sz
            chunks.append(
                box_model(
                    name=f"mixed_rough_patch_{k:03d}",
                    x=x,
                    y=y,
                    z=z,
                    roll=0.0,
                    pitch=0.0,
                    yaw=yaw,
                    sx=sx,
                    sy=sy,
                    sz=sz,
                    ambient="0.38 0.34 0.28 1",
                    diffuse="0.50 0.43 0.35 1",
                )
            )
            k += 1

    return "\n".join(chunks)


def build_world(seed: int = 7) -> str:
    width = 5.0
    thickness = 0.10
    theta = math.radians(5.0)
    h = math.tan(theta) * 2.0

    models = []

    # Extra start pad behind x=0 so the robot does not spawn exactly on an edge.
    models.append(
        flat_box_segment(
            "mixed_start_flat_pad",
            -2.0,
            2.0,
            0.0,
            width,
            thickness,
            "0.45 0.45 0.45 1",
            "0.55 0.55 0.55 1",
        )
    )

    # x=2~4: upslope 5 deg, z=0 -> h.
    models.append(
        slope_box_segment(
            "mixed_upslope_5deg",
            2.0,
            4.0,
            0.0,
            h,
            width,
            thickness,
            "0.35 0.45 0.35 1",
            "0.45 0.60 0.45 1",
        )
    )

    # x=4~6: rough solid at elevated plateau h.
    models.append(rough_patch_models(4.0, 6.0, h, width, seed))

    # x=6~8: downslope 5 deg, z=h -> 0.
    models.append(
        slope_box_segment(
            "mixed_downslope_5deg",
            6.0,
            8.0,
            h,
            0.0,
            width,
            thickness,
            "0.35 0.40 0.45 1",
            "0.45 0.52 0.60 1",
        )
    )

    # x=8~12: goal flat.
    models.append(
        flat_box_segment(
            "mixed_goal_flat_pad",
            8.0,
            12.0,
            0.0,
            width,
            thickness,
            "0.45 0.45 0.45 1",
            "0.55 0.55 0.55 1",
        )
    )

    body = "\n".join(models)

    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="default">
    <include>
      <uri>model://sun</uri>
    </include>

{PHYSICS}
{body}
  </world>
</sdf>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="generated/gazebo_worlds/tracer_mixed_solid_course_v0.world",
        help="Output .world path",
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_world(seed=args.seed), encoding="utf-8")
    print(f"[TRACER] wrote {out}")


if __name__ == "__main__":
    main()
