#!/usr/bin/env python3
import argparse
import math
import random
from pathlib import Path

import yaml


def surface_xml(mu, mu2, slip, contact_profile):
    # Gazebo Classic / ODE approximation.
    if contact_profile == "soft_damped":
        kp = 30000.0
        kd = 250.0
        max_vel = 0.05
        min_depth = 0.003
    elif contact_profile == "low_friction":
        kp = 100000.0
        kd = 50.0
        max_vel = 0.10
        min_depth = 0.001
    else:
        kp = 1000000.0
        kd = 100.0
        max_vel = 0.01
        min_depth = 0.001

    return f"""
          <surface>
            <friction>
              <ode>
                <mu>{mu}</mu>
                <mu2>{mu2}</mu2>
                <slip1>{slip}</slip1>
                <slip2>{slip}</slip2>
              </ode>
            </friction>
            <contact>
              <ode>
                <kp>{kp}</kp>
                <kd>{kd}</kd>
                <max_vel>{max_vel}</max_vel>
                <min_depth>{min_depth}</min_depth>
              </ode>
            </contact>
          </surface>"""


def world_header():
    return """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="default">
    <include>
      <uri>model://sun</uri>
    </include>

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


def world_footer():
    return """
  </world>
</sdf>
"""


def ground_plane_model(name, mu, mu2, slip, contact_profile):
    surf = surface_xml(mu, mu2, slip, contact_profile)
    return f"""
    <model name="{name}">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>30 12</size>
            </plane>
          </geometry>
{surf}
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>30 12</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.45 0.45 0.45 1</ambient>
            <diffuse>0.55 0.55 0.55 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""


def slope_box_model(name, slope_deg, mu, mu2, slip, contact_profile):
    angle = math.radians(float(slope_deg))
    pitch = -angle  # uphill along +x
    thickness = 0.10
    center_z = -0.5 * thickness * math.cos(angle)
    surf = surface_xml(mu, mu2, slip, contact_profile)

    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>0 0 {center_z:.6f} 0 {pitch:.6f} 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>30 12 {thickness}</size>
            </box>
          </geometry>
{surf}
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>30 12 {thickness}</size>
            </box>
          </geometry>
          <material>
            <ambient>0.35 0.45 0.35 1</ambient>
            <diffuse>0.45 0.60 0.45 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""


def rough_world_models(mu, mu2, slip, contact_profile, amp, seed=7):
    random.seed(seed)
    parts = [ground_plane_model("tracer_base_ground", mu, mu2, slip, contact_profile)]

    # Small boxes approximating rough patches.
    # Keep the center corridor clear-ish but still irregular.
    nx = 18
    ny = 5
    dx = 0.45
    dy = 0.45

    for ix in range(nx):
        for iy in range(ny):
            x = 0.5 + ix * dx
            y = (iy - (ny - 1) / 2.0) * dy
            h = random.uniform(0.25 * amp, amp)
            sx = random.uniform(0.18, 0.32)
            sy = random.uniform(0.18, 0.32)
            yaw = random.uniform(-0.5, 0.5)
            surf = surface_xml(mu, mu2, slip, contact_profile)
            parts.append(f"""
    <model name="rough_patch_{ix}_{iy}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {h/2.0:.4f} 0 0 {yaw:.4f}</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>{sx:.3f} {sy:.3f} {h:.4f}</size>
            </box>
          </geometry>
{surf}
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>{sx:.3f} {sy:.3f} {h:.4f}</size>
            </box>
          </geometry>
          <material>
            <ambient>0.38 0.34 0.28 1</ambient>
            <diffuse>0.50 0.43 0.35 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
""")
    return "\n".join(parts)


def make_world(terrain):
    name = terrain["name"]
    category = terrain.get("category", "flat")
    params = terrain.get("params", {})

    mu = float(params.get("mu", 0.9))
    mu2 = float(params.get("mu2", mu))
    slip = float(params.get("slip", 0.0))
    slope_deg = float(params.get("slope_deg", 0.0))
    amp = float(params.get("roughness_amp_m", 0.0))
    contact_profile = params.get("contact_profile", "nominal")

    body = world_header()

    if category in ("slope", "slippery_slope"):
        body += slope_box_model("tracer_slope_ground", slope_deg, mu, mu2, slip, contact_profile)
    elif category == "rough":
        body += rough_world_models(mu, mu2, slip, contact_profile, amp)
    else:
        body += ground_plane_model("tracer_ground", mu, mu2, slip, contact_profile)

    body += world_footer()
    return body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/terrain_dataset/terrain_set_v0.yaml")
    parser.add_argument("--out-dir", default="generated/gazebo_worlds")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for terrain in cfg.get("terrains", []):
        world_name = terrain.get("world_name")
        if not world_name:
            continue

        # Keep built-in earth as-is.
        if world_name == "earth":
            print("[TRACER] skip built-in earth")
            continue

        text = make_world(terrain)
        out = out_dir / f"{world_name}.world"
        out.write_text(text)
        print("[TRACER] wrote", out)


if __name__ == "__main__":
    main()
