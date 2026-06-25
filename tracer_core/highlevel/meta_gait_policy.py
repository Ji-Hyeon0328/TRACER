from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tracer_core.highlevel.decoder_mapper import beta_adjusted_defaults
from tracer_core.highlevel.meta_gait import HighLevelPolicyInput, MetaGaitCommand


class MetaGaitPolicy(Protocol):
    """Interface for high-level policy pi_high.

    pi_high(c_t, rho, sigma, beta, z_mode, robot_state, goal) -> theta_t
    """

    def predict(self, inp: HighLevelPolicyInput) -> MetaGaitCommand:
        ...


@dataclass(frozen=True)
class RuleBasedMetaGaitPolicy:
    """Interpretable policy stub for the current TRACER runtime.

    This is not the final RL policy. It maps HighLevelPolicyInput to a plausible
    MetaGaitCommand so the full high-level architecture can run end-to-end before
    learned policy training is introduced.
    """

    default_yaw_rate: float = 0.0

    def predict(self, inp: HighLevelPolicyInput) -> MetaGaitCommand:
        mode = inp.gait_mode
        beta = inp.beta.normalized()
        defaults = beta_adjusted_defaults(mode, beta)

        # Current command-level defaults by gait mode.
        # These are conservative because the current low-level interface is limited
        # to [vx, yaw_rate, body_height, swing_clearance, enable].
        if mode == "fast":
            vx = 0.28
            body_height = 0.295
            clearance = 0.030
            enable = 1.0
        elif mode == "normal":
            vx = 0.21
            body_height = 0.300
            clearance = 0.035
            enable = 1.0
        elif mode == "conservative":
            # Keep conservative locomotion slow. In sponge downslope sanity,
            # vx=0.050 reduced lateral drift but significantly reduced
            # terrain-relative height margin and increased roll/pitch.
            vx = 0.035
            body_height = 0.350
            clearance = 0.110
            enable = 1.0
        elif mode == "recovery":
            vx = 0.0
            body_height = 0.330
            clearance = 0.080
            enable = 0.0
        elif mode in {"avoid", "no_valid"}:
            vx = 0.0
            body_height = 0.305
            clearance = 0.055
            enable = 0.0
        else:
            vx = 0.05
            body_height = 0.335
            clearance = 0.080
            enable = 1.0
            mode = "conservative"

        # Online RAM can further reduce motion if the gate looks risky.
        risk = max(inp.ram.control_risk, inp.ram.fallen_prob, inp.ram.recovery_prob, inp.ram.sigma)
        if enable > 0.0 and risk >= 0.85:
            vx *= 0.50
            defaults["risk_scale"] = max(defaults["risk_scale"], 1.75)
        elif enable > 0.0 and risk >= 0.50:
            vx *= 0.70
            defaults["risk_scale"] = max(defaults["risk_scale"], 1.45)

        return MetaGaitCommand(
            vx=vx,
            yaw_rate=self.default_yaw_rate,
            body_height=body_height,
            swing_clearance=clearance,
            enable=enable,
            gait_period=defaults["gait_period"],
            duty_factor=defaults["duty_factor"],
            step_length=defaults["step_length"],
            stance_width=defaults["stance_width"],
            impedance_scale=defaults["impedance_scale"],
            residual_gain_scale=defaults["residual_gain_scale"],
            risk_scale=defaults["risk_scale"],
            source_mode=mode,
            source_reason="rule_based_meta_gait_policy",
            extras={
                "policy_type": "rule_based",
                "ram_level": inp.ram.ram_level,
                "sigma": inp.ram.sigma,
                "beta_motion": beta.motion,
                "beta_stability": beta.stability,
                "beta_energy": beta.energy,
            },
        )


class _MetaGaitMLPForRuntime:
    """Lazy torch MLP wrapper used only when learned policy is requested."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        import torch
        from torch import nn

        self.torch = torch
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def load_state_dict(self, state_dict):
        # Training script saves MetaGaitMLP.state_dict(), where the Sequential
        # module is stored under "net.*". Runtime uses the Sequential module
        # directly, so support both key formats.
        if any(str(k).startswith("net.") for k in state_dict.keys()):
            state_dict = {
                str(k).removeprefix("net."): v
                for k, v in state_dict.items()
            }
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def __call__(self, x):
        return self.model(x)


def _policy_input_feature_map(inp: HighLevelPolicyInput) -> dict[str, float]:
    mode = inp.gait_mode
    ram_level = inp.ram.ram_level

    def mode_is(name: str) -> float:
        return 1.0 if mode == name else 0.0

    def ram_is(name: str) -> float:
        return 1.0 if ram_level == name else 0.0

    rho_norm_feature = float(inp.ram.rho[0]) if inp.ram.rho else 0.0

    return {
        "beta_motion": float(inp.beta.motion),
        "beta_stability": float(inp.beta.stability),
        "beta_energy": float(inp.beta.energy),
        "sigma": float(inp.ram.sigma),
        "control_risk": float(inp.ram.control_risk),
        "fallen_prob": float(inp.ram.fallen_prob),
        "recovery_prob": float(inp.ram.recovery_prob),
        "rho_norm_feature": rho_norm_feature,
        "ram_level_unknown": ram_is("unknown"),
        "ram_level_stable": ram_is("stable"),
        "ram_level_caution": ram_is("caution"),
        "ram_level_unstable": ram_is("unstable"),
        "mode_fast": mode_is("fast"),
        "mode_normal": mode_is("normal"),
        "mode_conservative": mode_is("conservative"),
        "mode_recovery": mode_is("recovery"),
        "mode_avoid": mode_is("avoid"),
        "mode_no_valid": mode_is("no_valid"),
    }


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


@dataclass
class TorchMetaGaitPolicy:
    """Torch runtime loader for learned meta-gait policy.

    Torch is imported lazily so the default rule_based ROS2 runtime remains usable
    even when /usr/bin/python3 does not have torch installed.
    """

    model_path: Path

    def __post_init__(self):
        import torch

        self.torch = torch
        bundle = torch.load(self.model_path, map_location="cpu")

        self.feature_names = list(bundle["feature_names"])
        self.target_names = list(bundle["target_names"])
        self.x_mean = torch.tensor(bundle["x_mean"], dtype=torch.float32)
        self.x_std = torch.tensor(bundle["x_std"], dtype=torch.float32).clamp_min(1e-6)
        self.y_mean = torch.tensor(bundle["y_mean"], dtype=torch.float32)
        self.y_std = torch.tensor(bundle["y_std"], dtype=torch.float32).clamp_min(1e-6)

        self.net = _MetaGaitMLPForRuntime(
            input_dim=int(bundle["input_dim"]),
            output_dim=int(bundle["output_dim"]),
            hidden_dim=int(bundle["hidden_dim"]),
        )
        self.net.load_state_dict(bundle["model_state_dict"])

    def _feature_tensor(self, inp: HighLevelPolicyInput):
        fmap = _policy_input_feature_map(inp)
        values = [float(fmap.get(name, 0.0)) for name in self.feature_names]
        x = self.torch.tensor(values, dtype=self.torch.float32)
        return (x - self.x_mean) / self.x_std

    def _prediction_dict(self, inp: HighLevelPolicyInput) -> dict[str, float]:
        with self.torch.no_grad():
            x = self._feature_tensor(inp).unsqueeze(0)
            y_norm = self.net(x).squeeze(0)
            y = y_norm * self.y_std + self.y_mean

        return {
            name: float(y[i].item())
            for i, name in enumerate(self.target_names)
        }

    def predict(self, inp: HighLevelPolicyInput) -> MetaGaitCommand:
        pred = self._prediction_dict(inp)

        enable = _clamp(pred.get("enable", 0.0), 0.0, 1.0)
        vx = float(pred.get("vx", 0.0))
        if enable < 0.5:
            vx = 0.0
            enable = 0.0

        return MetaGaitCommand(
            vx=_clamp(vx, -0.20, 0.40),
            yaw_rate=_clamp(pred.get("yaw_rate", 0.0), -0.80, 0.80),
            body_height=_clamp(pred.get("body_height", 0.30), 0.20, 0.45),
            swing_clearance=_clamp(pred.get("swing_clearance", 0.05), 0.0, 0.20),
            enable=enable,
            gait_period=_clamp(pred.get("gait_period", 0.45), 0.15, 1.50),
            duty_factor=_clamp(pred.get("duty_factor", 0.60), 0.35, 0.90),
            step_length=_clamp(pred.get("step_length", 0.05), 0.0, 0.30),
            stance_width=_clamp(pred.get("stance_width", 0.25), 0.15, 0.45),
            impedance_scale=_clamp(pred.get("impedance_scale", 1.0), 0.40, 2.50),
            residual_gain_scale=_clamp(pred.get("residual_gain_scale", 1.0), 0.20, 2.50),
            risk_scale=_clamp(pred.get("risk_scale", 1.0), 0.50, 3.00),
            source_mode=inp.gait_mode,
            source_reason="torch_meta_gait_policy",
            extras={
                "policy_type": "learned_torch",
                "model_path": str(self.model_path),
                "ram_level": inp.ram.ram_level,
                "sigma": inp.ram.sigma,
                "beta_motion": inp.beta.motion,
                "beta_stability": inp.beta.stability,
                "beta_energy": inp.beta.energy,
            },
        )


def make_meta_gait_policy(kind: str = "rule_based", model_path: str | None = None) -> MetaGaitPolicy:
    kind = str(kind).strip().lower()

    if kind in {"rule", "rule_based", "stub", "default"}:
        return RuleBasedMetaGaitPolicy()

    if kind in {"torch", "learned"}:
        if not model_path:
            raise ValueError("model_path is required for learned/torch meta-gait policy")
        return TorchMetaGaitPolicy(Path(model_path).expanduser())

    raise ValueError(f"Unknown meta-gait policy kind: {kind}")
