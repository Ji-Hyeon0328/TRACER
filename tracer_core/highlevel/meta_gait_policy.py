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
            vx = 0.05
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


@dataclass(frozen=True)
class TorchMetaGaitPolicyStub:
    """Placeholder for a future learned torch policy.

    This class intentionally does not load a model yet. It keeps the expected API
    stable so later training can replace RuleBasedMetaGaitPolicy without changing
    the decoder/mapper or ROS2 publisher.
    """

    model_path: Path | None = None

    def predict(self, inp: HighLevelPolicyInput) -> MetaGaitCommand:
        raise NotImplementedError(
            "TorchMetaGaitPolicyStub is an interface placeholder. "
            "Train/export a learned meta-gait policy before using it."
        )


def make_meta_gait_policy(kind: str = "rule_based", model_path: str | None = None) -> MetaGaitPolicy:
    kind = str(kind).strip().lower()

    if kind in {"rule", "rule_based", "stub", "default"}:
        return RuleBasedMetaGaitPolicy()

    if kind in {"torch", "learned"}:
        return TorchMetaGaitPolicyStub(Path(model_path).expanduser() if model_path else None)

    raise ValueError(f"Unknown meta-gait policy kind: {kind}")
