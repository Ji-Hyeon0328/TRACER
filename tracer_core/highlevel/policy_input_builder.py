from __future__ import annotations

from typing import Any, Mapping, Sequence

from tracer_core.highlevel.gait_mode_selector import GaitModeInput, GaitModeOutput
from tracer_core.highlevel.meta_gait import (
    HighLevelPolicyInput,
    ObjectiveWeights,
    RamSignal,
    tuple_from_sequence,
)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _gate_get(gate: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if gate is None:
        return default
    return gate.get(key, default)


def build_ram_signal(
    *,
    gms_in: GaitModeInput | None = None,
    gate: Mapping[str, Any] | None = None,
    rho: Sequence[Any] | None = None,
) -> RamSignal:
    """Build RAM signal for HighLevelPolicyInput.

    Current runtime exposes scalar RAM/gate summaries on /tracer/ram_gate_advice.
    It does not yet expose the full latent rho vector to this node. For now:
      - sigma comes from RAM/gate when fresh, otherwise from GMS input / policy entry.
      - rho is either an explicitly provided vector or a one-dimensional placeholder
        containing rho_norm from RAM/gate.
    """

    ram_level = "unknown"
    sigma = 0.0
    control_risk = 0.0
    fallen_prob = 0.0
    recovery_prob = 0.0

    if gms_in is not None:
        ram_level = str(gms_in.ram_level)
        sigma = float(gms_in.sigma_mean)
        control_risk = float(gms_in.control_risk)
        fallen_prob = float(gms_in.fallen_prob)
        recovery_prob = float(gms_in.recovery_prob)

    if gate is not None:
        ram_level = str(_gate_get(gate, "ram_level", ram_level))
        sigma = _as_float(_gate_get(gate, "sigma_mean", sigma), sigma)
        control_risk = _as_float(_gate_get(gate, "control_risk", control_risk), control_risk)
        fallen_prob = _as_float(_gate_get(gate, "fallen_prob", fallen_prob), fallen_prob)
        recovery_prob = _as_float(_gate_get(gate, "recovery_prob", recovery_prob), recovery_prob)

    if rho is not None:
        rho_tuple = tuple_from_sequence(rho)
    else:
        rho_norm = _gate_get(gate, "rho_norm", None)
        rho_tuple = () if rho_norm is None else (_as_float(rho_norm, 0.0),)

    return RamSignal(
        rho=rho_tuple,
        sigma=sigma,
        ram_level=ram_level,
        control_risk=control_risk,
        fallen_prob=fallen_prob,
        recovery_prob=recovery_prob,
    )


def build_high_level_policy_input(
    *,
    policy_entry: Mapping[str, Any],
    gms_in: GaitModeInput | None = None,
    gms_out: GaitModeOutput | None = None,
    gate: Mapping[str, Any] | None = None,
    context: Sequence[Any] | None = None,
    robot_state: Sequence[Any] | None = None,
    goal: Sequence[Any] | None = None,
    rho: Sequence[Any] | None = None,
) -> HighLevelPolicyInput:
    """Build the canonical TRACER high-level policy input.

    This is the stable input contract for the future learned high-level policy:
      pi_high(c_t, rho, sigma, beta, z_mode, robot_state, goal) -> theta_t
    """

    beta = ObjectiveWeights.from_mapping(policy_entry.get("beta", {})).normalized()
    ram = build_ram_signal(gms_in=gms_in, gate=gate, rho=rho)

    if gms_out is not None:
        gait_mode = gms_out.mode
    elif gms_in is not None:
        gait_mode = gms_in.semantic_mode
    else:
        gait_mode = str(policy_entry.get("semantic_mode", "unknown"))

    return HighLevelPolicyInput(
        context=tuple_from_sequence(context),
        ram=ram,
        beta=beta,
        gait_mode=str(gait_mode),
        robot_state=tuple_from_sequence(robot_state),
        goal=tuple_from_sequence(goal),
    )


def high_level_policy_input_summary(inp: HighLevelPolicyInput) -> dict[str, Any]:
    """Compact serializable summary for logs/debugging."""

    return {
        "gait_mode": inp.gait_mode,
        "context_dim": len(inp.context),
        "rho_dim": len(inp.ram.rho),
        "rho_norm_feature": inp.ram.rho[0] if inp.ram.rho else 0.0,
        "sigma": inp.ram.sigma,
        "ram_level": inp.ram.ram_level,
        "control_risk": inp.ram.control_risk,
        "fallen_prob": inp.ram.fallen_prob,
        "recovery_prob": inp.ram.recovery_prob,
        "beta_motion": inp.beta.motion,
        "beta_stability": inp.beta.stability,
        "beta_energy": inp.beta.energy,
        "robot_state_dim": len(inp.robot_state),
        "goal_dim": len(inp.goal),
    }
