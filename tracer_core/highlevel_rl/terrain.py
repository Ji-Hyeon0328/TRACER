from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class M7TerrainPreset:
    """
    M7 high-level terrain contract.

    This describes environment-side terrain only.
    It does NOT modify M4, M5, or PyMPC controller logic.
    """

    name: str
    scene: str
    friction_coeff: float
    oracle_context: tuple[float, ...]
    description: str

    def __post_init__(self):
        if not self.name:
            raise ValueError(
                "terrain name must be non-empty"
            )

        if not self.scene:
            raise ValueError(
                "terrain scene must be non-empty"
            )

        if (
            not float(self.friction_coeff) > 0.0
        ):
            raise ValueError(
                "friction_coeff must be > 0"
            )

        if len(self.oracle_context) == 0:
            raise ValueError(
                "oracle_context must be non-empty"
            )


# ------------------------------------------------------------
# M7-v0 benchmark closure set
#
# IMPORTANT:
# low_friction=0.152 is the empirically frozen viable
# training coefficient from M7.2E characterization.
# The stress/evaluation coefficient is 0.150 and is
# supplied explicitly via terrain_friction override.
# ------------------------------------------------------------

M7_TERRAIN_PRESETS: Mapping[
    str,
    M7TerrainPreset,
] = {
    "flat": M7TerrainPreset(
        name="flat",
        scene="flat",
        friction_coeff=0.8,
        oracle_context=(
            1.0,
            0.0,
            0.0,
        ),
        description=(
            "flat reference terrain; "
            "nominal friction"
        ),
    ),

    "rough_boxes": M7TerrainPreset(
        name="rough_boxes",
        scene="random_boxes",
        friction_coeff=0.8,
        oracle_context=(
            0.0,
            1.0,
            0.0,
        ),
        description=(
            "existing Quadruped-PyMPC "
            "random-box rough terrain"
        ),
    ),

    "rough_perlin": M7TerrainPreset(
        name="rough_perlin",
        scene="perlin",
        friction_coeff=0.8,
        oracle_context=(
            0.0,
            1.0,
            0.0,
        ),
        description=(
            "seeded smooth procedural Perlin "
            "rough-terrain training candidate"
        ),
    ),

    "low_friction": M7TerrainPreset(
        name="low_friction",
        scene="flat",
        friction_coeff=0.152,
        oracle_context=(
            0.0,
            0.0,
            1.0,
        ),
        description=(
            "flat geometry with reduced friction; "
            "0.152 is the viable training coefficient"
        ),
    ),
}


def terrain_names() -> tuple[str, ...]:
    return tuple(
        M7_TERRAIN_PRESETS.keys()
    )


def get_terrain_preset(
    name: str,
    *,
    friction_override: float | None = None,
) -> M7TerrainPreset:
    key = str(name).strip()

    if key not in M7_TERRAIN_PRESETS:
        raise KeyError(
            f"unknown M7 terrain {key!r}; "
            f"available={terrain_names()}"
        )

    preset = M7_TERRAIN_PRESETS[key]

    if friction_override is None:
        return preset

    friction = float(
        friction_override
    )

    if friction <= 0.0:
        raise ValueError(
            "friction_override must be > 0"
        )

    return M7TerrainPreset(
        name=preset.name,
        scene=preset.scene,
        friction_coeff=friction,
        oracle_context=(
            preset.oracle_context
        ),
        description=(
            preset.description
            + f"; friction override={friction}"
        ),
    )


def oracle_context_for_terrain(
    name: str,
) -> tuple[float, ...]:
    return get_terrain_preset(
        name
    ).oracle_context
