from pathlib import Path


def tracer_repo_root() -> Path:
    # /repo/isaaclab_tracer/utils/paths.py -> /repo
    return Path(__file__).resolve().parents[2]


def tracer_config_path(relative_path: str) -> str:
    return str(tracer_repo_root() / relative_path)
