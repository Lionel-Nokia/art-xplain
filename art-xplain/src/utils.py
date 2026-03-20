from __future__ import annotations
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def resolve_project_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p

    root = Path(base_dir) if base_dir is not None else PROJECT_ROOT
    return (root / p).resolve()


def relativize_project_path(path: str | Path, *, base_dir: str | Path | None = None) -> str:
    p = resolve_project_path(path, base_dir=base_dir)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        # Si le fichier est en dehors du projet, on retombe sur un chemin absolu.
        return str(p)


def resolve_stored_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return resolve_project_path(p, base_dir=base_dir)


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    config_path = Path(config_path).expanduser()

    if config_path.is_absolute():
        resolved_config_path = config_path
    else:
        cwd_candidate = (Path.cwd() / config_path).resolve()
        project_candidate = resolve_project_path(config_path)

        if cwd_candidate.exists():
            resolved_config_path = cwd_candidate
        else:
            resolved_config_path = project_candidate

    if not resolved_config_path.exists():
        raise FileNotFoundError(
            f"config.yaml introuvable à {resolved_config_path}"
        )

    with resolved_config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
