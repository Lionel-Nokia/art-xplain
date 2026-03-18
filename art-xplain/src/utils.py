from __future__ import annotations
from pathlib import Path
import yaml

def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml introuvable à {config_path.resolve()}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
