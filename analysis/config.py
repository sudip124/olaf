import os
import json
from typing import Any, Dict, Optional

# Configuration resolution order
# 1) ENV: ANALYSIS_DB_PATH
# 2) JSON file next to this module: analysis_config.json -> {"db_path": "..."}
# 3) Default path inside the analysis module folder

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DB_PATH = os.path.join(_THIS_DIR, "analysis.sqlite3")
_CONFIG_JSON = os.path.join(_THIS_DIR, "analysis_config.json")


def _load_config_file() -> Dict[str, Any]:
    try:
        if os.path.isfile(_CONFIG_JSON):
            with open(_CONFIG_JSON, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        # Silently ignore malformed or unreadable config; fall back to defaults
        pass
    return {}


def get_db_path(override: Optional[str] = None) -> str:
    """
    Resolve the database path for the analysis module.

    Priority: override > env ANALYSIS_DB_PATH > analysis_config.json > default
    """
    if override:
        return os.path.abspath(override)

    env_path = os.getenv("ANALYSIS_DB_PATH")
    if env_path:
        return os.path.abspath(env_path)

    cfg = _load_config_file()
    cfg_path = cfg.get("db_path") if isinstance(cfg, dict) else None
    if cfg_path:
        return os.path.abspath(cfg_path)

    return os.path.abspath(_DEFAULT_DB_PATH)


def ensure_db_dir(db_path: Optional[str] = None) -> str:
    """Ensure the parent directory for the db file exists and return absolute path."""
    path = get_db_path(db_path)
    db_dir = os.path.dirname(path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    return path


def get_engine(db_path: Optional[str] = None):
    """Create and return a SQLAlchemy engine for the analysis database."""
    from sqlalchemy import create_engine  # local import to avoid hard dep at import-time
    path = ensure_db_dir(db_path)
    return create_engine(f"sqlite:///{path}", echo=False)
