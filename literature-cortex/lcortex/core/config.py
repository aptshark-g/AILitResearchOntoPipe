"""Configuration loader for Literature Cortex.

Loads configuration from:
  1. Default values (always available)
  2. ~/.lcortex/config.yaml (user override)
  3. LCORTEX_* environment variables (highest priority)

Usage:
    from lcortex.core.config import get_config
    cfg = get_config()
    print(cfg.llm.provider)
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_WORKSPACE = os.path.expanduser("~/lcortex-vault")
CONFIG_DIR = Path.home() / ".lcortex"
CONFIG_FILE = CONFIG_DIR / "config.yaml"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    """LLM adapter configuration."""
    provider: str = "none"
    api_key: str = ""
    model: str = ""
    fallback: str = ""

    @property
    def is_configured(self) -> bool:
        """Return True if a real LLM provider is configured."""
        return self.provider and self.provider != "none" and bool(self.api_key)


@dataclass
class WorkspaceConfig:
    """Workspace / vault configuration."""
    path: str = DEFAULT_WORKSPACE


@dataclass
class Config:
    """Top-level configuration container."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning an empty dict on any failure."""
    try:
        import yaml
    except ImportError:
        return {}
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _resolve_env_var(value: str) -> str:
    """Expand ${VAR} placeholders in a string using env vars."""
    if not isinstance(value, str):
        return value
    # Simple ${VAR} substitution
    import re
    def _sub(match):
        varname = match.group(1)
        return os.environ.get(varname, "")
    return re.sub(r"\$\{(\w+)\}", _sub, value)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base (mutates base)."""
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def _env_override(config_data: dict, prefix: str = "LCORTEX_") -> dict:
    """Apply env var overrides for keys like LCORTEX_LLM_PROVIDER."""
    env_map: dict[str, list] = {}
    for var, value in os.environ.items():
        if not var.startswith(prefix):
            continue
        # LCORTEX_LLM_PROVIDER → ["llm", "provider"]
        # LCORTEX_WORKSPACE_PATH → ["workspace", "path"]
        key_path = var[len(prefix):].lower().split("_")
        env_map.setdefault(".".join(key_path), []).append((key_path, value))

    for _, entries in env_map.items():
        for key_path, value in entries:
            # Walk into nested dict, creating as needed
            d = config_data
            for part in key_path[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[key_path[-1]] = value

    return config_data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_config() -> Config:
    """Load configuration: defaults → config.yaml → env vars.

    Returns a Config that is always usable — defaults make LLM optional.
    """
    data: dict = {"llm": {}, "workspace": {}}

    # Layer 1: file config
    file_data = _load_yaml(CONFIG_FILE)
    _deep_merge(data, file_data)

    # Layer 2: env var overrides
    data = _env_override(data)

    # Resolve ${VAR} placeholders in string values
    data = _resolve_config_values(data)

    # Build dataclass instances
    llm_cfg = LLMConfig(
        provider=data.get("llm", {}).get("provider", "none"),
        api_key=data.get("llm", {}).get("api_key", ""),
        model=data.get("llm", {}).get("model", ""),
        fallback=data.get("llm", {}).get("fallback", ""),
    )
    ws_cfg = WorkspaceConfig(
        path=data.get("workspace", {}).get("path", DEFAULT_WORKSPACE),
    )

    return Config(llm=llm_cfg, workspace=ws_cfg)


def _resolve_config_values(obj):
    """Recursively resolve ${VAR} in string values."""
    if isinstance(obj, dict):
        return {k: _resolve_config_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_config_values(v) for v in obj]
    if isinstance(obj, str):
        return _resolve_env_var(obj)
    return obj


def ensure_config_dir() -> Path:
    """Create ~/.lcortex if it doesn't exist; return the path."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR
