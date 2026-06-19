"""Optional shared configuration helpers for publication-ready wrappers.

These helpers are intentionally conservative: they load YAML config files only when
a caller explicitly passes a config path. They do not override scientific defaults
unless an individual script is later wired to do so deliberately.
"""

from __future__ import annotations


def load_yaml_config(config_path):
    """Load an optional YAML config file and return an empty dict when omitted."""
    if not config_path:
        return {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("Optional --config support requires PyYAML when a config path is provided") from exc
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
