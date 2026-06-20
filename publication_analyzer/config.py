"""Configuration for the publication analyzer.

Independent of the tracker's config. Settings come from (in order of
precedence): environment variables, then an optional YAML file. The Gemini API
key should live in the environment / a local .env, never in committed YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - yaml is a listed dependency
    yaml = None


@dataclass
class Config:
    gemini_api_key: str = ""
    model: str = "gemini-2.5-flash"
    # Journals counted as "top-tier". Empty means use the built-in default
    # (FT50 ∪ UTD24; see analysis.DEFAULT_TOP_TIER_JOURNALS).
    top_tier_journals: List[str] = field(default_factory=list)
    # Contact email for OpenAlex's "polite pool" (optional but recommended for
    # faster, more reliable responses). Set via PA_MAILTO or the YAML `mailto`.
    mailto: str = ""
    # How to fetch program pages when scraping: "auto" renders via a headless
    # browser only when a page looks JS-rendered, "always" renders every page,
    # "never" uses plain HTTP. Set via PA_RENDER or the YAML `render`.
    render: str = "auto"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from a YAML file (optional) overlaid with env vars."""
    data: dict = {}
    if path and os.path.exists(path):
        if yaml is None:
            raise RuntimeError("PyYAML is required to read a config file.")
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    return Config(
        # Accept GEMINI_API_KEY or GOOGLE_API_KEY (the SDK reads either, too).
        gemini_api_key=_env(
            "GEMINI_API_KEY",
            _env("GOOGLE_API_KEY", data.get("gemini_api_key", "")),
        ),
        model=_env("PA_MODEL", data.get("model", "gemini-2.5-flash")),
        top_tier_journals=list(data.get("top_tier_journals", []) or []),
        mailto=_env("PA_MAILTO", data.get("mailto", "")),
        render=_env("PA_RENDER", data.get("render", "auto")),
    )
