"""CausalArmor configuration.

Configuration is resolved with the following precedence (highest first):

1. Explicit keyword arguments passed to constructors / ``from_env()``
2. ``CAUSAL_ARMOR_*`` environment variables (secrets, deployment config)
3. ``causal_armor.toml`` file (tuning parameters, version-controlled)
4. Built-in defaults
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Env-var helpers
# ---------------------------------------------------------------------------


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


def _env_int_or_none(key: str, default: int | None) -> int | None:
    val = os.environ.get(key)
    if val is None:
        return default
    return int(val)


# ---------------------------------------------------------------------------
# TOML loader
# ---------------------------------------------------------------------------

_TOML_FILENAME = "causal_armor.toml"


def _find_toml(start: Path | None = None) -> Path | None:
    """Locate ``causal_armor.toml``.

    Resolution order:

    1. *start* argument (if provided)
    2. ``CAUSAL_ARMOR_CONFIG_PATH`` env var (async-safe, no cwd lookup)
    3. Walk up from cwd (fallback)
    """
    # Env var takes precedence — avoids blocking Path.cwd() in async contexts
    env_path = os.environ.get("CAUSAL_ARMOR_CONFIG_PATH")
    if env_path is not None:
        p = Path(env_path)
        return p if p.is_file() else None

    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / _TOML_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _load_toml(path: Path | None = None) -> dict[str, Any]:
    """Load and return the ``[causal_armor]`` table, or ``{}`` if missing."""
    toml_path = path or _find_toml()
    if toml_path is None:
        return {}
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    return dict(data.get("causal_armor", data))


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CausalArmorConfig:
    r"""Top-level configuration for the CausalArmor pipeline.

    Parameters
    ----------
    margin_tau:
        Detection threshold τ (Eq. 5 in the paper).  When τ = 0 the
        detector flags any span whose causal influence exceeds the user
        request's (pure causal-inversion mode).

        TOML: ``margin_tau``  |  Env: ``CAUSAL_ARMOR_MARGIN_TAU``
    privileged_tools:
        Set of tool names T_priv whose results are trusted and skip
        attribution.

        TOML: ``privileged_tools``
    cot_redaction_text:
        Replacement text inserted in place of chain-of-thought messages
        during CoT masking.

        TOML: ``cot_redaction_text``
    enable_cot_masking:
        Whether to redact assistant reasoning before regeneration.

        TOML: ``enable_cot_masking``  |  Env: ``CAUSAL_ARMOR_ENABLE_COT_MASKING``
    mask_cot_for_scoring:
        Whether to mask assistant reasoning in the context *before* LOO
        scoring.  In multi-turn conversations the agent's reasoning may
        propagate injected instructions, masking the true causal signal.
        Enabling this isolates the influence of external inputs (user
        request, tool results) during attribution.

        TOML: ``mask_cot_for_scoring``  |  Env: ``CAUSAL_ARMOR_MASK_COT_FOR_SCORING``
    enable_sanitization:
        Whether to run the sanitizer on flagged spans.

        TOML: ``enable_sanitization``  |  Env: ``CAUSAL_ARMOR_ENABLE_SANITIZATION``
    max_loo_batch_size:
        Optional cap on the number of concurrent LOO scoring requests.
        ``None`` means no limit.

        TOML: ``max_loo_batch_size``  |  Env: ``CAUSAL_ARMOR_MAX_LOO_BATCH_SIZE``
    log_attributions:
        Whether to emit attribution diagnostics.

        TOML: ``log_attributions``  |  Env: ``CAUSAL_ARMOR_LOG_ATTRIBUTIONS``
    """

    margin_tau: float = 0.0
    privileged_tools: frozenset[str] = field(default_factory=frozenset)
    cot_redaction_text: str = "[Reasoning redacted for security]"
    enable_cot_masking: bool = True
    mask_cot_for_scoring: bool = True
    enable_sanitization: bool = True
    max_loo_batch_size: int | None = None
    log_attributions: bool = True

    @classmethod
    def from_env(
        cls, *, config_path: Path | None = None, **overrides: Any
    ) -> CausalArmorConfig:
        """Build a config from TOML file + env vars + explicit overrides.

        Resolution order (highest precedence first):

        1. **overrides** — keyword arguments passed directly
        2. **env vars** — ``CAUSAL_ARMOR_*`` environment variables
        3. **TOML file** — ``causal_armor.toml`` (searched upward from cwd)
        4. **defaults** — built-in dataclass defaults
        """
        # Layer 1: TOML file (lowest precedence of the three sources)
        toml = _load_toml(config_path)

        # Layer 2: env vars override TOML
        values: dict[str, Any] = {
            "margin_tau": _env_float(
                "CAUSAL_ARMOR_MARGIN_TAU",
                float(toml.get("margin_tau", 0.0)),
            ),
            "enable_cot_masking": _env_bool(
                "CAUSAL_ARMOR_ENABLE_COT_MASKING",
                bool(toml.get("enable_cot_masking", True)),
            ),
            "mask_cot_for_scoring": _env_bool(
                "CAUSAL_ARMOR_MASK_COT_FOR_SCORING",
                bool(toml.get("mask_cot_for_scoring", True)),
            ),
            "enable_sanitization": _env_bool(
                "CAUSAL_ARMOR_ENABLE_SANITIZATION",
                bool(toml.get("enable_sanitization", True)),
            ),
            "max_loo_batch_size": _env_int_or_none(
                "CAUSAL_ARMOR_MAX_LOO_BATCH_SIZE",
                toml.get("max_loo_batch_size"),
            ),
            "log_attributions": _env_bool(
                "CAUSAL_ARMOR_LOG_ATTRIBUTIONS",
                bool(toml.get("log_attributions", True)),
            ),
        }

        # TOML-only fields (no env var equivalent)
        if "privileged_tools" in toml:
            values["privileged_tools"] = frozenset(toml["privileged_tools"])
        if "cot_redaction_text" in toml:
            values["cot_redaction_text"] = str(toml["cot_redaction_text"])

        # Layer 3: explicit overrides (highest precedence)
        values.update({k: v for k, v in overrides.items() if v is not None})

        return cls(**values)
