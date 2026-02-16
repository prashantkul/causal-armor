"""CausalArmor configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CausalArmorConfig:
    r"""Top-level configuration for the CausalArmor pipeline.

    Parameters
    ----------
    margin_tau:
        Detection threshold τ (Eq. 5 in the paper).  When τ = 0 the
        detector flags any span whose causal influence exceeds the user
        request's (pure causal-inversion mode).
    privileged_tools:
        Set of tool names T_priv whose results are trusted and skip
        attribution.
    cot_redaction_text:
        Replacement text inserted in place of chain-of-thought messages
        during CoT masking.
    enable_cot_masking:
        Whether to redact assistant reasoning before regeneration.
    enable_sanitization:
        Whether to run the sanitizer on flagged spans.
    max_loo_batch_size:
        Optional cap on the number of concurrent LOO scoring requests.
        ``None`` means no limit.
    log_attributions:
        Whether to emit attribution diagnostics.
    """

    margin_tau: float = 0.0
    privileged_tools: frozenset[str] = field(default_factory=frozenset)
    cot_redaction_text: str = "[Reasoning redacted for security]"
    enable_cot_masking: bool = True
    enable_sanitization: bool = True
    max_loo_batch_size: int | None = None
    log_attributions: bool = True
