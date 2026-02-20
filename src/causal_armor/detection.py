"""Dominance-shift detection (Algorithm 2, lines 11-12, Eq. 5).

Compares each untrusted span's causal influence against the user
request's influence. If a span dominates (within margin tau), it
signals a potential indirect prompt injection.
"""

from __future__ import annotations

from causal_armor.types import AttributionResult, DetectionResult


def detect_dominant_spans(
    attribution: AttributionResult,
    margin_tau: float,
) -> DetectionResult:
    """Flag untrusted spans whose influence dominates the user request.

    A span is flagged when its normalised causal influence exceeds
    the user's influence minus the margin threshold (Eq. 5):

        B_t(τ) = { S in S_t : D_S > D_U - τ }

    Parameters
    ----------
    attribution:
        LOO attribution result from :func:`compute_attribution`.
    margin_tau:
        Detection threshold tau (Eq. 5). At tau=0, any span more
        influential than the user triggers detection. Larger tau
        makes detection more sensitive (flags more spans).

    Returns
    -------
    DetectionResult
        Which spans are flagged and whether an attack was detected.
    """
    threshold = attribution.delta_user_normalized - margin_tau

    flagged: set[str] = set()
    for span_id, delta_s_norm in attribution.span_attributions_normalized.items():
        if delta_s_norm > threshold:
            flagged.add(span_id)

    flagged_frozen = frozenset(flagged)

    return DetectionResult(
        flagged_spans=flagged_frozen,
        is_attack_detected=bool(flagged_frozen),
        attribution=attribution,
        margin_tau=margin_tau,
    )
