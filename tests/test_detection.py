"""Tests for detect_dominant_spans."""

from __future__ import annotations

from causal_armor import AttributionResult, detect_dominant_spans


def _make_attribution(
    delta_user_norm: float,
    span_norms: dict[str, float],
) -> AttributionResult:
    """Helper to create an AttributionResult with specific normalized deltas."""
    return AttributionResult(
        delta_user=delta_user_norm * 2,  # raw = norm * token_count
        delta_user_normalized=delta_user_norm,
        span_attributions={sid: v * 2 for sid, v in span_norms.items()},
        span_attributions_normalized=span_norms,
        base_logprob=-2.0,
        ablated_logprobs={sid: -2.0 - v * 2 for sid, v in span_norms.items()},
        user_ablated_logprob=-2.0 - delta_user_norm * 2,
        action_token_count=2,
    )


class TestDetectDominantSpans:
    def test_span_dominates_user_tau_zero(self):
        """Span with higher influence than user is flagged at tau=0."""
        attr = _make_attribution(
            delta_user_norm=1.0,
            span_norms={"s1": 3.0},
        )
        det = detect_dominant_spans(attr, margin_tau=0.0)
        assert det.is_attack_detected
        assert "s1" in det.flagged_spans

    def test_user_dominates_span_tau_zero(self):
        """Span with lower influence than user is NOT flagged at tau=0."""
        attr = _make_attribution(
            delta_user_norm=5.0,
            span_norms={"s1": 1.0},
        )
        det = detect_dominant_spans(attr, margin_tau=0.0)
        assert not det.is_attack_detected
        assert len(det.flagged_spans) == 0

    def test_equal_influence_tau_zero(self):
        """Equal influence: NOT flagged at tau=0 (need strictly greater)."""
        attr = _make_attribution(
            delta_user_norm=3.0,
            span_norms={"s1": 3.0},
        )
        det = detect_dominant_spans(attr, margin_tau=0.0)
        assert not det.is_attack_detected

    def test_positive_tau_makes_detection_more_sensitive(self):
        """Positive tau lowers the threshold, flagging more spans."""
        attr = _make_attribution(
            delta_user_norm=5.0,
            span_norms={"s1": 3.0},
        )
        # tau=0: 3.0 > 5.0 - 0 = 5.0 → False
        det0 = detect_dominant_spans(attr, margin_tau=0.0)
        assert not det0.is_attack_detected

        # tau=3: 3.0 > 5.0 - 3.0 = 2.0 → True
        det3 = detect_dominant_spans(attr, margin_tau=3.0)
        assert det3.is_attack_detected
        assert "s1" in det3.flagged_spans

    def test_negative_tau_makes_detection_stricter(self):
        """Negative tau raises the threshold, flagging fewer spans."""
        attr = _make_attribution(
            delta_user_norm=1.0,
            span_norms={"s1": 3.0},
        )
        # tau=0: 3.0 > 1.0 → True
        det0 = detect_dominant_spans(attr, margin_tau=0.0)
        assert det0.is_attack_detected

        # tau=-10: 3.0 > 1.0 - (-10) = 11.0 → False
        det_neg = detect_dominant_spans(attr, margin_tau=-10.0)
        assert not det_neg.is_attack_detected

    def test_multiple_spans_partial_flagging(self):
        """Only spans exceeding threshold are flagged."""
        attr = _make_attribution(
            delta_user_norm=2.0,
            span_norms={"s1": 5.0, "s2": 1.0, "s3": 3.0},
        )
        det = detect_dominant_spans(attr, margin_tau=0.0)
        assert det.is_attack_detected
        assert "s1" in det.flagged_spans
        assert "s3" in det.flagged_spans
        assert "s2" not in det.flagged_spans

    def test_no_spans(self):
        """No spans means no detection."""
        attr = _make_attribution(delta_user_norm=5.0, span_norms={})
        det = detect_dominant_spans(attr, margin_tau=0.0)
        assert not det.is_attack_detected
        assert len(det.flagged_spans) == 0

    def test_result_preserves_attribution(self):
        """DetectionResult carries the attribution it was based on."""
        attr = _make_attribution(delta_user_norm=1.0, span_norms={"s1": 5.0})
        det = detect_dominant_spans(attr, margin_tau=0.5)
        assert det.attribution is attr
        assert det.margin_tau == 0.5
