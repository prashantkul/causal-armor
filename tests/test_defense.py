"""Tests for defense layer: sanitize, CoT mask, defend."""

from __future__ import annotations

import pytest
from conftest import MockActionProvider, MockSanitizer

from causal_armor import (
    AttributionResult,
    CausalArmorConfig,
    DetectionResult,
    Message,
    MessageRole,
    ToolCall,
    build_structured_context,
    defend,
    mask_cot_after_detection,
    sanitize_flagged_spans,
)


def _make_detection(flagged: frozenset[str], is_attack: bool) -> DetectionResult:
    attr = AttributionResult(
        delta_user=1.0,
        delta_user_normalized=0.5,
        span_attributions={sid: 5.0 for sid in flagged},
        span_attributions_normalized={sid: 2.5 for sid in flagged},
        base_logprob=-2.0,
        ablated_logprobs={sid: -7.0 for sid in flagged},
        user_ablated_logprob=-3.0,
        action_token_count=2,
    )
    return DetectionResult(
        flagged_spans=flagged,
        is_attack_detected=is_attack,
        attribution=attr,
        margin_tau=0.0,
    )


class TestSanitizeFlaggedSpans:
    @pytest.mark.asyncio
    async def test_sanitizes_flagged_span(self, attack_context):
        det = _make_detection(frozenset({"web_search:3"}), True)
        new_ctx, sanitized = await sanitize_flagged_spans(
            attack_context, det, MockSanitizer()
        )
        assert "web_search:3" in sanitized
        assert new_ctx.full_messages[3].content == "Flight AA123 to Paris, $450."
        assert (
            new_ctx.untrusted_spans["web_search:3"].content
            == "Flight AA123 to Paris, $450."
        )

    @pytest.mark.asyncio
    async def test_sanitizes_only_flagged_spans(self, attack_context):
        """Only flagged spans B_t(τ) are sanitized, matching the paper."""
        det = _make_detection(frozenset({"web_search:3"}), True)
        new_ctx, sanitized = await sanitize_flagged_spans(
            attack_context, det, MockSanitizer()
        )
        # Only the flagged span should be sanitized
        assert set(sanitized.keys()) == {"web_search:3"}

    @pytest.mark.asyncio
    async def test_multiple_flagged_spans(self):
        msgs = [
            Message(role=MessageRole.USER, content="Go"),
            Message(role=MessageRole.TOOL, content="Bad 1", tool_name="t"),
            Message(role=MessageRole.TOOL, content="Bad 2", tool_name="t"),
        ]
        ctx = build_structured_context(msgs, frozenset({"t"}))
        det = _make_detection(frozenset({"t:1", "t:2"}), True)
        _new_ctx, sanitized = await sanitize_flagged_spans(ctx, det, MockSanitizer())
        assert len(sanitized) == 2


class TestMaskCotAfterDetection:
    def test_masks_assistant_after_span(self, attack_context):
        det = _make_detection(frozenset({"web_search:3"}), True)
        masked = mask_cot_after_detection(attack_context, det, "[REDACTED]")
        # msg[4] is assistant after span at index 3
        assert masked.full_messages[4].content == "[REDACTED]"
        # msg[2] is assistant before span
        assert masked.full_messages[2].content == "Let me search for flights."

    def test_no_flagged_spans_returns_unchanged(self, attack_context):
        det = _make_detection(frozenset(), False)
        result = mask_cot_after_detection(attack_context, det, "[X]")
        assert result is attack_context

    def test_uses_earliest_span_for_k_min(self):
        msgs = [
            Message(role=MessageRole.USER, content="Go"),
            Message(role=MessageRole.TOOL, content="A", tool_name="t"),
            Message(role=MessageRole.ASSISTANT, content="Reasoning 1"),
            Message(role=MessageRole.TOOL, content="B", tool_name="t"),
            Message(role=MessageRole.ASSISTANT, content="Reasoning 2"),
        ]
        ctx = build_structured_context(msgs, frozenset({"t"}))
        det = _make_detection(frozenset({"t:1", "t:3"}), True)
        masked = mask_cot_after_detection(ctx, det, "[X]")
        # k_min = 1, so assistant at [2] and [4] should be masked
        assert masked.full_messages[2].content == "[X]"
        assert masked.full_messages[4].content == "[X]"


class TestDefend:
    @pytest.mark.asyncio
    async def test_no_attack_passthrough(self, attack_context, malicious_action):
        det = _make_detection(frozenset(), False)
        config = CausalArmorConfig()
        result = await defend(
            attack_context,
            malicious_action,
            det,
            MockSanitizer(),
            MockActionProvider(),
            config,
        )
        assert not result.was_defended
        assert result.final_action is malicious_action
        assert not result.regenerated

    @pytest.mark.asyncio
    async def test_full_defense(self, attack_context, malicious_action):
        det = _make_detection(frozenset({"web_search:3"}), True)
        config = CausalArmorConfig()
        result = await defend(
            attack_context,
            malicious_action,
            det,
            MockSanitizer(),
            MockActionProvider(),
            config,
        )
        assert result.was_defended
        assert result.regenerated
        assert result.cot_messages_masked
        assert result.final_action.name == "book_flight"
        assert result.original_action.name == "send_money"
        assert "web_search:3" in result.sanitized_spans

    @pytest.mark.asyncio
    async def test_sanitization_disabled(self, attack_context, malicious_action):
        det = _make_detection(frozenset({"web_search:3"}), True)
        config = CausalArmorConfig(enable_sanitization=False)
        result = await defend(
            attack_context,
            malicious_action,
            det,
            MockSanitizer(),
            MockActionProvider(),
            config,
        )
        assert result.was_defended
        assert len(result.sanitized_spans) == 0
        assert result.cot_messages_masked

    @pytest.mark.asyncio
    async def test_cot_masking_disabled(self, attack_context, malicious_action):
        det = _make_detection(frozenset({"web_search:3"}), True)
        config = CausalArmorConfig(enable_cot_masking=False)
        result = await defend(
            attack_context,
            malicious_action,
            det,
            MockSanitizer(),
            MockActionProvider(),
            config,
        )
        assert result.was_defended
        assert not result.cot_messages_masked
        assert len(result.sanitized_spans) > 0

    @pytest.mark.asyncio
    async def test_no_fallback_to_original_on_empty_regeneration(
        self, attack_context, malicious_action
    ):
        """When regeneration produces no tool call, must NOT fall back to original."""

        class EmptyActionProvider:
            async def generate(self, messages):
                return ("I need more information.", [])

        det = _make_detection(frozenset({"web_search:3"}), True)
        config = CausalArmorConfig()
        result = await defend(
            attack_context,
            malicious_action,
            det,
            MockSanitizer(),
            EmptyActionProvider(),
            config,
        )
        assert result.was_defended
        assert not result.regenerated
        # Must not contain the original attacker-controlled arguments
        assert result.final_action.arguments == {}
        assert result.final_action.name == malicious_action.name

    @pytest.mark.asyncio
    async def test_regeneration_context_has_no_attacker_artifacts(
        self, attack_context, malicious_action
    ):
        """Regeneration context must not leak attacker-controlled tool-call data."""
        captured_messages: list[tuple] = []

        class CapturingActionProvider:
            async def generate(self, messages):
                captured_messages.append(tuple(messages))
                return (
                    "book_flight flight=AA123",
                    [ToolCall(name="book_flight", arguments={"flight": "AA123"}, raw_text="book_flight flight=AA123")],
                )

        det = _make_detection(frozenset({"web_search:3"}), True)
        config = CausalArmorConfig()
        await defend(
            attack_context,
            malicious_action,
            det,
            MockSanitizer(),
            CapturingActionProvider(),
            config,
        )

        assert len(captured_messages) == 1
        regen_msgs = captured_messages[0]

        # Trailing assistant message (blocked action proposal) must be dropped
        assert regen_msgs[-1].role is not MessageRole.ASSISTANT

        # No message should carry attacker-controlled metadata or tool-call refs
        for msg in regen_msgs:
            if msg.role is MessageRole.ASSISTANT:
                assert msg.tool_name is None
                assert msg.tool_call_id is None
                assert msg.metadata == {}
            # Attacker values must not appear in any content
            for field in (msg.content, str(msg.metadata)):
                assert "XYZ" not in field
                assert "10000" not in field
