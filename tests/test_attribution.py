"""Tests for compute_attribution."""

from __future__ import annotations

import pytest
from conftest import MockProxyAttack, MockProxyBenign

from causal_armor import ToolCall, compute_attribution


@pytest.mark.asyncio
async def test_attack_attribution(attack_context, malicious_action):
    attr = await compute_attribution(
        attack_context, malicious_action, MockProxyAttack()
    )

    # Span should have much higher delta than user
    span_id = "web_search:3"
    assert attr.span_attributions[span_id] > attr.delta_user
    assert attr.span_attributions_normalized[span_id] > attr.delta_user_normalized

    # Base logprob should be the full-context score
    assert attr.base_logprob == -2.0
    assert attr.user_ablated_logprob == -2.5

    # Action token count should be positive
    assert attr.action_token_count > 0


@pytest.mark.asyncio
async def test_benign_attribution(benign_context, safe_action):
    attr = await compute_attribution(benign_context, safe_action, MockProxyBenign())

    # User should have much higher delta than span
    span_id = "web_search:3"
    assert attr.delta_user > attr.span_attributions[span_id]
    assert attr.delta_user_normalized > attr.span_attributions_normalized[span_id]


@pytest.mark.asyncio
async def test_attribution_with_semaphore(attack_context, malicious_action):
    """max_concurrent limits parallelism but produces same results."""
    attr = await compute_attribution(
        attack_context, malicious_action, MockProxyAttack(), max_concurrent=1
    )
    span_id = "web_search:3"
    assert attr.span_attributions[span_id] > attr.delta_user


@pytest.mark.asyncio
async def test_attribution_token_count():
    """Token count approximation uses ~4 chars/token heuristic."""
    from causal_armor import Message, MessageRole, build_structured_context

    msgs = [
        Message(role=MessageRole.USER, content="Go"),
        Message(role=MessageRole.TOOL, content="Data", tool_name="t"),
    ]
    ctx = build_structured_context(msgs, frozenset({"t"}))

    # "one two three four" = 18 chars, 18 // 4 = 4 tokens
    action = ToolCall(name="act", arguments={}, raw_text="one two three four")

    class ConstProxy:
        async def log_prob(self, messages, action_text):
            return -1.0

    attr = await compute_attribution(ctx, action, ConstProxy())
    assert attr.action_token_count == 4


@pytest.mark.asyncio
async def test_attribution_empty_spans():
    """Context with no untrusted spans: only user ablation."""
    from causal_armor import Message, MessageRole, build_structured_context

    msgs = [
        Message(role=MessageRole.USER, content="Hello"),
        Message(role=MessageRole.ASSISTANT, content="Hi"),
    ]
    ctx = build_structured_context(msgs, frozenset())

    action = ToolCall(name="greet", arguments={}, raw_text="hello")

    class ConstProxy:
        async def log_prob(self, messages, action_text):
            return -1.0

    attr = await compute_attribution(ctx, action, ConstProxy())
    assert len(attr.span_attributions) == 0
    assert attr.delta_user == 0.0  # same logprob with or without user
