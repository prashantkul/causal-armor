"""Tests for StructuredContext and build_structured_context."""

from __future__ import annotations

import pytest

from causal_armor import (
    ContextError,
    Message,
    MessageRole,
    build_structured_context,
)


class TestBuildStructuredContext:
    def test_picks_last_user_message(self):
        msgs = [
            Message(role=MessageRole.USER, content="First request"),
            Message(role=MessageRole.ASSISTANT, content="OK"),
            Message(role=MessageRole.USER, content="Second request"),
        ]
        ctx = build_structured_context(msgs, frozenset())
        assert ctx.user_request.content == "Second request"

    def test_collects_system_messages(self):
        msgs = [
            Message(role=MessageRole.SYSTEM, content="System 1"),
            Message(role=MessageRole.SYSTEM, content="System 2"),
            Message(role=MessageRole.USER, content="Hello"),
        ]
        ctx = build_structured_context(msgs, frozenset())
        assert len(ctx.system_messages) == 2

    def test_identifies_untrusted_spans(self):
        msgs = [
            Message(role=MessageRole.USER, content="Search"),
            Message(role=MessageRole.TOOL, content="result1", tool_name="web_search"),
            Message(role=MessageRole.TOOL, content="result2", tool_name="calculator"),
            Message(role=MessageRole.TOOL, content="result3", tool_name="web_search"),
        ]
        ctx = build_structured_context(msgs, frozenset({"web_search"}))
        assert len(ctx.untrusted_spans) == 2
        assert "web_search:1" in ctx.span_ids
        assert "web_search:3" in ctx.span_ids
        # calculator is trusted
        assert not any("calculator" in sid for sid in ctx.span_ids)

    def test_span_id_format(self):
        msgs = [
            Message(role=MessageRole.USER, content="Go"),
            Message(role=MessageRole.TOOL, content="data", tool_name="api"),
        ]
        ctx = build_structured_context(msgs, frozenset({"api"}))
        assert "api:1" in ctx.span_ids

    def test_preserves_full_messages(self):
        msgs = [
            Message(role=MessageRole.USER, content="Hi"),
            Message(role=MessageRole.ASSISTANT, content="Hello"),
        ]
        ctx = build_structured_context(msgs, frozenset())
        assert ctx.full_messages == tuple(msgs)

    def test_raises_on_no_user_message(self):
        msgs = [
            Message(role=MessageRole.SYSTEM, content="System"),
            Message(role=MessageRole.ASSISTANT, content="Hello"),
        ]
        with pytest.raises(ContextError, match="No user-role message"):
            build_structured_context(msgs, frozenset())

    def test_turn_index_propagated(self):
        msgs = [
            Message(role=MessageRole.USER, content="Go"),
            Message(role=MessageRole.TOOL, content="data", tool_name="t"),
        ]
        ctx = build_structured_context(msgs, frozenset({"t"}), turn_index=5)
        span = next(iter(ctx.untrusted_spans.values()))
        assert span.turn_index == 5

    def test_no_untrusted_tools_means_no_spans(self):
        msgs = [
            Message(role=MessageRole.USER, content="Hi"),
            Message(role=MessageRole.TOOL, content="data", tool_name="tool"),
        ]
        ctx = build_structured_context(msgs, frozenset())
        assert not ctx.has_untrusted_spans

    def test_tool_without_name_ignored(self):
        msgs = [
            Message(role=MessageRole.USER, content="Hi"),
            Message(role=MessageRole.TOOL, content="data", tool_name=None),
        ]
        ctx = build_structured_context(msgs, frozenset({"data"}))
        assert not ctx.has_untrusted_spans


class TestStructuredContext:
    def _make_ctx(self):
        msgs = [
            Message(role=MessageRole.SYSTEM, content="System"),
            Message(role=MessageRole.USER, content="User request"),
            Message(role=MessageRole.ASSISTANT, content="Thinking..."),
            Message(
                role=MessageRole.TOOL, content="Untrusted data", tool_name="tool_a"
            ),
            Message(role=MessageRole.ASSISTANT, content="Based on tool result..."),
        ]
        return build_structured_context(msgs, frozenset({"tool_a"}))

    def test_has_untrusted_spans(self):
        ctx = self._make_ctx()
        assert ctx.has_untrusted_spans

    def test_span_ids(self):
        ctx = self._make_ctx()
        assert ctx.span_ids == frozenset({"tool_a:3"})

    def test_messages_without_user_request(self):
        ctx = self._make_ctx()
        result = ctx.messages_without_user_request()
        assert len(result) == 4
        assert all(m.content != "User request" for m in result)

    def test_messages_without_span(self):
        ctx = self._make_ctx()
        result = ctx.messages_without_span("tool_a:3")
        assert len(result) == 4
        assert all(m.content != "Untrusted data" for m in result)

    def test_messages_without_span_unknown_id(self):
        ctx = self._make_ctx()
        with pytest.raises(ContextError, match="Unknown span_id"):
            ctx.messages_without_span("nonexistent:99")

    def test_replace_span_content(self):
        ctx = self._make_ctx()
        new_ctx = ctx.replace_span_content("tool_a:3", "Sanitized data")
        assert new_ctx.full_messages[3].content == "Sanitized data"
        assert new_ctx.untrusted_spans["tool_a:3"].content == "Sanitized data"
        # Original unchanged (frozen)
        assert ctx.full_messages[3].content == "Untrusted data"

    def test_replace_span_preserves_metadata(self):
        ctx = self._make_ctx()
        new_ctx = ctx.replace_span_content("tool_a:3", "Clean")
        assert new_ctx.full_messages[3].tool_name == "tool_a"
        assert new_ctx.full_messages[3].role == MessageRole.TOOL

    def test_replace_span_unknown_id(self):
        ctx = self._make_ctx()
        with pytest.raises(ContextError, match="Unknown span_id"):
            ctx.replace_span_content("bad:0", "x")

    def test_mask_assistant_messages_after(self):
        ctx = self._make_ctx()
        masked = ctx.mask_assistant_messages_after(3, "[REDACTED]")
        # msg[4] is assistant after index 3 -> redacted
        assert masked.full_messages[4].content == "[REDACTED]"
        # msg[2] is assistant before index 3 -> unchanged
        assert masked.full_messages[2].content == "Thinking..."
        # msg[3] is tool at index 3 -> unchanged
        assert masked.full_messages[3].content == "Untrusted data"

    def test_mask_preserves_non_assistant(self):
        ctx = self._make_ctx()
        masked = ctx.mask_assistant_messages_after(0, "[X]")
        # System at [0] unchanged, user at [1] unchanged
        assert masked.full_messages[0].content == "System"
        assert masked.full_messages[1].content == "User request"
        # Tool at [3] unchanged
        assert masked.full_messages[3].content == "Untrusted data"

    def test_multiple_spans(self):
        msgs = [
            Message(role=MessageRole.USER, content="Go"),
            Message(role=MessageRole.TOOL, content="A", tool_name="t"),
            Message(role=MessageRole.TOOL, content="B", tool_name="t"),
        ]
        ctx = build_structured_context(msgs, frozenset({"t"}))
        assert len(ctx.untrusted_spans) == 2
        assert "t:1" in ctx.span_ids
        assert "t:2" in ctx.span_ids

    def test_mask_clears_tool_call_artifacts(self):
        """Masked assistant messages must not retain tool-call metadata."""
        msgs = [
            Message(role=MessageRole.USER, content="Go"),
            Message(role=MessageRole.TOOL, content="data", tool_name="t"),
            Message(
                role=MessageRole.ASSISTANT,
                content="Calling send_money",
                tool_name="send_money",
                tool_call_id="call_002",
                metadata={
                    "tool_calls": [{"name": "send_money", "args": {"amount": 100}}],
                },
            ),
        ]
        ctx = build_structured_context(msgs, frozenset({"t"}))
        masked = ctx.mask_assistant_messages_after(1, "[REDACTED]")
        redacted_msg = masked.full_messages[2]
        assert redacted_msg.content == "[REDACTED]"
        assert redacted_msg.tool_name is None
        assert redacted_msg.tool_call_id is None
        assert redacted_msg.metadata == {}

    def test_drop_trailing_assistant_messages(self):
        ctx = self._make_ctx()
        # Original ends with assistant at index 4
        assert ctx.full_messages[-1].role == MessageRole.ASSISTANT
        trimmed = ctx.drop_trailing_assistant_messages()
        # Should now end with the tool message at index 3
        assert trimmed.full_messages[-1].role == MessageRole.TOOL
        assert len(trimmed.full_messages) == 4

    def test_drop_trailing_no_assistant_at_end(self):
        msgs = [
            Message(role=MessageRole.USER, content="Go"),
            Message(role=MessageRole.TOOL, content="data", tool_name="t"),
        ]
        ctx = build_structured_context(msgs, frozenset({"t"}))
        trimmed = ctx.drop_trailing_assistant_messages()
        assert len(trimmed.full_messages) == 2

    def test_drop_trailing_multiple_assistants(self):
        msgs = [
            Message(role=MessageRole.USER, content="Go"),
            Message(role=MessageRole.TOOL, content="data", tool_name="t"),
            Message(role=MessageRole.ASSISTANT, content="Step 1"),
            Message(role=MessageRole.ASSISTANT, content="Step 2"),
        ]
        ctx = build_structured_context(msgs, frozenset({"t"}))
        trimmed = ctx.drop_trailing_assistant_messages()
        assert len(trimmed.full_messages) == 2
        assert trimmed.full_messages[-1].role == MessageRole.TOOL
