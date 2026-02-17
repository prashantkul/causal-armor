"""Structured context decomposition for CausalArmor.

Decomposes the conversation context C_t into its causal components:
user request (U), system messages (H_t), and untrusted spans (S_t).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from causal_armor.exceptions import ContextError
from causal_armor.types import Message, MessageRole, UntrustedSpan


@dataclass(frozen=True, slots=True)
class StructuredContext:
    """Decomposed conversation context C_t = (U, H_t, S_t).

    Parameters
    ----------
    user_request:
        The user's task message U.
    system_messages:
        System-role messages H_t.
    untrusted_spans:
        Mapping span_id -> UntrustedSpan for tool-result messages from
        untrusted tools (S_t).
    full_messages:
        The complete ordered message sequence C_t.
    """

    user_request: Message
    system_messages: tuple[Message, ...]
    untrusted_spans: dict[str, UntrustedSpan]
    full_messages: tuple[Message, ...]

    # -- Algorithm 2 helpers -------------------------------------------------

    @property
    def has_untrusted_spans(self) -> bool:
        """Line 2 guard: whether any untrusted spans exist."""
        return bool(self.untrusted_spans)

    @property
    def span_ids(self) -> frozenset[str]:
        """All untrusted span identifiers."""
        return frozenset(self.untrusted_spans)

    def messages_without_user_request(self) -> tuple[Message, ...]:
        """Line 5: C_t \\ U — context with the user request removed."""
        return tuple(m for m in self.full_messages if m is not self.user_request)

    def messages_without_span(self, span_id: str) -> tuple[Message, ...]:
        """Line 5: C_t \\ S — context with a specific span removed."""
        span = self.untrusted_spans.get(span_id)
        if span is None:
            raise ContextError(f"Unknown span_id: {span_id!r}")
        idx = span.context_index
        return tuple(m for i, m in enumerate(self.full_messages) if i != idx)

    def replace_span_content(self, span_id: str, new_content: str) -> StructuredContext:
        """Line 15: REPLACE — return a new context with one span's content swapped."""
        span = self.untrusted_spans.get(span_id)
        if span is None:
            raise ContextError(f"Unknown span_id: {span_id!r}")

        idx = span.context_index
        old_msg = self.full_messages[idx]
        new_msg = Message(
            role=old_msg.role,
            content=new_content,
            tool_name=old_msg.tool_name,
            tool_call_id=old_msg.tool_call_id,
            metadata=old_msg.metadata,
        )
        new_messages = list(self.full_messages)
        new_messages[idx] = new_msg

        new_span = UntrustedSpan(
            span_id=span.span_id,
            content=new_content,
            source_tool_name=span.source_tool_name,
            context_index=span.context_index,
            turn_index=span.turn_index,
        )
        new_spans = {**self.untrusted_spans, span_id: new_span}

        # Update user_request if it happened to be at that index (unlikely but safe)
        new_user = new_msg if old_msg is self.user_request else self.user_request

        return replace(
            self,
            user_request=new_user,
            full_messages=tuple(new_messages),
            untrusted_spans=new_spans,
        )

    def mask_assistant_messages_after(
        self, context_index: int, redaction_text: str
    ) -> StructuredContext:
        """Lines 21-23: CoT mask — redact assistant messages after the given index."""
        new_messages = list(self.full_messages)
        for i in range(context_index + 1, len(new_messages)):
            msg = new_messages[i]
            if msg.role is MessageRole.ASSISTANT:
                new_messages[i] = Message(
                    role=MessageRole.ASSISTANT,
                    content=redaction_text,
                    tool_name=msg.tool_name,
                    tool_call_id=msg.tool_call_id,
                    metadata=msg.metadata,
                )
        return replace(self, full_messages=tuple(new_messages))


def build_structured_context(
    messages: Sequence[Message],
    untrusted_tool_names: frozenset[str],
    *,
    turn_index: int = 0,
) -> StructuredContext:
    """Build a :class:`StructuredContext` from a flat message sequence.

    Parameters
    ----------
    messages:
        Ordered conversation messages C_t.
    untrusted_tool_names:
        Tool names whose results should be treated as untrusted spans.
    turn_index:
        Current agentic turn number (used in span metadata).

    Raises
    ------
    ContextError
        If no user-role message is found.
    """
    msgs = tuple(messages)

    # Pick last user-role message as U (most recent task in multi-turn)
    user_request: Message | None = None
    for m in reversed(msgs):
        if m.role is MessageRole.USER:
            user_request = m
            break
    if user_request is None:
        raise ContextError("No user-role message found in context")

    system_messages = tuple(m for m in msgs if m.role is MessageRole.SYSTEM)

    untrusted_spans: dict[str, UntrustedSpan] = {}
    for i, m in enumerate(msgs):
        if (
            m.role is MessageRole.TOOL
            and m.tool_name
            and m.tool_name in untrusted_tool_names
        ):
            span_id = f"{m.tool_name}:{i}"
            untrusted_spans[span_id] = UntrustedSpan(
                span_id=span_id,
                content=m.content,
                source_tool_name=m.tool_name,
                context_index=i,
                turn_index=turn_index,
            )

    return StructuredContext(
        user_request=user_request,
        system_messages=system_messages,
        untrusted_spans=untrusted_spans,
        full_messages=msgs,
    )
