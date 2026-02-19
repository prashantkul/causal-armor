"""Provider protocol definitions for CausalArmor.

Three async protocols matching the paper's model roles:
- ActionProvider  (M_agent) — generates tool-call actions
- ProxyProvider   (M_proxy) — scores log-probabilities for LOO attribution
- SanitizerProvider (M_san) — rewrites untrusted content
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from causal_armor.types import Message, ToolCall


@runtime_checkable
class ActionProvider(Protocol):
    """M_agent: generates tool-call actions from a conversation context."""

    async def generate(self, messages: Sequence[Message]) -> tuple[str, list[ToolCall]]:
        """Generate an action from the given message context.

        Returns
        -------
        tuple[str, list[ToolCall]]
            (raw_response_text, parsed_tool_calls). The raw text is needed
            for log-prob scoring by the proxy model.
        """
        ...


@runtime_checkable
class ProxyProvider(Protocol):
    """M_proxy: scores action log-probabilities for LOO attribution."""

    async def log_prob(self, messages: Sequence[Message], action_text: str) -> float:
        """Compute log P(action_text | messages).

        Returns the total log-probability of the action text conditioned on
        the given message context. LOO batching is handled by the algorithm
        layer, not by this interface.
        """
        ...


@runtime_checkable
class SanitizerProvider(Protocol):
    """M_san: rewrites untrusted content to neutralise injections."""

    async def sanitize(
        self,
        user_request: str,
        tool_name: str,
        untrusted_content: str,
        proposed_action: str = "",
    ) -> str:
        """Sanitize untrusted tool-result content.

        Parameters match the slots in ``SANITIZATION_USER_TEMPLATE``:
        ``{user_request}``, ``{tool_name}``, ``{untrusted_content}``,
        ``{proposed_action}``.

        The proposed action Y_t gives the sanitizer context about what
        the agent was trying to do, enabling more targeted rewriting.

        Returns the cleaned text with injected instructions removed.
        """
        ...
