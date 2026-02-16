"""Anthropic provider implementations.

Requires the ``anthropic`` optional dependency: ``pip install causal-armor[anthropic]``
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

from causal_armor.exceptions import ProviderError
from causal_armor.prompts.sanitization import (
    SANITIZATION_SYSTEM_PROMPT,
    SANITIZATION_USER_TEMPLATE,
)
from causal_armor.types import Message, MessageRole, ToolCall

try:
    import anthropic
except ImportError as exc:
    raise ImportError(
        "Anthropic provider requires the 'anthropic' package. "
        "Install it with: pip install causal-armor[anthropic]"
    ) from exc


def _to_anthropic_messages(
    messages: Sequence[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert CausalArmor messages to Anthropic format.

    Returns (system_prompt, messages) since Anthropic takes system as a
    separate parameter.
    """
    system_prompt: str | None = None
    result: list[dict[str, Any]] = []

    for m in messages:
        if m.role is MessageRole.SYSTEM:
            # Anthropic uses a single system param; concatenate if multiple
            if system_prompt is None:
                system_prompt = m.content
            else:
                system_prompt += "\n" + m.content
        elif m.role is MessageRole.USER:
            result.append({"role": "user", "content": m.content})
        elif m.role is MessageRole.ASSISTANT:
            result.append({"role": "assistant", "content": m.content})
        elif m.role is MessageRole.TOOL:
            result.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id or "",
                        "content": m.content,
                    }
                ],
            })

    return system_prompt, result


class AnthropicActionProvider:
    """Action provider using Anthropic's messages API.

    Parameters
    ----------
    model:
        Anthropic model name (e.g. ``"claude-sonnet-4-5-20250929"``).
    tools:
        Anthropic tool definitions.
    client:
        Optional pre-configured ``AsyncAnthropic`` client.
    max_tokens:
        Maximum tokens for the response.
    """

    def __init__(
        self,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        client: anthropic.AsyncAnthropic | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self._model = model or os.environ.get("CAUSAL_ARMOR_ACTION_MODEL", "claude-sonnet-4-5-20250929")
        self._tools = tools
        self._client = client or anthropic.AsyncAnthropic()
        self._max_tokens = max_tokens

    async def generate(self, messages: Sequence[Message]) -> tuple[str, list[ToolCall]]:
        system_prompt, api_messages = _to_anthropic_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": self._max_tokens,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if self._tools:
            kwargs["tools"] = self._tools

        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.AnthropicError as exc:
            raise ProviderError(f"Anthropic generation failed: {exc}") from exc

        raw_text = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                raw_text += block.text
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else {}
                tool_calls.append(
                    ToolCall(
                        name=block.name,
                        arguments=args,
                        raw_text=json.dumps(args),
                    )
                )

        return raw_text, tool_calls


class AnthropicSanitizerProvider:
    """Sanitizer provider using Anthropic's messages API.

    Parameters
    ----------
    model:
        Anthropic model name (e.g. ``"claude-haiku-4-5-20251001"``).
    client:
        Optional pre-configured ``AsyncAnthropic`` client.
    """

    def __init__(
        self,
        model: str | None = None,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._model = model or os.environ.get("CAUSAL_ARMOR_SANITIZER_MODEL", "claude-haiku-4-5-20251001")
        self._client = client or anthropic.AsyncAnthropic()

    async def sanitize(self, user_request: str, tool_name: str, untrusted_content: str) -> str:
        user_msg = SANITIZATION_USER_TEMPLATE.format(
            user_request=user_request,
            tool_name=tool_name,
            untrusted_content=untrusted_content,
        )

        try:
            response = await self._client.messages.create(
                model=self._model,
                system=SANITIZATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                max_tokens=4096,
            )
        except anthropic.AnthropicError as exc:
            raise ProviderError(f"Anthropic sanitization failed: {exc}") from exc

        return response.content[0].text if response.content else ""
