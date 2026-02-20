"""OpenAI provider implementations.

Requires the ``openai`` optional dependency: ``pip install causal-armor[openai]``
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
    import openai
except ImportError as exc:
    raise ImportError(
        "OpenAI provider requires the 'openai' package. "
        "Install it with: pip install causal-armor[openai]"
    ) from exc


def _to_openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert CausalArmor messages to OpenAI chat format."""
    result: list[dict[str, Any]] = []
    for m in messages:
        msg: dict[str, Any] = {"role": m.role.value, "content": m.content}
        if m.role is MessageRole.TOOL:
            msg["tool_call_id"] = m.tool_call_id or ""
        result.append(msg)
    return result


class OpenAIActionProvider:
    """Action provider using OpenAI's chat completions API.

    Parameters
    ----------
    model:
        OpenAI model name (e.g. ``"gpt-4o"``).
    tools:
        OpenAI tool definitions for function calling.
    client:
        Optional pre-configured ``AsyncOpenAI`` client.
    """

    def __init__(
        self,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        self._model = model or os.environ.get("CAUSAL_ARMOR_ACTION_MODEL", "gpt-4o")
        self._tools = tools
        self._client = client or openai.AsyncOpenAI()

    async def generate(self, messages: Sequence[Message]) -> tuple[str, list[ToolCall]]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _to_openai_messages(messages),
        }
        if self._tools:
            kwargs["tools"] = self._tools

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except openai.OpenAIError as exc:
            raise ProviderError(f"OpenAI generation failed: {exc}") from exc

        choice = response.choices[0]
        raw_text = choice.message.content or ""

        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=args,
                        raw_text=tc.function.arguments,
                    )
                )

        return raw_text, tool_calls


class OpenAISanitizerProvider:
    """Sanitizer provider using OpenAI's chat completions API.

    Parameters
    ----------
    model:
        OpenAI model name (e.g. ``"gpt-4o-mini"``).
    client:
        Optional pre-configured ``AsyncOpenAI`` client.
    """

    def __init__(
        self,
        model: str | None = None,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        self._model = model or os.environ.get(
            "CAUSAL_ARMOR_SANITIZER_MODEL", "gpt-4o-mini"
        )
        self._client = client or openai.AsyncOpenAI()

    async def sanitize(
        self,
        user_request: str,
        tool_name: str,
        untrusted_content: str,
        proposed_action: str = "",
    ) -> str:
        user_msg = SANITIZATION_USER_TEMPLATE.format(
            user_request=user_request,
            tool_name=tool_name,
            untrusted_content=untrusted_content,
            proposed_action=proposed_action,
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SANITIZATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
        except openai.OpenAIError as exc:
            raise ProviderError(f"OpenAI sanitization failed: {exc}") from exc

        return response.choices[0].message.content or ""
