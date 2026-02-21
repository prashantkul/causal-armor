"""Google Gemini provider implementations.

Requires the ``google-genai`` optional dependency: ``pip install causal-armor[gemini]``
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
    from google import genai
    from google.genai import types as genai_types
except ImportError as exc:
    raise ImportError(
        "Gemini provider requires the 'google-genai' package. "
        "Install it with: pip install causal-armor[gemini]"
    ) from exc


def _to_gemini_contents(
    messages: Sequence[Message],
) -> tuple[str | None, list[genai_types.Content]]:
    """Convert CausalArmor messages to Gemini content format.

    Returns (system_instruction, contents).

    TOOL-role messages are represented as plain user messages with a
    ``[Tool: name]`` prefix.  Consecutive same-role contents are merged
    so the API never sees adjacent entries with the same role.
    """
    system_instruction: str | None = None
    raw: list[genai_types.Content] = []

    for m in messages:
        if m.role is MessageRole.SYSTEM:
            if system_instruction is None:
                system_instruction = m.content
            else:
                system_instruction += "\n" + m.content
        elif m.role is MessageRole.USER:
            raw.append(
                genai_types.Content(
                    role="user", parts=[genai_types.Part(text=m.content)]
                )
            )
        elif m.role is MessageRole.ASSISTANT:
            raw.append(
                genai_types.Content(
                    role="model", parts=[genai_types.Part(text=m.content)]
                )
            )
        elif m.role is MessageRole.TOOL:
            label = f"[Tool: {m.tool_name}] " if m.tool_name else ""
            raw.append(
                genai_types.Content(
                    role="user", parts=[genai_types.Part(text=f"{label}{m.content}")]
                )
            )

    # Merge consecutive same-role contents
    merged: list[genai_types.Content] = []
    for c in raw:
        prev_parts = merged[-1].parts if merged else None
        curr_parts = c.parts
        if merged and merged[-1].role == c.role and prev_parts and curr_parts:
            prev_text = prev_parts[0].text or ""
            curr_text = curr_parts[0].text or ""
            merged[-1] = genai_types.Content(
                role=c.role,
                parts=[genai_types.Part(text=f"{prev_text}\n{curr_text}")],
            )
        else:
            merged.append(c)

    return system_instruction, merged


class GeminiActionProvider:
    """Action provider using Google's Gemini API.

    Parameters
    ----------
    model:
        Gemini model name (e.g. ``"gemini-2.5-flash"``).
    tools:
        Gemini tool declarations.
    client:
        Optional pre-configured ``genai.Client``.
    """

    def __init__(
        self,
        model: str | None = None,
        tools: list[Any] | None = None,
        client: genai.Client | None = None,
    ) -> None:
        self._model = model or os.environ.get(
            "CAUSAL_ARMOR_ACTION_MODEL", "gemini-2.5-flash"
        )
        self._tools = tools
        self._client = client or genai.Client()

    async def generate(self, messages: Sequence[Message]) -> tuple[str, list[ToolCall]]:
        system_instruction, contents = _to_gemini_contents(messages)

        config: dict[str, Any] = {}
        if system_instruction:
            config["system_instruction"] = system_instruction
        if self._tools:
            config["tools"] = self._tools

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=genai_types.GenerateContentConfig(**config) if config else None,
            )
        except Exception as exc:
            raise ProviderError(f"Gemini generation failed: {exc}") from exc

        raw_text = response.text or ""
        tool_calls: list[ToolCall] = []

        if (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    fc = part.function_call
                    if fc.name is None:
                        continue
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append(
                        ToolCall(
                            name=fc.name,
                            arguments=args,
                            raw_text=json.dumps(args),
                        )
                    )

        return raw_text, tool_calls


class GeminiSanitizerProvider:
    """Sanitizer provider using Google's Gemini API.

    Parameters
    ----------
    model:
        Gemini model name (e.g. ``"gemini-2.5-flash"``).
    client:
        Optional pre-configured ``genai.Client``.
    """

    def __init__(
        self,
        model: str | None = None,
        client: genai.Client | None = None,
    ) -> None:
        self._model = model or os.environ.get(
            "CAUSAL_ARMOR_SANITIZER_MODEL", "gemini-2.5-flash"
        )
        self._client = client or genai.Client()

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
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=[
                    genai_types.Content(
                        role="user", parts=[genai_types.Part(text=user_msg)]
                    )
                ],
                config=genai_types.GenerateContentConfig(
                    system_instruction=SANITIZATION_SYSTEM_PROMPT,
                ),
            )
        except Exception as exc:
            raise ProviderError(f"Gemini sanitization failed: {exc}") from exc

        return response.text or ""
