"""LiteLLM provider implementations.

Unified interface that works with any LiteLLM-supported model backend.
Requires the ``litellm`` optional dependency: ``pip install causal-armor[litellm]``
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
    import litellm
except ImportError as exc:
    raise ImportError(
        "LiteLLM provider requires the 'litellm' package. "
        "Install it with: pip install causal-armor[litellm]"
    ) from exc


def _to_litellm_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert CausalArmor messages to LiteLLM chat format.

    TOOL-role messages are converted to plain user messages with a
    ``[Tool: name]`` prefix so they remain valid without a preceding
    assistant ``tool_calls`` entry (which may have been redacted during
    the defense pipeline).  Consecutive same-role messages are merged
    to satisfy the API constraint against adjacent duplicates.
    """
    raw: list[dict[str, Any]] = []
    for m in messages:
        if m.role is MessageRole.TOOL:
            label = f"[Tool: {m.tool_name}] " if m.tool_name else ""
            raw.append({"role": "user", "content": f"{label}{m.content}"})
        else:
            raw.append({"role": m.role.value, "content": m.content})

    # Merge consecutive same-role messages
    merged: list[dict[str, Any]] = []
    for msg in raw:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(msg)
    return merged


class LiteLLMActionProvider:
    """Action provider using LiteLLM's unified completion API.

    Parameters
    ----------
    model:
        Any LiteLLM-supported model string (e.g. ``"gpt-4o"``,
        ``"anthropic/claude-sonnet-4-5-20250929"``, ``"gemini/gemini-2.5-flash"``).
    tools:
        OpenAI-format tool definitions.
    """

    def __init__(
        self,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self._model = model or os.environ.get("CAUSAL_ARMOR_ACTION_MODEL", "gpt-4o")
        self._tools = tools

    async def generate(self, messages: Sequence[Message]) -> tuple[str, list[ToolCall]]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _to_litellm_messages(messages),
        }
        if self._tools:
            kwargs["tools"] = self._tools

        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as exc:
            raise ProviderError(f"LiteLLM generation failed: {exc}") from exc

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


class LiteLLMSanitizerProvider:
    """Sanitizer provider using LiteLLM's unified completion API.

    Parameters
    ----------
    model:
        Any LiteLLM-supported model string.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.environ.get(
            "CAUSAL_ARMOR_SANITIZER_MODEL", "gpt-4o-mini"
        )

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
            response = await litellm.acompletion(
                model=self._model,
                messages=[
                    {"role": "system", "content": SANITIZATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
        except Exception as exc:
            raise ProviderError(f"LiteLLM sanitization failed: {exc}") from exc

        return response.choices[0].message.content or ""


class LiteLLMProxyProvider:
    """Proxy provider using LiteLLM for log-prob scoring.

    Uses LiteLLM's completion (not chat) endpoint with logprobs enabled.
    Requires a model that supports the completions API with logprobs.

    Parameters
    ----------
    model:
        LiteLLM model string pointing to a completions-capable model.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.environ.get(
            "CAUSAL_ARMOR_PROXY_MODEL", "text-davinci-003"
        )

    async def log_prob(self, messages: Sequence[Message], action_text: str) -> float:
        # Build a flat prompt from messages
        parts: list[str] = []
        for m in messages:
            parts.append(f"{m.role.value}: {m.content}")
        prompt = "\n".join(parts) + f"\nassistant: {action_text}"

        try:
            response = await litellm.atext_completion(
                model=self._model,
                prompt=prompt,
                max_tokens=0,
                echo=True,
                logprobs=1,
            )
        except Exception as exc:
            raise ProviderError(f"LiteLLM log_prob scoring failed: {exc}") from exc

        try:
            logprobs_data = response.choices[0].logprobs
            token_logprobs = logprobs_data["token_logprobs"]
            text_offsets = logprobs_data.get("text_offset", [])
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Unexpected LiteLLM logprobs response: {exc}") from exc

        # Sum log-probs for action tokens only
        prompt_without_action = "\n".join(parts) + "\nassistant: "
        prompt_char_len = len(prompt_without_action)

        total_lp = 0.0
        for i, offset in enumerate(text_offsets):
            if offset >= prompt_char_len and token_logprobs[i] is not None:
                total_lp += token_logprobs[i]

        return total_lp
