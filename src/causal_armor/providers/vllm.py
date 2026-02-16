"""vLLM proxy provider — the paper's recommended proxy model backend.

Uses httpx to call vLLM's OpenAI-compatible ``/v1/completions`` endpoint
with ``logprobs=True, echo=True, max_tokens=0`` to score action
log-probabilities without generating new tokens.

No SDK dependency — just httpx (core dep).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import httpx

from causal_armor.exceptions import ProviderError
from causal_armor.types import Message, MessageRole


def _messages_to_prompt(messages: Sequence[Message]) -> str:
    """Convert structured messages into a plain-text prompt for the completions API."""
    parts: list[str] = []
    for m in messages:
        if m.role is MessageRole.SYSTEM:
            parts.append(f"System: {m.content}")
        elif m.role is MessageRole.USER:
            parts.append(f"User: {m.content}")
        elif m.role is MessageRole.ASSISTANT:
            parts.append(f"Assistant: {m.content}")
        elif m.role is MessageRole.TOOL:
            label = f"Tool({m.tool_name})" if m.tool_name else "Tool"
            parts.append(f"{label}: {m.content}")
    return "\n".join(parts)


class VLLMProxyProvider:
    """Proxy provider backed by a vLLM server's OpenAI-compatible API.

    Parameters
    ----------
    base_url:
        vLLM server URL (e.g. ``"http://localhost:8000"``).
    model:
        Model name as served by vLLM (e.g. ``"google/gemma-3-12b-it"``).
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = (base_url or os.environ.get("CAUSAL_ARMOR_PROXY_BASE_URL", "http://localhost:8000")).rstrip("/")
        self._model = model or os.environ.get("CAUSAL_ARMOR_PROXY_MODEL", "google/gemma-3-12b-it")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def log_prob(self, messages: Sequence[Message], action_text: str) -> float:
        """Score action log-probability via vLLM completions endpoint.

        Constructs ``prompt + action_text``, calls with ``echo=True,
        max_tokens=0, logprobs=1`` and sums log-probs over the action
        tokens only.
        """
        prompt = _messages_to_prompt(messages)
        full_prompt = f"{prompt}\nAssistant: {action_text}"

        payload = {
            "model": self._model,
            "prompt": full_prompt,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
        }

        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/completions",
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"vLLM request failed: {exc}") from exc

        data = resp.json()

        try:
            logprobs_data = data["choices"][0]["logprobs"]
            token_logprobs: list[float | None] = logprobs_data["token_logprobs"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"Unexpected vLLM response structure: {exc}") from exc

        # Count action tokens: tokenize action_text by splitting the full
        # token list. The prompt tokens come first; we want only the tail
        # tokens corresponding to action_text. Since we don't know the exact
        # prompt token count, we use the offset: vLLM returns text_offset
        # which tells us where each token starts in the full prompt string.
        text_offsets: list[int] = logprobs_data.get("text_offset", [])
        prompt_char_len = len(prompt) + len("\nAssistant: ")

        total_lp = 0.0
        for i, offset in enumerate(text_offsets):
            if offset >= prompt_char_len and token_logprobs[i] is not None:
                total_lp += token_logprobs[i]

        return total_lp

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> VLLMProxyProvider:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
