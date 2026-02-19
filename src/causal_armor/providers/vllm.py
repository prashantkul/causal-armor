"""vLLM proxy provider — the paper's recommended proxy model backend.

Uses httpx to call vLLM's OpenAI-compatible ``/v1/completions`` endpoint
with ``logprobs=True, echo=True, max_tokens=0`` to score action
log-probabilities without generating new tokens.

Supports **batched scoring**: all LOO variants are sent in a single
HTTP request (one ``prompt`` list), matching the paper's Algorithm 2.

No SDK dependency — just httpx (core dep).
"""

from __future__ import annotations

import json
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


def _normalize_action_text(action_text: str) -> str:
    """Convert action text to a natural-language form for log-prob scoring.

    Instruction-tuned proxy models (e.g. Gemma) produce poor log-probs
    when asked to score raw JSON tool calls because they don't naturally
    emit JSON blobs.  Converting to a natural-language function call
    format (``I will call func(arg=val)``) yields much more meaningful
    LOO deltas.

    If ``action_text`` is already in natural-language form (not JSON),
    it is returned unchanged.
    """
    try:
        parsed = json.loads(action_text)
    except (json.JSONDecodeError, TypeError):
        return action_text

    if not isinstance(parsed, dict):
        return action_text

    name = parsed.get("name")
    arguments = parsed.get("arguments")
    if name and isinstance(arguments, dict):
        args_str = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        return f"I will call {name}({args_str})"

    return action_text


def _extract_action_logprob(
    logprobs_data: dict, prompt_char_len: int
) -> float:
    """Sum log-probs for action tokens given a single choice's logprobs."""
    token_logprobs: list[float | None] = logprobs_data["token_logprobs"]
    text_offsets: list[int] = logprobs_data.get("text_offset", [])

    total_lp = 0.0
    for i, offset in enumerate(text_offsets):
        lp = token_logprobs[i]
        if offset >= prompt_char_len and lp is not None:
            total_lp += lp
    return total_lp


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
        self._base_url = (
            base_url
            or os.environ.get("CAUSAL_ARMOR_PROXY_BASE_URL", "http://localhost:8000")
        ).rstrip("/")
        self._model = model or os.environ.get(
            "CAUSAL_ARMOR_PROXY_MODEL", "google/gemma-3-12b-it"
        )
        self._client = httpx.AsyncClient(timeout=timeout)

    async def log_prob(self, messages: Sequence[Message], action_text: str) -> float:
        """Score action log-probability via vLLM completions endpoint.

        Constructs ``prompt + action_text``, calls with ``echo=True,
        max_tokens=0, logprobs=1`` and sums log-probs over the action
        tokens only.
        """
        prompt = _messages_to_prompt(messages)
        action_text = _normalize_action_text(action_text)
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
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"Unexpected vLLM response structure: {exc}") from exc

        prompt_char_len = len(prompt) + len("\nAssistant: ")
        return _extract_action_logprob(logprobs_data, prompt_char_len)

    async def log_prob_batch(
        self,
        variants: Sequence[tuple[tuple[Message, ...], str]],
    ) -> list[float]:
        """Score multiple (messages, action_text) pairs in a single vLLM request.

        Uses vLLM's batch prompt support: sending a list of prompts in
        one ``/v1/completions`` call.  This matches the paper's Algorithm 2
        which sends all LOO ablation variants as a single batch.

        Parameters
        ----------
        variants:
            List of (messages, action_text) pairs to score.

        Returns
        -------
        list[float]
            Log-probabilities in the same order as the input variants.
        """
        if not variants:
            return []

        prompts: list[str] = []
        prompt_char_lens: list[int] = []

        for messages, action_text in variants:
            prompt = _messages_to_prompt(messages)
            action_text_norm = _normalize_action_text(action_text)
            full_prompt = f"{prompt}\nAssistant: {action_text_norm}"
            prompts.append(full_prompt)
            prompt_char_lens.append(len(prompt) + len("\nAssistant: "))

        payload = {
            "model": self._model,
            "prompt": prompts,
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
            raise ProviderError(f"vLLM batch request failed: {exc}") from exc

        data = resp.json()
        choices = data.get("choices", [])

        if len(choices) != len(variants):
            raise ProviderError(
                f"vLLM batch returned {len(choices)} choices, expected {len(variants)}"
            )

        # vLLM returns choices ordered by the prompt index
        sorted_choices = sorted(choices, key=lambda c: c.get("index", 0))
        results: list[float] = []
        for i, choice in enumerate(sorted_choices):
            try:
                logprobs_data = choice["logprobs"]
            except KeyError as exc:
                raise ProviderError(
                    f"Unexpected vLLM batch response structure: {exc}"
                ) from exc
            results.append(_extract_action_logprob(logprobs_data, prompt_char_lens[i]))

        return results

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> VLLMProxyProvider:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
