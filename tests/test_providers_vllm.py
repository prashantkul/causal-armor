"""Tests for VLLMProxyProvider with respx mocking."""

from __future__ import annotations

import pytest
import respx
import httpx

from causal_armor import Message, MessageRole
from causal_armor.providers.vllm import VLLMProxyProvider


VLLM_URL = "http://localhost:8000"


@pytest.fixture()
def vllm_provider():
    return VLLMProxyProvider(base_url=VLLM_URL, model="test-model")


def _make_vllm_response(token_logprobs: list[float | None], text_offset: list[int]):
    """Build a mock vLLM completions response."""
    return {
        "choices": [
            {
                "logprobs": {
                    "token_logprobs": token_logprobs,
                    "text_offset": text_offset,
                    "tokens": [f"tok_{i}" for i in range(len(token_logprobs))],
                }
            }
        ]
    }


@pytest.mark.asyncio
@respx.mock
async def test_log_prob_sums_action_tokens(vllm_provider):
    """Log probs for action tokens are summed correctly."""
    messages = [
        Message(role=MessageRole.USER, content="Hello"),
    ]
    action_text = "do_thing"

    # The prompt will be "User: Hello\nAssistant: do_thing"
    # "User: Hello\nAssistant: " is the prompt prefix
    prompt_prefix = "User: Hello\nAssistant: "
    prefix_len = len(prompt_prefix)

    # Mock: 3 prompt tokens, 2 action tokens
    respx.post(f"{VLLM_URL}/v1/completions").mock(
        return_value=httpx.Response(
            200,
            json=_make_vllm_response(
                token_logprobs=[None, -1.0, -0.5, -0.3, -0.2],
                text_offset=[0, 5, 10, prefix_len, prefix_len + 4],
            ),
        )
    )

    result = await vllm_provider.log_prob(messages, action_text)
    # Should sum only action tokens: -0.3 + -0.2 = -0.5
    assert abs(result - (-0.5)) < 1e-6


@pytest.mark.asyncio
@respx.mock
async def test_log_prob_handles_none_logprobs(vllm_provider):
    """None logprobs (first token) are skipped."""
    messages = [Message(role=MessageRole.USER, content="Hi")]
    prompt_prefix = "User: Hi\nAssistant: "
    prefix_len = len(prompt_prefix)

    respx.post(f"{VLLM_URL}/v1/completions").mock(
        return_value=httpx.Response(
            200,
            json=_make_vllm_response(
                token_logprobs=[None, -1.0, None, -0.5],
                text_offset=[0, 5, prefix_len, prefix_len + 3],
            ),
        )
    )

    result = await vllm_provider.log_prob(messages, "act")
    # Only the non-None action token: -0.5
    assert abs(result - (-0.5)) < 1e-6


@pytest.mark.asyncio
@respx.mock
async def test_log_prob_raises_on_http_error(vllm_provider):
    """HTTP errors raise ProviderError."""
    from causal_armor.exceptions import ProviderError

    messages = [Message(role=MessageRole.USER, content="Hi")]

    respx.post(f"{VLLM_URL}/v1/completions").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    with pytest.raises(ProviderError, match="vLLM request failed"):
        await vllm_provider.log_prob(messages, "act")


@pytest.mark.asyncio
@respx.mock
async def test_log_prob_raises_on_bad_response(vllm_provider):
    """Malformed response raises ProviderError."""
    from causal_armor.exceptions import ProviderError

    messages = [Message(role=MessageRole.USER, content="Hi")]

    respx.post(f"{VLLM_URL}/v1/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )

    with pytest.raises(ProviderError, match="Unexpected vLLM response"):
        await vllm_provider.log_prob(messages, "act")


@pytest.mark.asyncio
async def test_context_manager():
    """Provider works as async context manager."""
    async with VLLMProxyProvider(base_url=VLLM_URL) as provider:
        assert provider._base_url == VLLM_URL


@pytest.mark.asyncio
@respx.mock
async def test_log_prob_all_prompt_tokens(vllm_provider):
    """If no tokens fall in action range, return 0.0."""
    messages = [Message(role=MessageRole.USER, content="Hi")]
    prompt_prefix = "User: Hi\nAssistant: "
    prefix_len = len(prompt_prefix)

    # All offsets are below prefix_len -> no action tokens
    respx.post(f"{VLLM_URL}/v1/completions").mock(
        return_value=httpx.Response(
            200,
            json=_make_vllm_response(
                token_logprobs=[None, -1.0, -0.5],
                text_offset=[0, 3, 6],
            ),
        )
    )

    result = await vllm_provider.log_prob(messages, "act")
    assert result == 0.0
