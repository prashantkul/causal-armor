"""LOO causal attribution (Algorithm 2, lines 4-10).

Measures how much each context component (user request, untrusted spans)
contributes to the agent's proposed action by ablating each one and
scoring the log-probability change via the proxy model.
"""

from __future__ import annotations

import asyncio

from causal_armor.context import StructuredContext
from causal_armor.providers import ProxyProvider
from causal_armor.types import AttributionResult, ToolCall


async def compute_attribution(
    ctx: StructuredContext,
    action: ToolCall,
    proxy: ProxyProvider,
    *,
    max_concurrent: int | None = None,
) -> AttributionResult:
    """Batched leave-one-out attribution over the structured context.

    For each context component (user request U, each untrusted span S_i),
    we ablate it from C_t and measure the drop in log P(action | context).

    Parameters
    ----------
    ctx:
        Structured context C_t.
    action:
        The agent's proposed tool-call action Y_t.
    proxy:
        Proxy model for log-prob scoring.
    max_concurrent:
        Optional cap on concurrent proxy calls. ``None`` means no limit.

    Returns
    -------
    AttributionResult
        Full attribution with per-component deltas.
    """
    action_text = action.raw_text
    # Rough token count approximation (word-level split, per paper)
    action_token_count = max(len(action_text.split()), 1)

    sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    async def _score(messages: tuple, label: str) -> tuple[str, float]:
        if sem:
            async with sem:
                lp = await proxy.log_prob(messages, action_text)
        else:
            lp = await proxy.log_prob(messages, action_text)
        return label, lp

    # Build all ablation variants
    tasks: list[asyncio.Task[tuple[str, float]]] = []

    # Base: full context
    tasks.append(asyncio.ensure_future(_score(ctx.full_messages, "_base")))

    # User-ablated
    tasks.append(
        asyncio.ensure_future(_score(ctx.messages_without_user_request(), "_user"))
    )

    # Span-ablated (one per untrusted span)
    for span_id in ctx.span_ids:
        tasks.append(
            asyncio.ensure_future(_score(ctx.messages_without_span(span_id), span_id))
        )

    results = await asyncio.gather(*tasks)
    scores: dict[str, float] = dict(results)

    base_lp = scores["_base"]
    user_ablated_lp = scores["_user"]

    # Deltas: positive means this component *increased* action likelihood
    delta_user = base_lp - user_ablated_lp
    delta_user_normalized = delta_user / action_token_count

    span_attributions: dict[str, float] = {}
    span_attributions_normalized: dict[str, float] = {}
    ablated_logprobs: dict[str, float] = {}

    for span_id in ctx.span_ids:
        span_lp = scores[span_id]
        ablated_logprobs[span_id] = span_lp
        delta_s = base_lp - span_lp
        span_attributions[span_id] = delta_s
        span_attributions_normalized[span_id] = delta_s / action_token_count

    return AttributionResult(
        delta_user=delta_user,
        delta_user_normalized=delta_user_normalized,
        span_attributions=span_attributions,
        span_attributions_normalized=span_attributions_normalized,
        base_logprob=base_lp,
        ablated_logprobs=ablated_logprobs,
        user_ablated_logprob=user_ablated_lp,
        action_token_count=action_token_count,
    )
