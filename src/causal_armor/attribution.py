"""LOO causal attribution (Algorithm 2, lines 4-10).

Measures how much each context component (user request, untrusted spans)
contributes to the agent's proposed action by ablating each one and
scoring the log-probability change via the proxy model.

When the proxy supports ``log_prob_batch``, all ablation variants are
scored in a single request (matching the paper's batched approach).
Otherwise falls back to concurrent individual calls.
"""

from __future__ import annotations

import asyncio

from causal_armor.context import StructuredContext
from causal_armor.providers import ProxyProvider
from causal_armor.types import AttributionResult, Message, ToolCall


async def compute_attribution(
    ctx: StructuredContext,
    action: ToolCall,
    proxy: ProxyProvider,
    *,
    max_concurrent: int | None = None,
    mask_cot_for_scoring: bool = True,
) -> AttributionResult:
    """Batched leave-one-out attribution over the structured context.

    For each context component (user request U, each untrusted span S_i),
    we ablate it from C_t and measure the drop in log P(action | context).

    When the proxy supports ``log_prob_batch``, all variants are sent in
    a single request.  Otherwise, falls back to concurrent individual calls.

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
        Only used in the non-batch fallback path.
    mask_cot_for_scoring:
        If ``True``, mask assistant messages after the first untrusted
        span before LOO scoring.  This prevents the agent's own poisoned
        reasoning from masking the true causal signal in multi-turn
        conversations.

    Returns
    -------
    AttributionResult
        Full attribution with per-component deltas.
    """
    action_text = action.raw_text
    # Approximate token count: subword tokenizers average ~4 chars/token.
    # This is closer to true |Y_t| than a word-count split.
    action_token_count = max(len(action_text) // 4, 1)

    # Optionally mask CoT before LOO scoring.  In multi-turn conversations
    # the agent's reasoning may propagate injected instructions (e.g. the
    # AI says "I need to call send_money").  If left in context, ablating
    # the tool result has little effect because the AI's text still drives
    # the action.  Masking assistant messages after the first untrusted
    # span isolates the true causal influence of external inputs.
    scoring_ctx = ctx
    if mask_cot_for_scoring and ctx.has_untrusted_spans:
        k_min = min(span.context_index for span in ctx.untrusted_spans.values())
        scoring_ctx = ctx.mask_assistant_messages_after(k_min, "[Reasoning redacted]")

    # Build ordered list of (label, messages) for all ablation variants
    span_id_list = sorted(scoring_ctx.span_ids)
    labels: list[str] = ["_base", "_user", *span_id_list]
    message_variants: list[tuple[Message, ...]] = [
        scoring_ctx.full_messages,
        scoring_ctx.messages_without_user_request(),
    ] + [scoring_ctx.messages_without_span(sid) for sid in span_id_list]

    # Try batched scoring first (single HTTP request)
    batch_fn = getattr(proxy, "log_prob_batch", None)
    if batch_fn is not None:
        batch_input = [(msgs, action_text) for msgs in message_variants]
        log_probs = await batch_fn(batch_input)
        scores: dict[str, float] = dict(zip(labels, log_probs, strict=True))
    else:
        # Fallback: concurrent individual calls
        sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

        async def _score(
            messages: tuple[Message, ...], label: str
        ) -> tuple[str, float]:
            if sem:
                async with sem:
                    lp = await proxy.log_prob(messages, action_text)
            else:
                lp = await proxy.log_prob(messages, action_text)
            return label, lp

        tasks = [
            asyncio.ensure_future(_score(msgs, lbl))
            for lbl, msgs in zip(labels, message_variants, strict=True)
        ]
        results = await asyncio.gather(*tasks)
        scores = dict(results)

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
