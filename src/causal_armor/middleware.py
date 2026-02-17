"""CausalArmor middleware — single entry point for the full pipeline.

Orchestrates Algorithm 1: context building, attribution, detection,
and defense in one ``guard()`` call.
"""

from __future__ import annotations

from collections.abc import Sequence

from causal_armor.attribution import compute_attribution
from causal_armor.config import CausalArmorConfig
from causal_armor.context import StructuredContext, build_structured_context
from causal_armor.defense import defend
from causal_armor.detection import detect_dominant_spans
from causal_armor.providers import ActionProvider, ProxyProvider, SanitizerProvider
from causal_armor.types import DefenseResult, Message, ToolCall


class CausalArmorMiddleware:
    """Main middleware orchestrating the CausalArmor defense pipeline.

    Parameters
    ----------
    action_provider:
        M_agent — generates tool-call actions.
    proxy_provider:
        M_proxy — scores log-probabilities for LOO attribution.
    sanitizer_provider:
        M_san — rewrites untrusted content.
    config:
        Pipeline configuration (thresholds, toggles, batch size).
    """

    def __init__(
        self,
        action_provider: ActionProvider,
        proxy_provider: ProxyProvider,
        sanitizer_provider: SanitizerProvider,
        config: CausalArmorConfig = CausalArmorConfig(),
    ) -> None:
        self._action_provider = action_provider
        self._proxy_provider = proxy_provider
        self._sanitizer_provider = sanitizer_provider
        self._config = config

    @property
    def config(self) -> CausalArmorConfig:
        return self._config

    async def guard(
        self,
        messages: Sequence[Message],
        action: ToolCall,
        *,
        untrusted_tool_names: frozenset[str] | None = None,
        turn_index: int = 0,
    ) -> DefenseResult:
        """Run the full CausalArmor pipeline on a proposed action.

        Parameters
        ----------
        messages:
            The conversation context C_t.
        action:
            The agent's proposed tool-call action Y_t.
        untrusted_tool_names:
            Tool names whose results are untrusted. If ``None``, all
            non-privileged tool results are treated as untrusted.
        turn_index:
            Current agentic turn number.

        Returns
        -------
        DefenseResult
            Contains the original action, final (possibly regenerated)
            action, and full detection/defense metadata.
        """
        # Step 1: Check if action targets a privileged tool (skip attribution)
        if self._config.privileged_tools and action.name in self._config.privileged_tools:
            return DefenseResult(
                original_action=action,
                final_action=action,
                was_defended=False,
                detection=None,
            )

        # Step 2: Build structured context
        effective_untrusted = untrusted_tool_names if untrusted_tool_names is not None else frozenset()
        ctx = build_structured_context(
            messages,
            untrusted_tool_names=effective_untrusted,
            turn_index=turn_index,
        )

        # Step 3: If no untrusted spans, pass through
        if not ctx.has_untrusted_spans:
            return DefenseResult(
                original_action=action,
                final_action=action,
                was_defended=False,
                detection=None,
            )

        # Step 4: Compute LOO attribution
        attribution = await compute_attribution(
            ctx,
            action,
            self._proxy_provider,
            max_concurrent=self._config.max_loo_batch_size,
            mask_cot_for_scoring=self._config.mask_cot_for_scoring,
        )

        # Step 5: Detect dominant spans
        detection = detect_dominant_spans(attribution, self._config.margin_tau)

        # Step 6: Defend if attack detected
        return await defend(
            ctx,
            action,
            detection,
            self._sanitizer_provider,
            self._action_provider,
            self._config,
        )

    async def close(self) -> None:
        """Close any provider resources (e.g. HTTP clients)."""
        for provider in (self._action_provider, self._proxy_provider, self._sanitizer_provider):
            close_fn = getattr(provider, "close", None)
            if close_fn and callable(close_fn):
                await close_fn()

    async def __aenter__(self) -> CausalArmorMiddleware:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
