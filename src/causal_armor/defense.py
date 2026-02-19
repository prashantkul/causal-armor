"""Defense layer (Algorithm 2, lines 13-27).

Selective sanitization of flagged spans, retroactive CoT masking, and
action regeneration through the agent model.
"""

from __future__ import annotations

from causal_armor.config import CausalArmorConfig
from causal_armor.context import StructuredContext
from causal_armor.providers import ActionProvider, SanitizerProvider
from causal_armor.types import DefenseResult, DetectionResult, ToolCall


async def sanitize_flagged_spans(
    ctx: StructuredContext,
    detection: DetectionResult,
    sanitizer: SanitizerProvider,
) -> tuple[StructuredContext, dict[str, str]]:
    """Sanitize content of **all** untrusted spans when an attack is detected.

    Although only *flagged* spans dominate the blocked action, injection
    text may reside in any untrusted span.  Sanitizing every untrusted
    span prevents the regeneration model from being re-influenced by
    attacker-controlled content that lives in an unflagged tool result.

    Parameters
    ----------
    ctx:
        Current structured context.
    detection:
        Detection result (used to confirm an attack was detected).
    sanitizer:
        Sanitizer provider to clean span content.

    Returns
    -------
    tuple[StructuredContext, dict[str, str]]
        (modified context, {span_id: sanitized_content} map)
    """
    sanitized_map: dict[str, str] = {}
    modified_ctx = ctx

    for span_id, span in ctx.untrusted_spans.items():
        sanitized_content = await sanitizer.sanitize(
            user_request=ctx.user_request.content,
            tool_name=span.source_tool_name,
            untrusted_content=span.content,
        )
        modified_ctx = modified_ctx.replace_span_content(span_id, sanitized_content)
        sanitized_map[span_id] = sanitized_content

    return modified_ctx, sanitized_map


def mask_cot_after_detection(
    ctx: StructuredContext,
    detection: DetectionResult,
    redaction_text: str,
) -> StructuredContext:
    """Redact assistant chain-of-thought messages after the earliest flagged span.

    Prevents the agent from being re-influenced by its own compromised
    reasoning during regeneration (Algorithm 2, lines 21-23).

    Parameters
    ----------
    ctx:
        Current structured context.
    detection:
        Detection result identifying flagged spans.
    redaction_text:
        Replacement text for redacted assistant messages.

    Returns
    -------
    StructuredContext
        Context with assistant messages masked after k_min.
    """
    if not detection.flagged_spans:
        return ctx

    # Find earliest flagged span position
    k_min = min(
        ctx.untrusted_spans[span_id].context_index
        for span_id in detection.flagged_spans
    )

    return ctx.mask_assistant_messages_after(k_min, redaction_text)


async def defend(
    ctx: StructuredContext,
    action: ToolCall,
    detection: DetectionResult,
    sanitizer: SanitizerProvider,
    action_provider: ActionProvider,
    config: CausalArmorConfig,
) -> DefenseResult:
    """Full defense pipeline (Algorithm 2, lines 9-27).

    If an attack is detected:
    1. Sanitize flagged spans (if enabled)
    2. Mask chain-of-thought (if enabled)
    3. Regenerate the action with the cleaned context

    Parameters
    ----------
    ctx:
        Structured context C_t.
    action:
        The agent's original proposed action Y_t.
    detection:
        Detection result from the dominance-shift detector.
    sanitizer:
        Sanitizer provider for cleaning flagged spans.
    action_provider:
        Action provider for regenerating the action.
    config:
        CausalArmor configuration.

    Returns
    -------
    DefenseResult
        Complete defense output with original and final actions.
    """
    if not detection.is_attack_detected:
        return DefenseResult(
            original_action=action,
            final_action=action,
            was_defended=False,
            detection=detection,
        )

    modified_ctx = ctx
    sanitized_spans: dict[str, str] = {}
    cot_masked = False

    # Step 1: Sanitize flagged spans
    if config.enable_sanitization:
        modified_ctx, sanitized_spans = await sanitize_flagged_spans(
            modified_ctx, detection, sanitizer
        )

    # Step 2: Mask chain-of-thought
    if config.enable_cot_masking:
        modified_ctx = mask_cot_after_detection(
            modified_ctx, detection, config.cot_redaction_text
        )
        cot_masked = True

    # Step 3: Drop trailing assistant messages (blocked action proposal)
    modified_ctx = modified_ctx.drop_trailing_assistant_messages()

    # Step 4: Regenerate action with cleaned context
    _raw_text, tool_calls = await action_provider.generate(modified_ctx.full_messages)

    if tool_calls:
        final_action = tool_calls[0]
        regenerated = True
    else:
        # Regeneration produced no tool call — block entirely rather than
        # falling back to the original attacker-controlled action.
        final_action = ToolCall(
            name=action.name,
            arguments={},
            raw_text="",
        )
        regenerated = False

    return DefenseResult(
        original_action=action,
        final_action=final_action,
        was_defended=True,
        detection=detection,
        sanitized_spans=sanitized_spans,
        cot_messages_masked=cot_masked,
        regenerated=regenerated,
    )
