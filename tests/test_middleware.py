"""Tests for CausalArmorMiddleware end-to-end."""

from __future__ import annotations

import pytest
from conftest import MockActionProvider, MockProxyAttack, MockProxyBenign, MockSanitizer

from causal_armor import (
    CausalArmorConfig,
    CausalArmorMiddleware,
    Message,
    MessageRole,
    ToolCall,
)

UNTRUSTED = frozenset({"web_search"})


@pytest.fixture()
def attack_middleware():
    return CausalArmorMiddleware(
        action_provider=MockActionProvider(),
        proxy_provider=MockProxyAttack(),
        sanitizer_provider=MockSanitizer(),
        config=CausalArmorConfig(margin_tau=0.0),
    )


@pytest.fixture()
def benign_middleware():
    return CausalArmorMiddleware(
        action_provider=MockActionProvider(),
        proxy_provider=MockProxyBenign(),
        sanitizer_provider=MockSanitizer(),
        config=CausalArmorConfig(margin_tau=0.0),
    )


class TestMiddlewareGuard:
    @pytest.mark.asyncio
    async def test_attack_detected_and_defended(
        self, attack_middleware, attack_messages
    ):
        action = ToolCall(
            name="send_money", arguments={}, raw_text="send_money amount=10000"
        )
        result = await attack_middleware.guard(
            attack_messages, action, untrusted_tool_names=UNTRUSTED
        )
        assert result.was_defended
        assert result.final_action.name == "book_flight"
        assert result.original_action.name == "send_money"
        assert result.detection is not None
        assert result.detection.is_attack_detected

    @pytest.mark.asyncio
    async def test_benign_passthrough(self, benign_middleware, benign_messages):
        action = ToolCall(
            name="book_flight", arguments={}, raw_text="book_flight AA123"
        )
        result = await benign_middleware.guard(
            benign_messages, action, untrusted_tool_names=UNTRUSTED
        )
        assert not result.was_defended
        assert result.final_action is action

    @pytest.mark.asyncio
    async def test_no_untrusted_tools_passthrough(
        self, attack_middleware, attack_messages
    ):
        action = ToolCall(name="send_money", arguments={}, raw_text="send_money")
        result = await attack_middleware.guard(
            attack_messages, action, untrusted_tool_names=frozenset()
        )
        assert not result.was_defended
        assert result.detection is None

    @pytest.mark.asyncio
    async def test_non_privileged_tool_skips_attribution(self, attack_messages):
        """When privileged_tools is set, non-privileged actions skip defense."""
        config = CausalArmorConfig(privileged_tools=frozenset({"send_money"}))
        mw = CausalArmorMiddleware(
            MockActionProvider(), MockProxyAttack(), MockSanitizer(), config
        )
        # book_flight is NOT in privileged_tools, so it should skip defense
        action = ToolCall(name="book_flight", arguments={}, raw_text="book_flight")
        result = await mw.guard(attack_messages, action, untrusted_tool_names=UNTRUSTED)
        assert not result.was_defended
        assert result.detection is None

    @pytest.mark.asyncio
    async def test_privileged_tool_gets_defended(self, attack_messages):
        """When privileged_tools is set, privileged actions ARE defended."""
        config = CausalArmorConfig(privileged_tools=frozenset({"send_money"}))
        mw = CausalArmorMiddleware(
            MockActionProvider(), MockProxyAttack(), MockSanitizer(), config
        )
        action = ToolCall(name="send_money", arguments={}, raw_text="send_money amount=10000")
        result = await mw.guard(attack_messages, action, untrusted_tool_names=UNTRUSTED)
        assert result.was_defended
        assert result.detection is not None
        assert result.detection.is_attack_detected

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with CausalArmorMiddleware(
            MockActionProvider(), MockProxyAttack(), MockSanitizer()
        ) as mw:
            assert mw.config.margin_tau == 0.0

    @pytest.mark.asyncio
    async def test_user_only_context(self, attack_middleware):
        """Messages with only a user message and no tools."""
        msgs = [Message(role=MessageRole.USER, content="Hello")]
        action = ToolCall(name="greet", arguments={}, raw_text="greet")
        result = await attack_middleware.guard(
            msgs, action, untrusted_tool_names=UNTRUSTED
        )
        assert not result.was_defended

    @pytest.mark.asyncio
    async def test_config_property(self, attack_middleware):
        assert attack_middleware.config.margin_tau == 0.0
