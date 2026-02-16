"""CausalArmor — Indirect Prompt Injection guardrails via causal attribution."""

from dotenv import load_dotenv

load_dotenv()

from causal_armor.attribution import compute_attribution
from causal_armor.config import CausalArmorConfig
from causal_armor.context import StructuredContext, build_structured_context
from causal_armor.defense import defend, mask_cot_after_detection, sanitize_flagged_spans
from causal_armor.detection import detect_dominant_spans
from causal_armor.exceptions import (
    AttributionError,
    CausalArmorError,
    ConfigurationError,
    ContextError,
    ProviderError,
    SanitizationError,
)
from causal_armor.middleware import CausalArmorMiddleware
from causal_armor.providers import ActionProvider, ProxyProvider, SanitizerProvider
from causal_armor.types import (
    AttributionResult,
    DefenseResult,
    DetectionResult,
    Message,
    MessageRole,
    ToolCall,
    UntrustedSpan,
)

__all__ = [
    # Algorithm
    "compute_attribution",
    "detect_dominant_spans",
    # Config
    "CausalArmorConfig",
    # Defense
    "defend",
    "mask_cot_after_detection",
    "sanitize_flagged_spans",
    # Context
    "StructuredContext",
    "build_structured_context",
    # Middleware
    "CausalArmorMiddleware",
    # Exceptions
    "AttributionError",
    "CausalArmorError",
    "ConfigurationError",
    "ContextError",
    "ProviderError",
    "SanitizationError",
    # Providers (protocols)
    "ActionProvider",
    "ProxyProvider",
    "SanitizerProvider",
    # Types
    "AttributionResult",
    "DefenseResult",
    "DetectionResult",
    "Message",
    "MessageRole",
    "ToolCall",
    "UntrustedSpan",
]
