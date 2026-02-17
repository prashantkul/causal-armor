"""CausalArmor — Indirect Prompt Injection guardrails via causal attribution."""

from dotenv import load_dotenv

load_dotenv()

from causal_armor.attribution import compute_attribution  # noqa: E402
from causal_armor.config import CausalArmorConfig  # noqa: E402
from causal_armor.context import (  # noqa: E402
    StructuredContext,
    build_structured_context,
)
from causal_armor.defense import (  # noqa: E402
    defend,
    mask_cot_after_detection,
    sanitize_flagged_spans,
)
from causal_armor.detection import detect_dominant_spans  # noqa: E402
from causal_armor.exceptions import (  # noqa: E402
    AttributionError,
    CausalArmorError,
    ConfigurationError,
    ContextError,
    ProviderError,
    SanitizationError,
)
from causal_armor.middleware import CausalArmorMiddleware  # noqa: E402
from causal_armor.providers import (  # noqa: E402
    ActionProvider,
    ProxyProvider,
    SanitizerProvider,
)
from causal_armor.types import (  # noqa: E402
    AttributionResult,
    DefenseResult,
    DetectionResult,
    Message,
    MessageRole,
    ToolCall,
    UntrustedSpan,
)

__all__ = [
    "ActionProvider",
    "AttributionError",
    "AttributionResult",
    "CausalArmorConfig",
    "CausalArmorError",
    "CausalArmorMiddleware",
    "ConfigurationError",
    "ContextError",
    "DefenseResult",
    "DetectionResult",
    "Message",
    "MessageRole",
    "ProviderError",
    "ProxyProvider",
    "SanitizationError",
    "SanitizerProvider",
    "StructuredContext",
    "ToolCall",
    "UntrustedSpan",
    "build_structured_context",
    "compute_attribution",
    "defend",
    "detect_dominant_spans",
    "mask_cot_after_detection",
    "sanitize_flagged_spans",
]
