"""CausalArmor exception hierarchy."""


class CausalArmorError(Exception):
    """Base exception for all CausalArmor errors."""


class AttributionError(CausalArmorError):
    """Proxy model scoring failed during LOO attribution."""


class SanitizationError(CausalArmorError):
    """Sanitizer model failed to clean untrusted content."""


class ProviderError(CausalArmorError):
    """LLM provider connection or API error."""


class ConfigurationError(CausalArmorError):
    """Invalid CausalArmor configuration."""


class ContextError(CausalArmorError):
    """Malformed context (e.g., missing user request)."""
