# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-06-15

### Added

- Core middleware (`CausalArmorMiddleware`) with single `guard()` entry point
- LOO causal attribution (Algorithm 2) for measuring per-component influence
- Dominance-shift detection (Eq. 5) with configurable `margin_tau` threshold
- Full defense pipeline: sanitization, CoT masking, and action regeneration
- Structured context decomposition into user request, history, and untrusted spans
- Provider protocol interfaces (`ActionProvider`, `ProxyProvider`, `SanitizerProvider`)
- vLLM proxy provider using OpenAI-compatible `/v1/completions` endpoint
- OpenAI, Anthropic, Google Gemini, and LiteLLM provider implementations
- `CausalArmorConfig` with TOML file, env var, and constructor-arg layered resolution
- `py.typed` marker for PEP 561 typed package support
- 56 tests covering all core modules and the vLLM provider
- MIT license

[0.1.0]: https://github.com/prashantkul/causal-armor/releases/tag/v0.1.0
