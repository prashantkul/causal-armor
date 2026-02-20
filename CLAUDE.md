# CLAUDE.md — CausalArmor

## Important: Open source library

This is a published open source library on PyPI. Exercise extra caution:

- **Do not introduce errors** — always run the full check suite before committing
- **Always run `make lint` and `make format` before committing** (or equivalently `.venv/bin/ruff check src/ tests/` and `.venv/bin/ruff format src/ tests/`)
- **Run tests** (`make test`) to verify nothing is broken
- Prefer `make check` (lint + typecheck + test) for a complete pre-commit validation
- Be careful with public API changes — they affect downstream users

## Project overview

CausalArmor is a Python library implementing indirect prompt injection guardrails via causal attribution. It follows a two-phase architecture (scoring + regeneration) based on Algorithm 2 from [arXiv:2602.07918](https://arxiv.org/abs/2602.07918).

**Core pipeline**: `build_structured_context` → `compute_attribution` → `detect_dominant_spans` → `defend` (sanitize + mask CoT + regenerate)

**Entry point for users**: `CausalArmorMiddleware.guard()` orchestrates the full pipeline.

## Repository layout

```
src/causal_armor/
  __init__.py          # Public API re-exports
  types.py             # Frozen dataclasses: Message, ToolCall, AttributionResult, etc.
  config.py            # CausalArmorConfig dataclass
  context.py           # StructuredContext, build_structured_context
  attribution.py       # LOO causal attribution (compute_attribution)
  detection.py         # Dominance-shift detection (Eq. 5)
  defense.py           # Sanitize, mask CoT, regenerate action
  middleware.py         # CausalArmorMiddleware (top-level orchestrator)
  exceptions.py        # Exception hierarchy
  providers/           # LLM provider implementations
    _protocols.py      # Protocol classes: ProxyProvider, ActionProvider, SanitizerProvider
    openai.py, anthropic.py, gemini.py, litellm.py, vllm.py
  prompts/
    sanitization.py    # Sanitization prompt templates
tests/
  conftest.py          # Shared fixtures and mock providers
  test_context.py, test_attribution.py, test_detection.py, test_defense.py, test_middleware.py
```

## Development commands

```bash
# All commands use the venv at .venv/
# Or use Makefile targets:

make lint        # ruff check src/ tests/
make format      # ruff format src/ tests/
make typecheck   # mypy src/causal_armor/
make test        # pytest tests/ -v
make check       # lint + typecheck + test
```

## Code style and conventions

- **Linter**: ruff with rules `E, F, W, I, UP, B, SIM, RUF` (see `pyproject.toml`)
- **Line length**: 88 characters
- **Target Python**: 3.11+ (`from __future__ import annotations` in all files)
- **Formatter**: ruff format (double quotes, space indent)
- **Type checking**: mypy strict mode
- **Quote style**: double quotes
- **Imports**: isort via ruff, `causal_armor` as known first-party

### Key lint rules to watch

- `RUF005`: Use `[*a, *b]` instead of `a + b` for list concatenation
- `B905`: Use `strict=True` in `zip()` calls
- `RUF002`: No Unicode confusables (use ASCII `-` not `−` in docstrings)
- `RUF059`: Prefix unused variables with `_`
- `E501`: Keep lines under 88 chars

## Architecture principles

- **Immutable data**: All core types (`Message`, `ToolCall`, `StructuredContext`, result types) are frozen dataclasses. Mutations return new instances.
- **Protocol-based providers**: `ProxyProvider`, `ActionProvider`, `SanitizerProvider` are typing Protocols — any object with the right method signatures works.
- **Async-first**: All provider calls and the middleware are async.
- **No fallback to original action on failed regeneration**: When regeneration produces no tool call, the defense returns a stripped action (same name, empty arguments) — never the original attacker-controlled action.

## Testing

- **Framework**: pytest + pytest-asyncio
- **Async mode**: `asyncio_mode = "auto"` (no need for `@pytest.mark.asyncio` on fixtures, only on test methods)
- **Mock providers** are in `tests/conftest.py` — `MockProxyAttack`, `MockProxyBenign`, `MockSanitizer`, `MockActionProvider`
- **CI matrix**: Python 3.11, 3.12, 3.13 on ubuntu-latest

## Git workflow

- **Main branch**: `master`
- **PRs target**: `master`
- **CI**: Runs on push to master and all PRs — lint, format check, typecheck, test
- Always run `make check` (or at minimum `make lint`) before committing
