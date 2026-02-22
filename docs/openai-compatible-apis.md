# Using OpenRouter and OpenAI-Compatible APIs

CausalArmor's OpenAI providers work with any service that exposes an
[OpenAI-compatible chat completions API](https://platform.openai.com/docs/api-reference/chat).
This includes **OpenRouter**, **Azure OpenAI**, **Together AI**, **Anyscale**,
**Fireworks AI**, local servers like **vLLM** and **Ollama**, and many others.

## Quick start with OpenRouter

[OpenRouter](https://openrouter.ai/) provides a unified API gateway to models
from OpenAI, Google, Anthropic, Meta, and others -- all through a single
OpenAI-compatible endpoint.

```python
from causal_armor.providers.openai import (
    OpenAIActionProvider,
    OpenAISanitizerProvider,
)
from causal_armor.providers.vllm import VLLMProxyProvider
from causal_armor import CausalArmorMiddleware, CausalArmorConfig

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_KEY = "sk-or-..."  # your OpenRouter API key

action = OpenAIActionProvider(
    model="google/gemini-2.5-flash",
    tools=your_tool_definitions,  # OpenAI function-calling format
    base_url=OPENROUTER_BASE,
    api_key=OPENROUTER_KEY,
)

sanitizer = OpenAISanitizerProvider(
    model="google/gemini-2.5-flash",
    base_url=OPENROUTER_BASE,
    api_key=OPENROUTER_KEY,
)

middleware = CausalArmorMiddleware(
    action_provider=action,
    proxy_provider=VLLMProxyProvider(),
    sanitizer_provider=sanitizer,
    config=CausalArmorConfig.from_env(),
)
```

### Available models on OpenRouter

OpenRouter model names are prefixed with the provider
(e.g. `google/gemini-2.5-flash`, `openai/gpt-4o`, `anthropic/claude-sonnet-4-20250514`).
See the [OpenRouter models page](https://openrouter.ai/models) for the full list.

## Using the `base_url` and `api_key` parameters

Both `OpenAIActionProvider` and `OpenAISanitizerProvider` accept optional
keyword-only `base_url` and `api_key` parameters:

```python
provider = OpenAIActionProvider(
    model="my-model",
    tools=[...],
    base_url="https://my-service.example.com/v1",
    api_key="my-secret-key",
)
```

**Precedence**: explicit `client` parameter > `base_url`/`api_key` parameters
> environment variables (`OPENAI_BASE_URL`, `OPENAI_API_KEY`) > SDK defaults.

If you pass a pre-configured `AsyncOpenAI` client via the `client` parameter,
`base_url` and `api_key` are ignored:

```python
import openai

# Full control via a custom client
custom_client = openai.AsyncOpenAI(
    base_url="https://my-service.example.com/v1",
    api_key="my-key",
    timeout=60.0,
    max_retries=5,
)

provider = OpenAIActionProvider(
    model="my-model",
    client=custom_client,  # base_url/api_key ignored when client is set
)
```

## Provider-specific examples

### Azure OpenAI

Azure uses a different client class, so pass a pre-configured client:

```python
from openai import AsyncAzureOpenAI

azure_client = AsyncAzureOpenAI(
    api_key="your-azure-key",
    api_version="2024-02-01",
    azure_endpoint="https://your-resource.openai.azure.com",
)

action = OpenAIActionProvider(
    model="your-deployment-name",
    tools=[...],
    client=azure_client,
)
```

### Together AI

```python
action = OpenAIActionProvider(
    model="meta-llama/Llama-3-70b-chat-hf",
    tools=[...],
    base_url="https://api.together.xyz/v1",
    api_key="your-together-key",
)
```

### Local vLLM / Ollama

For a local OpenAI-compatible server (no API key needed):

```python
action = OpenAIActionProvider(
    model="my-local-model",
    tools=[...],
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # some servers require a non-empty value
)
```

## Environment variables

You can also configure the endpoint via environment variables supported by the
OpenAI SDK. When no `base_url` or `client` is passed, the SDK reads:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Default API key |
| `OPENAI_BASE_URL` | Default base URL |

This means you can switch to OpenRouter globally without any code changes:

```bash
export OPENAI_API_KEY="sk-or-your-openrouter-key"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

## Tool format requirements

OpenAI-compatible APIs expect tool definitions in the
[OpenAI function-calling format](https://platform.openai.com/docs/guides/function-calling):

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    }
]
```

If your tools are in a different format (e.g. Gemini's `FunctionDeclaration`),
you'll need to convert them to this format before passing them to the provider.
