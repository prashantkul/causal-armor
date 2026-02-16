# Paper Models Reference

Models used in the CausalArmor paper (arXiv:2602.07918) and their roles.

## Model Roles in CausalArmor

```mermaid
flowchart LR
    classDef agent fill:#2196F3,color:#fff,stroke:#1565C0
    classDef proxy fill:#FF9800,color:#fff,stroke:#E65100
    classDef san fill:#4CAF50,color:#fff,stroke:#2E7D32
    classDef baseline fill:#607D8B,color:#fff,stroke:#37474F
    classDef output fill:#E1F5FE,color:#333,stroke:#1565C0
    classDef output2 fill:#FFF3E0,color:#333,stroke:#E65100
    classDef output3 fill:#E8F5E9,color:#333,stroke:#2E7D32
    classDef output4 fill:#ECEFF1,color:#333,stroke:#37474F
    AG["Agent Backbone (M_agent)"]:::agent
    PR["Proxy Model (M_proxy)"]:::proxy
    SN["Sanitizer (M_san)"]:::san
    BL["Classifier Baselines"]:::baseline
    AG -->|"Proposes actions"| ACT["Tool-call actions Y_t"]:::output
    PR -->|"Scores log-probs"| LOO["LOO Attribution"]:::output2
    SN -->|"Cleans flagged spans"| DEF["Sanitized content"]:::output3
    BL -->|"Binary detection"| CLS["Injection / Clean"]:::output4
```

```mermaid
flowchart TD
    classDef agent fill:#2196F3,color:#fff,stroke:#1565C0
    classDef proxy fill:#FF9800,color:#fff,stroke:#E65100
    classDef san fill:#4CAF50,color:#fff,stroke:#2E7D32
    classDef mw fill:#9C27B0,color:#fff,stroke:#6A1B9A
    classDef result fill:#00897B,color:#fff,stroke:#004D40
    subgraph Agents
        G25["Gemini-2.5-Flash"]:::agent
        G3P["Gemini-3-Pro"]:::agent
        C4S["Claude-4.0-Sonnet"]:::agent
    end
    subgraph Proxies
        GEM12["Gemma-3-12B-IT (default)"]:::proxy
        GEM["Gemma-3: 1B to 27B"]:::proxy
        QW["Qwen3: 1.7B to 32B"]:::proxy
        MIN["Ministral: 3B to 14B"]:::proxy
    end
    subgraph Sanitizer
        SAN["Gemini-2.5-Flash"]:::san
    end
    Agents -->|"proposes action"| MW["CausalArmor Guard"]:::mw
    Proxies -->|"scores log-probs"| MW
    MW -->|"if attack"| Sanitizer
    Sanitizer --> RES["Safe action"]:::result
```

## Agent Backbones (M_agent)

The main LLM agents being defended — CausalArmor sits in front of these.

| Model | Role |
|-------|------|
| **Gemini-2.5-Flash** | Default agent backbone across all experiments |
| **Gemini-3-Pro** | Stronger backbone for generalizability testing |
| **Claude-4.0-Sonnet** | Cross-family generalizability testing |

## Proxy Models (M_proxy)

Lightweight models for LOO attribution scoring, served via vLLM. The proxy model never generates text — it only scores log-probabilities.

### Default
| Model | Notes |
|-------|-------|
| **Gemma-3-12B-IT** | Default proxy, best Pareto trade-off (latency vs. security) |

### Proxy size ablation (Gemma-3 family)
| Model | Size | Detection reliability |
|-------|------|---------------------|
| Gemma-3-1B | 1B | Unreliable |
| Gemma-3-4B | 4B | Partial |
| **Gemma-3-12B-IT** | 12B | Near-perfect |
| Gemma-3-27B | 27B | Near-perfect |

### Proxy family ablation
| Family | Sizes tested |
|--------|-------------|
| **Qwen3** | 1.7B, 4B, 8B, 14B, 32B |
| **Ministral** | 3B, 8B, 14B |

### Key finding

> Models above **8B** reliably detect IPI attacks. **12B+** achieve near-perfect detection. Same-family proxies (Gemma for Gemini agents) are most **Pareto-optimal** for latency vs. security.

## Sanitizer (M_san)

Rewrites flagged untrusted spans to remove injected instructions while preserving factual data.

| Model | Notes |
|-------|-------|
| **Gemini-2.5-Flash** | Fixed across all experiments |

The sanitizer is the most expensive step (full generation call), which is why CausalArmor only triggers it when an attack is detected — keeping the common case (benign traffic) fast.

## Trained Classifier Baselines

Used for comparison against CausalArmor's causal attribution approach.

| Model | Type | Source |
|-------|------|--------|
| **DeBERTa-v3-base-prompt-injection-v2** | Binary injection detector | [HuggingFace](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) |
| **PIGuard** | Injection guardrail classifier | [HuggingFace](https://huggingface.co/leolee99/PIGuard) |

## Other Roles

| Model | Role |
|-------|------|
| **Gemini-2.5-Flash** | User Simulator in DoomArena's conversational setting |
| **Gemini-2.5-Flash** | Adversarial LLM attacker in DoomArena |
| **Gemini-3-Pro** | Generated the two novel attack templates (`task_dependency`, `tool_output_hijack`) |

## Mapping to CausalArmor Code

```mermaid
flowchart LR
    classDef proto fill:#9C27B0,color:#fff,stroke:#6A1B9A
    classDef agent fill:#2196F3,color:#fff,stroke:#1565C0
    classDef proxy fill:#FF9800,color:#fff,stroke:#E65100
    classDef san fill:#4CAF50,color:#fff,stroke:#2E7D32
    AP["ActionProvider"]:::proto
    PP["ProxyProvider"]:::proto
    SP["SanitizerProvider"]:::proto
    AP --- OAI["OpenAI"]:::agent
    AP --- ANT["Anthropic"]:::agent
    AP --- GEM["Gemini"]:::agent
    AP --- LIT1["LiteLLM"]:::agent
    PP --- VLLM["vLLM + Gemma-12B"]:::proxy
    PP --- LIT2["LiteLLM"]:::proxy
    SP --- OAI2["OpenAI"]:::san
    SP --- ANT2["Anthropic"]:::san
    SP --- GEM2["Gemini"]:::san
    SP --- LIT3["LiteLLM"]:::san
```

| Paper role | Protocol | Recommended provider | Config |
|-----------|----------|---------------------|--------|
| M_agent | `ActionProvider` | OpenAI / Anthropic / Gemini / LiteLLM | `OPENAI_ACTION_MODEL`, etc. |
| M_proxy | `ProxyProvider` | `VLLMProxyProvider` with Gemma-3-12B-IT | `VLLM_BASE_URL`, `VLLM_MODEL` |
| M_san | `SanitizerProvider` | OpenAI / Anthropic / Gemini / LiteLLM | `OPENAI_SANITIZER_MODEL`, etc. |
