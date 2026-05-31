# SavVio — Future Enhancement Plan

**Last Updated:** April 2026  
**Authors:** SavVio Team  
**Status:** Planning / Pre-Implementation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Architecture Snapshot](#current-architecture-snapshot)
3. [Enhancement 1 — vLLM: Self-Hosted LLM Inference Engine](#enhancement-1--vllm-self-hosted-llm-inference-engine)
4. [Enhancement 2 — KV Caching: Inference Memory Optimization](#enhancement-2--kv-caching-inference-memory-optimization)
5. [Enhancement 3 — Speculative Decoding (Speculators)](#enhancement-3--speculative-decoding-speculators)
6. [Enhancement 4 — Mixture of Experts (MoE) Models](#enhancement-4--mixture-of-experts-moe-models)
7. [Enhancement 5 — llm-d: Distributed Inference Orchestration](#enhancement-5--llm-d-distributed-inference-orchestration)
8. [Enhancement 6 — NVIDIA NeMo Agent Toolkit](#enhancement-6--nvidia-nemo-agent-toolkit)
9. [Enhancement 7 — NVIDIA AI Blueprints](#enhancement-7--nvidia-ai-blueprints)
10. [Technology Comparison Matrix](#technology-comparison-matrix)
11. [Recommended Adoption Roadmap](#recommended-adoption-roadmap)
12. [Alternatives & Trade-offs](#alternatives--trade-offs)
13. [Open Questions](#open-questions)

---

## Executive Summary

This document evaluates seven technology areas for future integration into SavVio, tailored to its existing architecture (deterministic financial engine + XGBoost/LightGBM confidence layer + external LLM APIs for intent parsing and response generation).

### Priority Ranking

| Priority | Technology | Why |
|----------|-----------|-----|
| 🔴 **P0** | **vLLM** + **KV Caching** | Foundation for everything else — eliminates API costs (~90% reduction), cuts latency (~60%), enables data privacy |
| 🟡 **P1** | **Speculative Decoding**, **MoE Models**, **NeMo Agent Toolkit** | Performance and architecture upgrades once vLLM is live |
| 🟢 **P2-P3** | **NVIDIA Blueprints**, **llm-d** | Reference patterns (Blueprints) and future scaling (llm-d when on GKE) |

### Key Recommendations

1. **vLLM** → Deploy with `Qwen3-8B` (FP8 quantized). The existing `OpenAIProvider` can be repointed to localhost because vLLM exposes an OpenAI-compatible API
2. **KV Caching** → Free win — enable Automatic Prefix Caching in vLLM for the fixed system prompt
3. **Speculators** → Swap to `RedHatAI/Qwen3-8B-speculator.eagle3` for 2-3x token generation speedup (zero code changes)
4. **MoE** → Start dense; evaluate `Qwen3-30B-A3B` only if quality gaps appear
5. **NeMo Agent Toolkit** → Perfect fit for evolving SavVio from a 2-step pipeline to a multi-agent system with evaluation and safety middleware
6. **llm-d** → Skip until GKE migration and >50 QPS traffic
7. **Blueprints** → Study AI-Q and Retail Shopping Assistant as architecture references

Each technology is assessed in detail below for: **if** SavVio should use it, **where** it fits in the architecture, **how** to integrate it, and **what alternatives** exist. See [Open Questions](#open-questions) at the end for 6 decision points that will refine this roadmap.

---

## Current Architecture Snapshot

```
User Query
    │
    ▼
┌─── LLM Role 1: Intent Parser ───┐
│  External API (Gemini/OpenAI)    │  ← Per-token cost, network latency
└──────────────┬───────────────────┘
               │
               ▼
┌─── Product Resolver ────────────┐
│  pgvector + all-MiniLM-L6-v2   │  ← Self-hosted embeddings
└──────────────┬──────────────────┘
               │
               ▼
    [Deterministic Engine + XGBoost]   ← Self-hosted, Dockerized
               │
               ▼
┌─── LLM Role 2: Response Gen ───┐
│  External API (Gemini/OpenAI)    │  ← Per-token cost, network latency
└──────────────┬─────────────────┘
               │
               ▼
┌─── Guardrails (6 checks) ──────┐
│  G1-G6 Code-level validation    │  ← Self-hosted
└────────────────────────────────┘
```

**Current LLM Providers:** OpenRouter hub → Gemini 2.5 Flash (primary), OpenAI GPT-4.1, Claude 4.5 Sonnet  
**Current Infrastructure:** Docker Compose on GCP Cloud Run, PostgreSQL with pgvector, MLflow tracking  
**Current Pain Points:**
- Per-token API costs scale with user growth
- Network round-trip latency (~200-500ms TTFT for external APIs)
- No control over model weights, quantization, or context window behavior
- Guardrails are code-level only — no learned safety models

---

## Enhancement 1 — vLLM: Self-Hosted LLM Inference Engine

### What Is It?

[vLLM](https://docs.vllm.ai) is a high-throughput, memory-efficient inference engine for LLMs. It provides:
- **PagedAttention** for efficient KV cache memory management
- OpenAI-compatible API server (drop-in replacement for current providers)
- Continuous batching for high-throughput serving
- Support for 100+ model architectures (Llama, Mistral, Qwen, Gemma, etc.)
- Quantization support (FP8, INT8, INT4, GPTQ, AWQ, GGUF)
- LoRA adapter hot-swapping
- Speculative decoding, prefix caching, disaggregated prefilling

### Should SavVio Use It?

**Yes — this is the highest-priority enhancement.** vLLM is the foundation for most other enhancements listed here.

### Where in SavVio?

Replace the external LLM API calls in `llm/llm_provider.py` with a self-hosted vLLM server:

```
┌─────────────────────────────────────────────────┐
│  vLLM Server (OpenAI-compatible API)            │
│  Model: Qwen3-8B or Llama-3.1-8B (quantized)   │
│  Port: 8000                                     │
│  Features: PagedAttention, Prefix Caching       │
└─────────────────────┬───────────────────────────┘
                      │ localhost:8000/v1/chat/completions
                      ▼
         SavVio LLM Provider (OpenAI SDK)
```

### How to Integrate

1. **Add a `VLLMProvider` to `llm/llm_provider.py`** — Since vLLM exposes an OpenAI-compatible API, the existing `OpenAIProvider` stub can be repointed to `localhost:8000`.
2. **Choose a model** — For SavVio's financial advisory use case:
   - **Budget:** `Qwen3-8B` (quantized to FP8: ~4GB VRAM) — excellent multilingual, instruction-following
   - **Quality:** `Llama-3.1-8B-Instruct` or `Mistral-7B-Instruct-v0.3`
   - **Cost-optimized:** `Qwen3-4B` or `Phi-4-mini` (~2GB VRAM quantized)
3. **Dockerize vLLM** — Add a `vllm-server` service to `model_pipeline/docker-compose.yml`:
   ```yaml
   vllm-server:
     image: vllm/vllm-openai:latest
     command: ["--model", "Qwen/Qwen3-8B", "--quantization", "fp8", "--max-model-len", "4096"]
     deploy:
       resources:
         reservations:
           devices:
             - capabilities: [gpu]
     ports:
       - "8000:8000"
   ```
4. **Enable Prefix Caching** — SavVio uses a fixed system prompt (`system_prompt.py v1.0`). Prefix caching will cache the KV states for this prompt, reducing TTFT for every request.

### Key Metrics to Expect

| Metric | External API (Current) | vLLM Self-Hosted (Projected) |
|--------|----------------------|------------------------------|
| TTFT | 200-500ms | 50-150ms |
| Cost per 1K tokens | ~$0.10-0.60 | ~$0.002 (GPU amortized) |
| Max concurrent users | Rate-limited (RPM) | 50-100+ (single GPU) |
| Data privacy | Tokens sent externally | All data stays on-prem |

### Alternatives

| Alternative | Pros | Cons | Recommendation |
|-------------|------|------|----------------|
| **SGLang** | Comparable throughput, RadixAttention | Smaller community, less model support | Monitor — consider if prefix caching becomes critical |
| **TensorRT-LLM** | Best NVIDIA GPU optimization | Complex setup, NVIDIA-only | Consider for production NVIDIA deployments |
| **Ollama** | Easiest setup | Lower throughput, no production batching | Good for local dev only |
| **llama.cpp** | CPU-friendly, edge deployment | No continuous batching, lower throughput | Not suitable for production serving |

### Prerequisites

- GPU with ≥8GB VRAM (A10G, L4, or T4 on GCP)
- NVIDIA Container Toolkit installed
- ~10GB disk for model weights

---

## Enhancement 2 — KV Caching: Inference Memory Optimization

### What Is It?

[KV Caching](https://huggingface.co/blog/not-lain/kv-caching) stores the Key and Value tensors computed during the attention mechanism for previously generated tokens. Instead of recomputing attention over the entire sequence at each generation step, only the new token's Q/K/V are computed, and the cached K/V from prior tokens are reused.

### Should SavVio Use It?

**Yes — but indirectly.** KV Caching is not something you implement yourself; it's a feature built into inference engines like vLLM, which enables it by default.

### Where in SavVio?

KV caching benefits SavVio in two specific scenarios:

1. **System Prompt Prefix Caching** — SavVio's system prompt (`system_prompt.py`) is ~500 tokens and identical across all requests. With vLLM's Automatic Prefix Caching (APC), the KV states for these tokens are computed once and reused for every user query. This reduces TTFT by 30-50%.

2. **Multi-Turn Conversations** — If SavVio evolves from single-turn Q&A to multi-turn conversational sessions (e.g., "What about a cheaper laptop?" as a follow-up), KV caching will preserve the context from prior turns without recomputation.

### How to Enable

When running vLLM, prefix caching is enabled with:
```bash
vllm serve Qwen/Qwen3-8B --enable-prefix-caching
```

### Advanced: Quantized KV Cache

vLLM also supports **Quantized KV Cache** (FP8 KV), which reduces KV cache memory by 50% with minimal quality loss. This is critical for serving many concurrent users:
```bash
vllm serve Qwen/Qwen3-8B --kv-cache-dtype fp8_e5m2
```

### Advanced: Tiered KV Cache Offloading (via llm-d)

For very-high-traffic scenarios, llm-d enables **tiered KV cache offloading** — moving evicted KV entries from GPU memory → CPU memory → SSD → remote storage. This dramatically increases effective context window and cache hit rates.

### Alternatives

| Alternative | What It Does | When to Use |
|-------------|-------------|-------------|
| **Sliding Window Attention** | Limits attention to last N tokens | If conversations are very long and older context can be discarded |
| **Multi-Query Attention (MQA/GQA)** | Reduces KV cache size at the model architecture level | Already built into modern models (Llama 3, Qwen, Mistral) — no action needed |
| **LMCache** | External KV cache store for cross-request sharing | If running multiple vLLM replicas behind a load balancer |

### SavVio-Specific Recommendation

Enable APC in vLLM on day one. No code changes needed in SavVio itself — it's purely an infrastructure optimization. Quantized KV comes later when GPU memory becomes a bottleneck under concurrent load.

---

## Enhancement 3 — Speculative Decoding (Speculators)

### What Is It?

[Speculators](https://github.com/vllm-project/speculators) is a library for building and deploying **speculative decoding** algorithms. Speculative decoding uses a small, fast "draft" model (the speculator) to propose multiple tokens at once, then the larger "verifier" model validates them in a single forward pass. This is a **lossless** optimization — accepted tokens are guaranteed identical to what the main model would have generated.

RedHat AI publishes [pre-trained speculator models](https://huggingface.co/collections/RedHatAI/speculator-models) for popular base models (Llama 3, Qwen3, GPT-OSS).

### Should SavVio Use It?

**Yes — but as a Phase 2 optimization after vLLM is deployed.** Speculative decoding reduces **inter-token latency** (TPOT) by 2-3x, which directly improves the user experience for SavVio's conversational responses (typically 100-300 tokens).

### Where in SavVio?

This is a vLLM serving configuration — no SavVio code changes required:

```
┌─────────────────────────────────────────┐
│  vLLM Server                            │
│  Verifier: Qwen3-8B                     │
│  Speculator: RedHatAI/Qwen3-8B-spec..  │
│  Method: EAGLE-3                        │
│  Draft tokens per step: 3-5             │
└─────────────────────────────────────────┘
```

### How to Enable

Using a pre-trained speculator with vLLM:
```bash
vllm serve RedHatAI/Qwen3-8B-speculator.eagle3
```

The speculator config is embedded in the model's `config.json` — vLLM auto-detects and loads it.

### Expected Speedup

| Scenario | Without Spec Decode | With Spec Decode | Speedup |
|----------|--------------------|--------------------|---------|
| Short response (~50 tokens) | 1.5s | 0.7s | ~2.1x |
| Medium response (~200 tokens) | 5.0s | 2.2s | ~2.3x |
| Long response (~500 tokens) | 12s | 5.5s | ~2.2x |

*Note: Speedup depends on acceptance rate, which varies by prompt type. Financial advisory text tends to have high acceptance rates due to formulaic language patterns.*

### Available Speculator Models for SavVio's Candidate Base Models

| Base Model | Speculator | Size | Downloads |
|------------|-----------|------|-----------|
| Llama-3.1-8B-Instruct | `RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3` | 1.0B | 21.6K |
| Qwen3-8B | `RedHatAI/Qwen3-8B-speculator.eagle3` | 1B | 81.2K |
| Qwen3-14B | `RedHatAI/Qwen3-14B-speculator.eagle3` | 1B | 698 |

### Alternatives

| Alternative | Pros | Cons |
|-------------|------|------|
| **Medusa** | Multi-head decoding, no draft model needed | Less speedup than EAGLE-3, limited model support |
| **Lookahead Decoding** | No additional model needed | Marginal speedup for short outputs |
| **Streaming without speculation** | Simple, no overhead | Doesn't reduce total latency, just perceived latency |

### SavVio-Specific Recommendation

After deploying vLLM with a base model (Enhancement 1), simply switch to serving the speculator variant. If using Qwen3-8B, swap to `RedHatAI/Qwen3-8B-speculator.eagle3` and benchmark with GuideLLM.

---

## Enhancement 4 — Mixture of Experts (MoE) Models

### What Is It?

[Mixture of Experts (MoE)](https://huggingface.co/blog/moe) models replace standard FFN layers with multiple "expert" sub-networks and a learned router that dynamically selects 1-2 experts per token. This enables models with many more total parameters while keeping inference FLOPs comparable to a much smaller dense model.

**Examples:** Mixtral 8x7B (47B total, ~12B active), Qwen3-30B-A3B (30B total, 3B active), DeepSeek-R1 (671B total, ~37B active)

### Should SavVio Use It?

**Conditionally yes — MoE models offer the best quality-per-FLOP, but only if SavVio's use case justifies the memory overhead.**

### Analysis for SavVio

| Factor | Assessment |
|--------|-----------|
| **Quality needs** | SavVio's LLM tasks are relatively straightforward (intent parsing, templated response generation). A dense 7-8B model is likely sufficient. |
| **Memory overhead** | MoE models require all experts loaded in VRAM (e.g., Mixtral 8x7B needs ~25GB quantized). Dense 8B models need ~5GB quantized. |
| **Throughput benefit** | MoE has faster inference per token than a same-quality dense model (fewer active params). |
| **Cost-benefit** | Only worthwhile if SavVio needs quality beyond what 8B dense models provide (e.g., complex multi-step financial reasoning). |

### Where in SavVio (If Adopted)?

Replace the dense base model in the vLLM server:

```bash
# Instead of:
vllm serve Qwen/Qwen3-8B

# Use MoE for better quality at similar speed:
vllm serve Qwen/Qwen3-30B-A3B --tensor-parallel-size 1
```

The Qwen3-30B-A3B (30B total, 3B active per token) is particularly interesting — it provides 30B-class quality at 3B-class inference speed, fitting in ~16GB VRAM.

### Recommended MoE Models for SavVio

| Model | Total Params | Active Params | VRAM (FP8) | Best For |
|-------|-------------|---------------|-----------|----------|
| `Qwen3-30B-A3B` | 30B | 3B | ~16GB | Best quality/speed ratio for SavVio |
| `Mixtral-8x7B-Instruct` | 47B | 12B | ~25GB | Strong instruction following |
| `DeepSeek-R1-Distill-Qwen-14B` | 14B (dense distill) | 14B | ~8GB | Reasoning tasks, but dense |

### When NOT to Use MoE

- If SavVio's responses are adequately handled by 7-8B dense models (test this first)
- If GPU memory is constrained (<16GB)
- If fine-tuning is planned (MoE fine-tuning is more complex and prone to overfitting)

### Alternatives to MoE

| Alternative | When Better |
|-------------|------------|
| **Quantized dense models** (e.g., Qwen3-8B in INT4) | When VRAM is very limited; simpler to deploy and fine-tune |
| **Model distillation** | When you want MoE-quality from a smaller model; train a student on MoE teacher outputs |
| **RAG augmentation** | When quality gaps come from missing knowledge, not model capacity |

### SavVio-Specific Recommendation

**Start with a dense 8B model (Qwen3-8B-Instruct or Llama-3.1-8B-Instruct).** Benchmark quality on SavVio's guardrail test suite (12 checks). If any checks consistently fail or response quality is insufficient, evaluate Qwen3-30B-A3B as an upgrade path.

---

## Enhancement 5 — llm-d: Distributed Inference Orchestration

### What Is It?

[llm-d](https://llm-d.ai/docs/architecture) is a production inference orchestration layer built on top of vLLM and Kubernetes. It provides:

1. **Intelligent Inference Scheduling** — Prefix-cache-aware routing, utilization-based load balancing, multi-tenant fairness
2. **Disaggregated Serving** — Split prefill (prompt processing) and decode (token generation) onto separate GPU pools for better TTFT
3. **Wide Expert Parallelism** — Distribute MoE experts across multiple GPUs for frontier models like DeepSeek-R1
4. **Tiered KV Cache Offloading** — GPU → CPU → SSD → Remote storage hierarchy
5. **Workload Autoscaling** — SLA-aware scaling of model replicas

### Should SavVio Use It?

**Not yet — this is a Phase 3+ enhancement for high-scale production.** llm-d requires:
- Kubernetes (1.29+) with Kubernetes Gateway API
- Multi-GPU infrastructure
- High concurrent user loads (100+ QPS)

SavVio is currently on Docker Compose / Cloud Run. llm-d becomes relevant when:
- SavVio moves to GKE (Google Kubernetes Engine)
- Traffic exceeds what a single vLLM instance can handle
- Multiple LLM models need to be served simultaneously (e.g., different models for intent parsing vs. response generation)

### Where in SavVio (If Adopted)?

```
                    Internet
                       │
                       ▼
              ┌─── llm-d Inference Gateway ───┐
              │   Prefix-cache-aware routing   │
              │   SLA-based prioritization     │
              └──────────┬────────────────────┘
                    ┌────┤────┐
                    ▼         ▼
            ┌────────────┐  ┌────────────┐
            │ vLLM Pod 1 │  │ vLLM Pod 2 │
            │ (Prefill)  │  │ (Decode)   │
            └────────────┘  └────────────┘
```

### Prerequisites

- GKE cluster with GPU node pools
- Kubernetes Gateway API support
- Helm for deploying llm-d components
- NVIDIA GPU Operator installed

### Alternatives

| Alternative | When Better |
|-------------|------------|
| **Simple nginx/HAProxy load balancing** | <5 vLLM replicas, no prefix-cache-awareness needed |
| **Ray Serve** | If already in Ray ecosystem for training |
| **Cloud Run with multiple instances** | For serverless auto-scaling without Kubernetes |
| **GCP Vertex AI endpoints** | If you want fully managed LLM serving (higher cost, zero ops) |

### SavVio-Specific Recommendation

**Skip for now.** Focus on single-instance vLLM serving first. Plan for llm-d adoption when SavVio transitions from Docker Compose to GKE and anticipates >50 QPS sustained traffic.

---

## Enhancement 6 — NVIDIA NeMo Agent Toolkit

### What Is It?

[NVIDIA NeMo Agent Toolkit](https://developer.nvidia.com/nemo-agent-toolkit) (formerly AgentIQ) is an open-source library for building, evaluating, and optimizing multi-agent AI systems. Key capabilities:

- **YAML-based agent workflow configuration** — Define agents, tools, and orchestration declaratively
- **Framework-agnostic** — Works with LangChain, Google ADK, CrewAI, or custom frameworks
- **Built-in evaluation** — Test agents against datasets, score with customizable metrics
- **Agent Hyperparameter Optimizer** — Auto-tune model selection, temperature, max_tokens, prompts
- **OpenTelemetry observability** — Trace every step of agent workflows
- **Safety & Security middleware** — Red-team agentic workflows, detect prompt injection, jailbreaks

### Should SavVio Use It?

**Yes — this is highly relevant for SavVio's evolution toward agentic workflows.** Currently SavVio's LLM integration is a simple two-step pipeline (Intent Parse → Response Generate). NeMo Agent Toolkit would enable:

1. **Multi-agent architecture** — Separate agents for financial analysis, product research, and response synthesis
2. **Tool use** — Agents that can query the PostgreSQL database, call the deterministic engine, and invoke the ML model as tools
3. **Evaluation pipeline** — Systematic testing of agent accuracy against SavVio's guardrail test suite
4. **Prompt optimization** — Auto-tune system prompts for each LLM role
5. **Safety middleware** — Replace/augment SavVio's 6 code-level guardrails with NeMo's learned safety models

### Where in SavVio?

```
┌─── NeMo Agent Toolkit Workflow ─────────────────┐
│                                                  │
│  Agent 1: Intent Parser                          │
│  ├── Tool: product_search (pgvector)             │
│  └── Tool: intent_classify (LLM)                 │
│                                                  │
│  Agent 2: Financial Analyzer                     │
│  ├── Tool: run_financial_engine (deterministic)  │
│  ├── Tool: run_ml_model (XGBoost)                │
│  └── Tool: compute_affordability (Python)        │
│                                                  │
│  Agent 3: Response Generator                     │
│  ├── Tool: generate_recommendation (LLM)         │
│  └── Tool: apply_guardrails (G1-G6)              │
│                                                  │
│  Orchestrator: Sequential Pipeline               │
│  Observability: OpenTelemetry → Grafana          │
│  Safety: NeMo middleware (prompt injection, etc.) │
└──────────────────────────────────────────────────┘
```

### How to Integrate

1. Install: `pip install nvidia-nat`
2. Define SavVio's workflow in YAML:
   ```yaml
   workflow:
     name: savvio_advisor
     agents:
       - name: intent_parser
         framework: custom
         model: Qwen/Qwen3-8B
         tools:
           - product_search
           - intent_classify
       - name: financial_analyzer
         framework: custom
         tools:
           - financial_engine
           - ml_model
       - name: response_generator
         framework: custom
         model: Qwen/Qwen3-8B
         tools:
           - generate_response
           - guardrail_check
   ```
3. Wrap existing SavVio functions as NeMo-compatible tools
4. Add evaluation datasets for automated accuracy testing
5. Enable OpenTelemetry export for observability

### Alternatives

| Alternative | Pros | Cons |
|-------------|------|------|
| **LangChain/LangGraph** | Most popular, huge ecosystem | No built-in optimization, less focus on evaluation |
| **CrewAI** | Easy multi-agent setup | Less production-grade, limited observability |
| **AutoGen (Microsoft)** | Strong multi-agent patterns | Complex, more research-oriented |
| **Custom Python orchestration** | Full control, no dependencies | No evaluation framework, no observability out-of-box |
| **Semantic Kernel** | Microsoft-backed, enterprise pattern | .NET-first, less Python-native |

### SavVio-Specific Recommendation

**Adopt NeMo Agent Toolkit when transitioning SavVio from a simple pipeline to a multi-agent system.** The immediate win is the **evaluation and safety middleware** — SavVio already has a NeMo-ready guardrail interface (`guardrails.py`), and the toolkit can augment this with learned safety models for prompt injection detection.

---

## Enhancement 7 — NVIDIA AI Blueprints

### What Is It?

[NVIDIA AI Blueprints](https://build.nvidia.com/blueprints) are end-to-end reference implementations for common AI application patterns. Each blueprint includes code, models, and deployment configurations.

### Relevant Blueprints for SavVio

| Blueprint | Relevance to SavVio | Priority |
|-----------|---------------------|----------|
| **[AI-Q Research Agent](https://build.nvidia.com/nvidia/aiq)** | Reference architecture for agentic RAG systems. Directly applicable to SavVio's intent parsing + product resolution pipeline. | 🔴 High |
| **[Retail Shopping Assistant](https://build.nvidia.com/nvidia/retail-shopping-assistant)** | Shopping assistant with NIM models. SavVio is fundamentally a shopping advisor — this blueprint's architecture can be studied. | 🔴 High |
| **[Safety for Agentic AI](https://build.nvidia.com/nvidia/safety-for-agentic-ai)** | Safety patterns for AI agents. Directly relevant to SavVio's guardrail system (G1-G6). | 🟡 Medium |
| **[Financial Fraud Detection](https://build.nvidia.com/nvidia/financial-fraud-detection)** | Financial AI patterns. Some overlap with SavVio's financial risk assessment (RED rules). | 🟡 Medium |
| **[Streaming Data to RAG](https://build.nvidia.com/nvidia/streaming-data-to-rag)** | Real-time data ingestion for RAG. Useful if SavVio adds live price tracking or real-time financial data feeds. | 🟢 Low |
| **[AI Model Distillation](https://build.nvidia.com/nvidia/ai-model-distillation-for-financial-data)** | Model distillation for financial data. Could produce a smaller, SavVio-specialized LLM from a larger teacher. | 🟢 Low |

### Should SavVio Use Them?

**Yes — as reference architectures and learning resources.** Blueprints are not libraries to install; they are complete example applications. The value is:
- **Architecture patterns** — How NVIDIA structures agent workflows, safety layers, and RAG pipelines
- **NIM integration examples** — How to deploy models using NVIDIA NIM microservices
- **Safety patterns** — How to implement enterprise-grade guardrails beyond code-level checks

### How to Leverage

1. **Clone the AI-Q Blueprint** and study its agentic workflow structure
2. **Extract the guardrail patterns** from the Safety for Agentic AI blueprint
3. **Benchmark SavVio against the Retail Shopping Assistant** blueprint to identify architecture gaps
4. Use blueprint code samples as templates when building NeMo Agent Toolkit integrations

### SavVio-Specific Recommendation

- **Study and reference the AI-Q and Retail Shopping Assistant blueprints** before implementing NeMo Agent Toolkit (Enhancement 6)
- Do NOT try to wholesale adopt a blueprint — SavVio's deterministic engine + ML confidence layer architecture is unique and should be preserved
- Extract specific components (safety middleware, RAG patterns) rather than full blueprints

---

## Technology Comparison Matrix

| Technology | Priority | Complexity | Cost Impact | Quality Impact | Latency Impact | Prerequisites |
|-----------|----------|-----------|------------|----------------|---------------|--------------|
| **vLLM** | 🔴 P0 | Medium | -90% API cost | Neutral | -60% TTFT | GPU (A10G/L4/T4) |
| **KV Caching** | 🔴 P0 | Low | Free (built into vLLM) | Neutral | -30% TTFT | vLLM deployed |
| **Speculative Decoding** | 🟡 P1 | Low | Free (built into vLLM) | Neutral (lossless) | -50% TPOT | vLLM deployed |
| **MoE Models** | 🟡 P1 | Low | GPU VRAM increase | +Quality | ~Neutral | ≥16GB VRAM |
| **NeMo Agent Toolkit** | 🟡 P1 | High | Dev time | +Quality, +Safety | +Latency (multi-step) | NeMo installed |
| **NVIDIA Blueprints** | 🟢 P2 | Low | Free (reference only) | Learning | N/A | None |
| **llm-d** | 🟢 P3 | Very High | Kubernetes costs | Neutral | -TTFT at scale | GKE, multi-GPU |

---

## Recommended Adoption Roadmap

### Phase 1 — Self-Hosted Inference Foundation (Weeks 1-4)

```
[ ] Deploy vLLM with Qwen3-8B-Instruct (FP8 quantized)
[ ] Enable Automatic Prefix Caching for system prompt
[ ] Add VLLMProvider to llm/llm_provider.py (reuse OpenAI SDK interface)
[ ] Benchmark against current Gemini API: latency, quality, guardrail pass rate
[ ] Enable FP8 Quantized KV Cache if VRAM is tight
[ ] Update docker-compose.yml with vllm-server service
[ ] Add fallback: if vLLM is down, fall back to external API
```

### Phase 2 — Inference Optimization (Weeks 5-8)

```
[ ] Enable speculative decoding: switch to RedHatAI/Qwen3-8B-speculator.eagle3
[ ] Benchmark TPOT improvement with GuideLLM
[ ] Evaluate MoE model (Qwen3-30B-A3B) if quality gaps exist
[ ] Implement structured output mode for intent parsing (vLLM supports JSON mode)
[ ] Add vLLM Prometheus metrics to SavVio's monitoring dashboard
```

### Phase 3 — Agentic Architecture (Weeks 9-16)

```
[ ] Study NVIDIA AI-Q and Retail Shopping Assistant blueprints
[ ] Install NeMo Agent Toolkit (pip install nvidia-nat)
[ ] Define SavVio agentic workflow in YAML (3 agents: intent, financial, response)
[ ] Wrap existing SavVio components as NeMo tools
[ ] Add NeMo safety middleware for prompt injection detection
[ ] Build evaluation datasets for automated agent accuracy testing
[ ] Run Agent Hyperparameter Optimizer on system prompts and model params
```

### Phase 4 — Production Scaling (When traffic demands it)

```
[ ] Migrate from Docker Compose to GKE
[ ] Deploy llm-d with inference scheduling
[ ] Enable prefill/decode disaggregation for lower TTFT
[ ] Configure workload autoscaling with SLO targets
[ ] Add tiered KV cache offloading (GPU → CPU → SSD)
```

---

## Alternatives & Trade-offs

### Alternative Path: Stay on External APIs + Focus on Prompt Engineering

**When this makes sense:**
- SavVio traffic stays low (<1000 requests/day)
- No GPU budget available
- Team expertise is in ML/data engineering, not GPU infrastructure
- Time-to-market is the priority

**Trade-off:** Higher per-token cost but zero infrastructure complexity. OpenRouter already provides reliable routing across Gemini/OpenAI/Claude.

### Alternative Path: Fully Managed Solutions (Vertex AI, AWS Bedrock)

**When this makes sense:**
- Zero tolerance for GPU ops overhead
- Budget allows managed pricing
- Compliance requirements demand cloud-provider SLAs

**Options:**
- **GCP Vertex AI with Gemini** — Already partially used (direct Gemini SDK)
- **AWS Bedrock** — If migrating from GCP
- **Azure OpenAI Service** — If enterprise Azure estate exists

**Trade-off:** Higher cost but no GPU management. Less control over optimization.

### Alternative Path: Fine-Tuned Small Model

**When this makes sense:**
- SavVio's LLM tasks are narrow enough that a 1-3B model fine-tuned on SavVio data can match larger model quality
- Extreme cost sensitivity
- Edge deployment needed

**How:**
1. Collect SavVio conversation logs (intent parsing inputs/outputs, response generation inputs/outputs)
2. Fine-tune `Qwen3-4B` or `Phi-4-mini` on this data
3. Serve via vLLM on a T4 GPU

**Trade-off:** Requires training data collection and fine-tuning expertise, but result is a purpose-built model with lowest possible inference cost.

---

## Open Questions

> **Q1:** What GPU budget is available for self-hosted inference? This determines whether to deploy on T4 (cheapest, 16GB), L4 (mid-tier, 24GB), or A10G (performance, 24GB).

> **Q2:** Is SavVio planning to evolve from single-turn Q&A to multi-turn conversational sessions? This affects the priority of KV caching, session management, and NeMo Agent Toolkit adoption.

> **Q3:** Is there a migration timeline from Docker Compose + Cloud Run to GKE? This determines when llm-d becomes relevant.

> **Q4:** What is the expected traffic volume at launch? If <100 QPS, single-instance vLLM is sufficient. If >100 QPS, llm-d and autoscaling planning should begin early.

> **Q5:** Is fine-tuning a possibility? If SavVio can collect conversation logs from the current API-based deployment, these can be used to fine-tune a small specialized model later.

> **Q6:** Does the team want to keep OpenRouter as a fallback even after deploying vLLM? A hybrid approach (vLLM primary + OpenRouter fallback) provides maximum reliability.

---

## References

| Technology | Primary Link | Documentation |
|-----------|-------------|---------------|
| vLLM | https://docs.vllm.ai | [Quickstart](https://docs.vllm.ai/en/v0.8.0/getting_started/quickstart.html) |
| llm-d | https://llm-d.ai | [Architecture](https://llm-d.ai/docs/architecture) |
| MoE (HuggingFace) | https://huggingface.co/blog/moe | [Mixtral Collection](https://huggingface.co/mistralai) |
| KV Caching | https://huggingface.co/blog/not-lain/kv-caching | [HF Cache Docs](https://huggingface.co/docs/transformers/main/en/generation_strategies#kv-caching) |
| Speculators | https://github.com/vllm-project/speculators | [RedHatAI Models](https://huggingface.co/collections/RedHatAI/speculator-models) |
| NeMo Agent Toolkit | https://developer.nvidia.com/nemo-agent-toolkit | [Docs](https://docs.nvidia.com/nemo/agent-toolkit/latest/index.html) |
| NVIDIA Blueprints | https://build.nvidia.com/blueprints | [AI-Q Blueprint](https://build.nvidia.com/nvidia/aiq) |
