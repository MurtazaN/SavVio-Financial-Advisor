# LLM Inference Pipeline & Fiduciary Advocacy

The LLM layer is the "Advocate" of SavVio. It bridges the gap between natural language user queries and the deterministic financial core. Its primary roles are to extract intent, resolve products, and translate complex financial math into empathetic, clear, and safe advice.

Source files:
- `model_pipeline/src/llm/llm_provider.py` — Provider abstraction (OpenRouter, OpenAI, Gemini, Claude)
- `model_pipeline/src/llm/intent_parser.py` — Natural language intent extraction
- `model_pipeline/src/llm/product_resolver.py` — Vector-based product matching
- `model_pipeline/src/llm/response_generator.py` — Conversational recommendation generation
- `model_pipeline/src/llm/guardrails.py` — 6-point fiduciary safety verification

---

## Overview: The Inference Flow

```
User Query ("Should I buy this $1,500 laptop?")
     │
     ▼
┌──────────────────────────────┐
│     Part 1: Intent Parser    │  Extracts: "laptop", $1,500, "buy"
│      (intent_parser.py)      │  Uses LLM with Regex fallback.
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Part 2: Product Resolver   │  Searches PostgreSQL + pgvector.
│     (product_resolver.py)    │  Matches "laptop" -> product_id: 8214.
└──────────────┬───────────────┘
               │
               ▼
   [Deterministic Engine]      ← Authoritative Financial Math (Layer 1 & 2)
   [ML Model Confidence]       ← Scoring Signal (XGBoost)
               │
               ▼
┌──────────────────────────────┐
│  Part 3: Response Generator  │  Combines Engine Output + ML Score +
│    (response_generator.py)   │  User Context -> Fiduciary Recommendation.
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Part 4: Guardrails      │  Checks for hallucinations, contradictions, 
│       (guardrails.py)        │  and "internal leakage" before delivery.
└──────────────┬───────────────┘
               │
               ▼
      Final LLM Response
```

---

## LLM Provider Abstraction (`llm_provider.py`)

SavVio uses a **Provider Strategy Pattern** to remain model-agnostic and resilient to rate limits.

| Provider | Implementation | Best Use Case |
|----------|----------------|---------------|
| **OpenRouter** | `urllib` (no SDK) | **Production Hub**. Accesses `gemini-2.0-flash` with 100% uptime and no RPM limits. |
| **Direct SDKs**| `openai`, `anthropic`, `google-genai` | **Redundancy**. High-performance direct access if OpenRouter is down. |
| **Mock** | Deterministic Regex | **CI/CD & Local Debugging**. Returns valid JSON without using API credits. |

---

## Part 1: Intent Parser (`intent_parser.py`)

The Intent Parser translates messy natural language into a structured **`ParsedIntent`** object.

*   **Hybrid Logic**: Uses an LLM to follow complex instructions, but falls back to a robust **Regex Engine** if the API fails or returns invalid JSON.
*   **Extraction details**:
    *   `intent`: purchase_query, browse, or generic_chat.
    *   `product_reference`: The subject of the query (e.g., "Airpods Pro").
    *   `price_context`: Any specific price mentioned by the user to override catalog defaults.

---

## Part 2: Product Resolver (`product_resolver.py`)

Once a product is identified by name, the resolver finds the best-matching entry in the PostgreSQL catalog using **pgvector**.

*   **Vector Search**: Converts the product name into a 384-dimensional embedding using `sentence-transformers`.
*   **Ranking**: Uses cosine similarity to find the top match, cross-referenced against category metadata to ensure accuracy.

---

## Part 3: Response Generator (`response_generator.py`)

The Response Generator takes the raw **Green/Yellow/Red** results and crafts a conversation.

*   **Fiduciary Voice**: The prompt forces the LLM to remain objective, honest, and data-driven. It is strictly forbidden from "upselling" a Red purchase.
*   **Structured Explanations**: 
    1.  **Direct Answer**: Clear Green/Yellow/Red status.
    2.  **Financial Reasoning**: Reference to PIR, RUS, or specific rules triggered.
    3.  **Product Context**: Mentioning reviews or quality signals from Layer 2.
    4.  **Final Advocacy**: Actionable advice (e.g., "Wait until next month's bonus").

---

## Part 4: Fiduciary Guardrails (`guardrails.py`)

To ensure the LLM never violates its fiduciary duty, every response undergoes **6 code-level safety checks**:

| ID | Guardrail | Logic |
|----|-----------|-------|
| **G1** | **Color Contradiction** | Blocks a response if the LLM says "Go ahead" (Green) when the engine says RED. |
| **G2** | **Hallucinated Figures** | Ensures no dollar amounts appear in the advice that weren't in the input data. |
| **G3** | **Out-of-Scope Advice** | Flags investment, stock, or credit advice that SavVio is not qualified to give. |
| **G4** | **Internal Leakage** | Blocks mentions of internal rule names (e.g., `RED_RULE_3`) or system prompts. |
| **G5** | **Tone Mismatch** | Ensures RED alerts sound serious and GREEN encouragements sound supportive. |
| **G6** | **Length Check** | Truncates over-talkative responses to keep the UX concise. |

---

## API Usage

```python
from llm.inference_pipeline import process_user_query

# One-stop shop for everything: Parsing -> Resolving -> Generating -> Guarding
response = process_user_query(
    query="Should I buy this $1200 MacBook?",
    user_id="user_99",
    provider=get_provider("openrouter")
)

print(response.decision)    # "RED"
print(response.explanation) # "SavVio recommends waiting. This purchase would leave..."
```
