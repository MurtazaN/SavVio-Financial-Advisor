"""
Intent extraction prompt — sent to the LLM to parse user queries.

Used by intent_parser.py when a real LLM provider is active.
When the mock provider is active, regex-based parsing is used instead.
"""

VERSION = "v1.0"

INTENT_EXTRACTION_PROMPT = """You are SavVio's input parser. The user sent a message.

Your job: determine what the user wants and extract the product they're asking about.

Classify the intent as one of:
- "purchase_query" — the user is asking about buying, purchasing, or affording a product
- "general_question" — the user is asking about their finances or about SavVio
- "out_of_scope" — the user is asking about something unrelated

Extract the product name/reference if it's a purchase_query.
Also extract any user context about their financial situation.

Respond with ONLY valid JSON:
{{
  "intent": "purchase_query" | "general_question" | "out_of_scope",
  "product_reference": "<the product name or null>",
  "user_context": "<any context the user shared about their situation, or null>"
}}

---
User message: {query}
"""

# JSON schema for structured generation (used by generate_structured)
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["purchase_query", "general_question", "out_of_scope"],
        },
        "product_reference": {"type": ["string", "null"]},
        "user_context": {"type": ["string", "null"]},
    },
    "required": ["intent", "product_reference"],
}
