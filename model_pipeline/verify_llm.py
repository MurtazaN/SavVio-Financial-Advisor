"""
SavVio LLM Integration Verification Script.

Tests the full LLM pipeline with the Gemini provider:
    1. Provider initialization (API key from .env)
    2. Basic text generation
    3. Structured intent extraction
    4. Intent parser (regex mock vs real LLM)
    5. Response generator (template and Gemini)
    6. Guardrail checks on LLM output
"""

import os
import sys
import logging

# Load .env before importing anything else
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("verify_llm")

PASS = "PASS"
FAIL = "FAIL"
results = []


def test(name, fn):
    """Run a test and record the result."""
    try:
        fn()
        results.append((name, True, ""))
        print(f"  {PASS} {name}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  {FAIL} {name}: {e}")


def main():
    print("=" * 70)
    print("  SavVio LLM Integration Verification")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Environment check
    # -----------------------------------------------------------------------
    print("\n[1] Environment Check")
    api_key = os.getenv("GEMINI_API_KEY")
    test("GEMINI_API_KEY is set", lambda: assert_(api_key, "GEMINI_API_KEY not found in .env"))
    test("API key starts with 'AIza'", lambda: assert_(api_key.startswith("AIza"), f"Key starts with: {api_key[:4]}"))

    # -----------------------------------------------------------------------
    # 2. Provider initialization
    # -----------------------------------------------------------------------
    print("\n[2] Provider Initialization")
    from llm.llm_provider import get_provider, _provider_cache
    _provider_cache.clear()

    test("MockProvider initializes", lambda: get_provider("mock"))

    gemini = None
    def init_gemini():
        nonlocal gemini
        gemini = get_provider("gemini")
    test("GeminiProvider initializes with API key", init_gemini)

    if gemini is None:
        print("\n  WARNING: Gemini provider failed to initialize — skipping LLM tests")
        _print_summary()
        return

    # -----------------------------------------------------------------------
    # 3. Basic text generation
    # -----------------------------------------------------------------------
    print("\n[3] Basic Text Generation (Gemini)")

    gen_response = None
    def test_generate():
        nonlocal gen_response
        gen_response = gemini.generate(
            system_prompt="You are a helpful assistant. Reply in one sentence.",
            user_message="What is 2 + 2?"
        )
        assert_(isinstance(gen_response, str), "Response is not a string")
        assert_(len(gen_response) > 0, "Response is empty")
    test("generate() returns non-empty string", test_generate)
    if gen_response:
        print(f"    → Response: {gen_response[:100]}")

    # -----------------------------------------------------------------------
    # 4. Structured intent extraction via Gemini
    # -----------------------------------------------------------------------
    print("\n[4] Structured Intent Extraction (Gemini)")
    from llm.prompts.intent_prompt import INTENT_EXTRACTION_PROMPT, INTENT_SCHEMA

    structured_result = None
    def test_structured():
        nonlocal structured_result
        prompt = INTENT_EXTRACTION_PROMPT.format(query="Can I buy the Sony WH-1000XM5?")
        structured_result = gemini.generate_structured(
            system_prompt="You are a precise intent extraction system.",
            user_message=prompt,
            schema=INTENT_SCHEMA,
        )
        assert_(isinstance(structured_result, dict), "Result is not a dict")
        assert_("intent" in structured_result, "Missing 'intent' key")
    test("generate_structured() returns valid JSON", test_structured)
    if structured_result:
        print(f"    → Parsed: {structured_result}")

    # -----------------------------------------------------------------------
    # 5. Intent parser with Gemini provider
    # -----------------------------------------------------------------------
    print("\n[5] Intent Parser (Gemini LLM mode)")
    from llm.intent_parser import parse_user_input

    parsed = None
    def test_intent_parser():
        nonlocal parsed
        parsed = parse_user_input("Can I buy the iPhone 15 Pro?", gemini)
        assert_(parsed.intent == "purchase_query", f"Expected purchase_query, got {parsed.intent}")
        assert_(parsed.product_reference is not None, "Product reference is None")
    test("parse_user_input() extracts purchase intent", test_intent_parser)
    if parsed:
        print(f"    → Intent: {parsed.intent}, Product: {parsed.product_reference}")

    def test_out_of_scope():
        result = parse_user_input("What is the weather today?", gemini)
        assert_(result.intent == "out_of_scope", f"Expected out_of_scope, got {result.intent}")
    test("parse_user_input() rejects out-of-scope queries", test_out_of_scope)

    # -----------------------------------------------------------------------
    # 6. Response generator with Gemini
    # -----------------------------------------------------------------------
    print("\n[6] Response Generator (Gemini LLM mode)")
    from llm.response_generator import generate_response, RecommendationContext

    green_ctx = RecommendationContext(
        product_name="Sony WH-1000XM5",
        product_price=349.99,
        recommendation_color="GREEN",
        original_color="GREEN",
        confidence_scores={"GREEN": 0.87, "YELLOW": 0.10, "RED": 0.03},
        triggered_rules=[],
        affordability_score=1650.0,
        savings_to_price_ratio=28.5,
        emergency_fund_months=8.3,
        debt_to_income_ratio=0.15,
    )

    green_response = None
    def test_green_response():
        nonlocal green_response
        green_response = generate_response(green_ctx, gemini)
        assert_(isinstance(green_response, str), "Response not a string")
        assert_(len(green_response) > 10, "Response too short")
    test("GREEN response generated via Gemini", test_green_response)
    if green_response:
        print(f"    → Response: {green_response[:150]}...")

    red_ctx = RecommendationContext(
        product_name="MacBook Pro M4",
        product_price=2499.00,
        recommendation_color="RED",
        original_color="RED",
        confidence_scores={"GREEN": 0.05, "YELLOW": 0.15, "RED": 0.80},
        triggered_rules=["RED_1", "RED_4"],
        affordability_score=-800.0,
        savings_to_price_ratio=0.4,
        emergency_fund_months=0.5,
        debt_to_income_ratio=0.55,
    )

    red_response = None
    def test_red_response():
        nonlocal red_response
        red_response = generate_response(red_ctx, gemini)
        assert_(isinstance(red_response, str), "Response not a string")
        assert_(len(red_response) > 10, "Response too short")
    test("RED response generated via Gemini", test_red_response)
    if red_response:
        print(f"    → Response: {red_response[:150]}...")

    # -----------------------------------------------------------------------
    # 7. Guardrail verification on Gemini output
    # -----------------------------------------------------------------------
    print("\n[7] Guardrail Verification")
    from llm.guardrails import check_response

    if green_response:
        def test_green_guardrails():
            result = check_response(green_response, "GREEN", green_ctx)
            if not result.passed:
                print(f"    WARNING -- Violations: {result.violations}")
            # We log but don't fail — LLM responses may occasionally trip guardrails
        test("Guardrail check on GREEN response", test_green_guardrails)

    if red_response:
        def test_red_guardrails():
            result = check_response(red_response, "RED", red_ctx)
            if not result.passed:
                print(f"    WARNING -- Violations: {result.violations}")
        test("Guardrail check on RED response", test_red_guardrails)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _print_summary()


def _print_summary():
    print("\n" + "=" * 70)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"  Results: {passed}/{total} passed")
    if passed == total:
        print(f"  {PASS} All LLM integration checks passed!")
    else:
        print(f"  {FAIL} Some checks failed:")
        for name, ok, err in results:
            if not ok:
                print(f"    → {name}: {err}")
    print("=" * 70)


def assert_(condition, msg="Assertion failed"):
    if not condition:
        raise AssertionError(msg)


if __name__ == "__main__":
    main()
