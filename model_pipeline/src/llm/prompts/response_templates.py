"""
Static response templates when LLM is unavailable or for error paths.
"""

OUT_OF_SCOPE_RESPONSE = (
    "I'm focused on helping you evaluate specific purchase decisions. "
    "Try asking whether you can afford a product by name or price, "
    "for example: \"Should I buy the Sony WH-1000XM5?\" or \"Can I afford a $500 laptop?\""
)

PRODUCT_NOT_FOUND_RESPONSE = (
    "I couldn't find that product in our catalog. Try naming the product more specifically "
    "or check that we have listings for it."
)

USER_NOT_FOUND_RESPONSE = (
    "We don't have a financial profile on file for this user ID. "
    "Please verify your user ID or contact support to set up your profile."
)

# Template keys used by _generate_from_template in response_generator
COLOR_INTROS = {
    "GREEN": (
        "Based on your current finances, this purchase looks **manageable**. "
        "Here's what the analysis shows for **{product_name}** at **${product_price:,.2f}**."
    ),
    "YELLOW": (
        "This purchase is **borderline** for your situation. "
        "Review the details below for **{product_name}** (**${product_price:,.2f}**)."
    ),
    "RED": (
        "Given your current financial picture, this purchase is **not recommended** right now. "
        "Details for **{product_name}** (**${product_price:,.2f}**):"
    ),
}
