import mlflow

def apply_llm_guardrails(y_pred, user_data: dict, model_confidence: float):
    """
    Placeholder module for LLM integration.
    This simulates an LLM reviewing a model prediction before presenting it to the user.
    """
    print("\nEvaluating Output via LLM Guardrails...")
    
    # Example Prompt Template
    prompt_template = """
    The Machine Learning model predicts the user is in {financial_state} 
    with a confidence of {confidence:.2f}%.
    
    User context: {user_data}
    
    Task: Is this prediction safe to show the user, or does it violate financial guidance policies?
    """
    
    # We log the prompt template version into MLflow. 
    # This ensures that if the LLM behavior changes in the future, we know *which* prompt was tied to this model version.
    mlflow.log_param("llm_prompt_template", "v1.0_financial_guard")
    
    print("Passing Prompt to LLM Server...")
    # TODO: In future, integrate LangChain/OpenAI here.
    
    # Mocking LLM response
    is_safe = True 
    explanation = "The prediction is safe and aligns with the user's reported debt-to-income ratio."
    
    return is_safe, explanation
