Phase 2 (Data Pipeline — current):
    Financial data → features → PostgreSQL
    Products → enriched with rating_variance → PostgreSQL + pgvector
    Reviews → preprocessed → PostgreSQL
    ✅ No artificial merge, no features.csv

Phase 3 (Model Pipeline — 3 weeks):
    Read from PostgreSQL
    → Generate synthetic training scenarios
    → Deterministic engine computes affordability
    → Label scenarios (Green/Yellow/Red)
    → Train XGBoost classifier (Green/Yellow/Red + confidence score to LLM)
    → MLflow tracking, validation, bias detection
    → Model registry - GCP artifact registry
    → FastAPI Decision API
    -----------
    → Integrate deterministic engine + XGBoost + LLM
    → LLM wraps XGBoost's decision in conversation
    → NeMo Guardrails - Security, compliance and prevent hallucinations. 
    → Basic prompt engineering (Option B from Claude)
    ------------
    → Monitoring, CI/CD
    LLM integration happens here
    
Phase 4 Deployment - 3 weeks