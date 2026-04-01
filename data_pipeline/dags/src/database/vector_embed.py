"""
Generate vector embeddings and store in pgvector.

Two embedding targets:
  1. Products  — name + category + description + features + details
                 → product_embeddings table (for product-level RAG retrieval)
  2. Reviews   — review_title + review_text
                 → review_embeddings table (for contextual utility analysis)

Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim) by default.
Processes in batches to keep memory usage reasonable.
"""

from __future__ import annotations
import json
import os
import logging
import pandas as pd
import numpy as np
from sqlalchemy import text

from savviocore.database.db_connection import get_engine, ensure_pgvector
from savviocore.database.db_schema import create_tables
# free. No API keys, no billing, no rate limits
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
'''
If retrieval quality is lacking, swap EMBEDDING_MODEL to "all-mpnet-base-v2" (768-dim) 
for better semantic understanding at the cost of larger storage and slightly slower embedding time. 
(just change the model name string) or an API-based option without changing the rest of the pipeline.
'''

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load the sentence-transformer model once."""
    # from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

def build_product_text(row: pd.Series) -> str:
    """
    Combine product text fields into a single string for embedding.
    Concatenates: name | category | description | features | details
    """
    parts = []
    for col in ["product_name", "category", "description", "features"]:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())

    # Flatten details (dict/JSON) into readable key-value pairs
    details = row.get("details", None)
    if details:
        details_str = _flatten_details(details)
        if details_str:
            parts.append(details_str)

    return " | ".join(parts)


def _flatten_details(details) -> str:
    """Convert details dict/JSON into a readable string for embedding."""
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except (json.JSONDecodeError, TypeError):
            return details.strip()
    if isinstance(details, dict):
        return ", ".join(f"{k}: {v}" for k, v in details.items() if v)
    return ""


def build_review_text(row: pd.Series) -> str:
    """
    Combine review title and body into a single string for embedding.
    Title provides a concise signal; body provides detail.
    """
    parts = []
    for col in ["review_title", "review_text"]:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Embedding generation (shared)
# ---------------------------------------------------------------------------

def generate_embeddings(texts: list[str], model) -> np.ndarray:
    """
    Encode a list of texts into embeddings, batched.
    Returns numpy array of shape (len(texts), EMBEDDING_DIM).
    """
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,  # Disabled tqdm as it doesn't work well in Airflow logs
        normalize_embeddings=True,
    )
    return embeddings


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------
'''
Separate table:

Embeddings are large (384 floats per row) — keeping them separate means queries 
on the products table that don't need embeddings stay fast.
You can rebuild/reindex embeddings without touching the main table.
You can have multiple embedding versions (swap models, compare results).
pgvector indexing (IVFFlat, HNSW) works on a dedicated table without bloating the main table's index.

For SavVio, the separate table is better because we likely iterate on embedding models, 
and we don't want to ALTER the products table every time. 
It's also the standard pattern for RAG systems. 
For a small academic project, either works — it's a design preference, not a hard requirement.
'''

def _ensure_embedding_tables(engine):
    """Create product_embeddings and review_embeddings tables."""
    product_ddl = f"""
    CREATE TABLE IF NOT EXISTS product_embeddings (
        id          SERIAL PRIMARY KEY,
        product_id  VARCHAR(100) UNIQUE NOT NULL REFERENCES products(product_id),
        embedding   vector({EMBEDDING_DIM}),
        created_at  TIMESTAMP DEFAULT NOW()
    );
    """
    review_ddl = f"""
    CREATE TABLE IF NOT EXISTS review_embeddings (
        id          SERIAL PRIMARY KEY,
        product_id  VARCHAR(100) NOT NULL,
        user_id     VARCHAR(255) NOT NULL,
        embedding   vector({EMBEDDING_DIM}),
        created_at  TIMESTAMP DEFAULT NOW(),
        UNIQUE (product_id, user_id)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(product_ddl))
        conn.execute(text(review_ddl))
    logger.info("Embedding tables ensured (product_embeddings, review_embeddings)")


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

'''
To join back to the full review text, do 
JOIN reviews ON 
    review_embeddings.product_id = reviews.product_id 
    AND review_embeddings.user_id = reviews.user_id
'''

def store_product_embeddings(engine, product_ids: list[str], embeddings: np.ndarray):
    """Upsert product embeddings into product_embeddings table."""
    logger.info("Storing %d product embeddings", len(product_ids))
    sql = text("""
        INSERT INTO product_embeddings (product_id, embedding)
        VALUES (:pid, CAST(:emb AS vector))
        ON CONFLICT (product_id) DO UPDATE SET embedding = EXCLUDED.embedding
    """)
    with engine.begin() as conn:
        for pid, emb in zip(product_ids, embeddings):
            conn.execute(sql, {"pid": str(pid), "emb": str(emb.tolist())})
    logger.info("Product embeddings stored")


def store_review_embeddings(engine, product_ids: list[str], user_ids: list[str], embeddings: np.ndarray):
    """Upsert review embeddings into review_embeddings table."""
    logger.info("Storing %d review embeddings", len(product_ids))
    sql = text("""
        INSERT INTO review_embeddings (product_id, user_id, embedding)
        VALUES (:pid, :uid, CAST(:emb AS vector))
        ON CONFLICT (product_id, user_id) DO UPDATE SET embedding = EXCLUDED.embedding
    """)
    with engine.begin() as conn:
        for pid, uid, emb in zip(product_ids, user_ids, embeddings):
            conn.execute(sql, {
                "pid": str(pid),
                "uid": str(uid),
                "emb": str(emb.tolist()),
            })
    logger.info("Review embeddings stored")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def embed_products(engine, products_path: str, model):
    """Read processed products → embed → store."""
    df = _read_file(products_path)
    logger.info("Loaded %d products from %s", len(df), products_path)

    if "product_id" not in df.columns:
        raise ValueError("Products file must contain a 'product_id' column")

    df["_embed_text"] = df.apply(build_product_text, axis=1)
    df = df[df["_embed_text"].str.strip().astype(bool)].reset_index(drop=True)
    logger.info("Products with non-empty embedding text: %d", len(df))

    texts = df["_embed_text"].tolist()
    product_ids = df["product_id"].tolist()
    total = len(texts)
    
    # Process and store in chunks to provide explicit progress logs
    chunk_size = 5000
    for i in range(0, total, chunk_size):
        chunk_texts = texts[i : i + chunk_size]
        chunk_pids = product_ids[i : i + chunk_size]
        
        logger.info("Generating embeddings for products %d to %d (out of %d)...", i, i + len(chunk_texts), total)
        embeddings = generate_embeddings(chunk_texts, model)
        store_product_embeddings(engine, chunk_pids, embeddings)
        
    return len(df)


def embed_reviews(engine, reviews_path: str, model):
    """
    Read processed reviews → embed → store.
    Only embeds reviews that have text content (title or body).
    Reviews are linked to products via product_id for filtered retrieval.
    """
    df = _read_file(reviews_path)
    logger.info("Loaded %d reviews from %s", len(df), reviews_path)

    for col in ["product_id", "user_id"]:
        if col not in df.columns:
            raise ValueError(f"Reviews file must contain a '{col}' column")

    df["_embed_text"] = df.apply(build_review_text, axis=1)

    # Only embed reviews that have actual text
    df = df[df["_embed_text"].str.strip().astype(bool)].reset_index(drop=True)
    logger.info("Reviews with text content to embed: %d", len(df))

    if df.empty:
        logger.warning("No reviews with text found — skipping review embeddings")
        return 0

    texts = df["_embed_text"].tolist()
    product_ids = df["product_id"].tolist()
    user_ids = df["user_id"].tolist()
    total = len(texts)
    
    # Process and store in chunks to provide explicit progress logs
    chunk_size = 5000
    for i in range(0, total, chunk_size):
        chunk_texts = texts[i : i + chunk_size]
        chunk_pids = product_ids[i : i + chunk_size]
        chunk_uids = user_ids[i : i + chunk_size]
        
        logger.info("Generating embeddings for reviews %d to %d (out of %d)...", i, i + len(chunk_texts), total)
        embeddings = generate_embeddings(chunk_texts, model)
        store_review_embeddings(engine, chunk_pids, chunk_uids, embeddings)

    return len(df)


# ---------------------------------------------------------------------------
# File reader helper
# ---------------------------------------------------------------------------

def _read_file(path: str) -> pd.DataFrame:
    """Read CSV or JSONL based on file extension."""
    if path.endswith(".jsonl"):
        # return pd.read_json(path, lines=True)
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        if file_size_mb > 300:
            return pd.concat(pd.read_json(path, lines=True, chunksize=100_000), ignore_index=True)
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_embed(
    products_path: str,
    reviews_path: str | None = None,
    env: str = "dev",
):
    """
    End-to-end embedding pipeline.

    Args:
        products_path: Path to processed products file (CSV or JSONL)
        reviews_path:  Path to processed reviews file (CSV or JSONL).
                       If None, only product embeddings are generated.
        env: "dev" or "prod"
    """
    engine = get_engine(env)
    ensure_pgvector(engine)
    create_tables(engine)
    _ensure_embedding_tables(engine)

    model = load_model()

    n_products = embed_products(engine, products_path, model)
    logger.info("Embedded %d products", n_products)

    n_reviews = 0
    if reviews_path:
        n_reviews = embed_reviews(engine, reviews_path, model)
        logger.info("Embedded %d reviews", n_reviews)
    else:
        logger.info("No reviews path provided — skipping review embeddings")

    summary = {"products_embedded": n_products, "reviews_embedded": n_reviews}
    logger.info("Embedding pipeline complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from src.utils import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="Generate & store embeddings (products + reviews)")
    parser.add_argument("--products", required=True, help="Path to processed products file")
    parser.add_argument("--reviews", default=None, help="Path to processed reviews file (optional)")
    parser.add_argument("--env", default="dev", choices=["dev", "prod"])
    args = parser.parse_args()

    result = run_embed(args.products, args.reviews, args.env)
    print("Done:", result)
