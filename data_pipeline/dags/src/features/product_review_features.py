"""
Product & Review Feature Engineering  (product_review_features.py)

Computes product-level quality signals derived from individual reviews.
The only metric that requires review-level data is rating_variance,
since the product metadata already provides average_rating and rating_number.

Pipeline:
  1. Compute per-product rating_variance from individual reviews.
  2. Merge rating_variance onto the preprocessed product data.
  3. Save enriched product data  → data/features/product_featured.jsonl
  4. Copy preprocessed reviews   → data/features/review_featured.jsonl
     (no new columns; keeps the file available for downstream DB loading)

Input:  data/processed/review_preprocessed.jsonl
        data/processed/product_preprocessed.jsonl
Output: data/features/product_featured.jsonl
        data/features/review_featured.jsonl
"""

import os
import logging
import pandas as pd
import shutil

from src.features.utils import setup_logging, ensure_output_dir

# Configure module logging.
setup_logging()
logger = logging.getLogger(__name__)


def compute_rating_variance(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-product rating variance (standard deviation).

    A high variance indicates polarizing opinions (mixed quality signal).
    A low variance indicates consensus among reviewers.

    Args:
        reviews_df: DataFrame with at least 'product_id' and 'rating' columns.

    Returns:
        DataFrame with columns: product_id, rating_variance
    """
    if "product_id" not in reviews_df.columns or "rating" not in reviews_df.columns:
        logger.error("Required columns 'product_id' and 'rating' not found.")
        raise ValueError("Missing required columns: product_id, rating")

    rating_variance_df = (
        reviews_df
        .groupby("product_id")["rating"]
        .std()
        .reset_index()
        .rename(columns={"rating": "rating_variance"})
    )

    # Products with a single review have NaN std dev; default to 0.0.
    rating_variance_df["rating_variance"] = rating_variance_df["rating_variance"].fillna(0.0)

    return rating_variance_df


def run_review_features(
    reviews_path: str,
    products_path: str,
    product_output_path: str,
    review_output_path: str,
) -> None:
    """
    Executes the review feature engineering pipeline.

    1. Reads individual reviews and computes per-product rating_variance.
    2. Merges rating_variance onto the preprocessed product data.
    3. Saves enriched products → product_output_path  (JSONL)
    4. Copies reviews          → review_output_path   (JSONL)
    """
    logger.info("Starting review feature engineering pipeline...")

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    for path, label in [(reviews_path, "Reviews"), (products_path, "Products")]:
        if not os.path.exists(path):
            logger.error(f"{label} input file not found: {path}")
            raise FileNotFoundError(f"{label} input file not found: {path}")

    try:
        # --------------------------------------------------------------
        # Load reviews (memory-efficient: only keep needed columns)
        # --------------------------------------------------------------
        logger.info(f"Loading reviews to compute variance from {reviews_path}...")
        variance_chunks = []
        for chunk in pd.read_json(reviews_path, lines=True, chunksize=50_000):
            if "product_id" in chunk.columns and "rating" in chunk.columns:
                variance_chunks.append(chunk[["product_id", "rating"]])
                
        if not variance_chunks:
            raise ValueError("No valid product_id or rating columns found in reviews.")
            
        reviews_minimal_df = pd.concat(variance_chunks, ignore_index=True)
        logger.info(f"Extracted {len(reviews_minimal_df)} review ratings.")

        # --------------------------------------------------------------
        # Compute rating variance
        # --------------------------------------------------------------
        rating_variance_df = compute_rating_variance(reviews_minimal_df)
        del reviews_minimal_df  # Free memory ASAP
        
        logger.info(f"Computed rating_variance for {len(rating_variance_df)} products.")
        logger.info(
            "rating_variance summary — mean: %.4f, median: %.4f, max: %.4f",
            float(rating_variance_df["rating_variance"].mean()),
            float(rating_variance_df["rating_variance"].median()),
            float(rating_variance_df["rating_variance"].max()),
        )

        # --------------------------------------------------------------
        # Load preprocessed products, merge, and save in CHUNKS
        # --------------------------------------------------------------
        logger.info(f"Processing products in chunks from {products_path}...")
        ensure_output_dir(product_output_path)
        
        # Write new product features to temp file to preserve existing for merge.
        temp_product_output = product_output_path + ".new.tmp"
        first_chunk = True
        n_matched = 0
        n_total_products = 0
        
        for chunk in pd.read_json(products_path, lines=True, chunksize=10_000):
            n_total_products += len(chunk)
            chunk_featured = chunk.merge(rating_variance_df, on="product_id", how="left")
            chunk_featured["rating_variance"] = chunk_featured["rating_variance"].fillna(0.0)
            
            n_matched += chunk_featured["rating_variance"].gt(0).sum()
            
            # Write chunk to temp file
            mode = 'w' if first_chunk else 'a'
            chunk_featured.to_json(temp_product_output, orient="records", lines=True, mode=mode)
            first_chunk = False

        logger.info(
            f"Merged rating_variance onto products — "
            f"{n_matched}/{n_total_products} products had review data."
        )

        # --- Incremental merge: product featured output ---
        if os.path.exists(product_output_path) and os.path.getsize(product_output_path) > 0:
            from src.incremental import merge_jsonl
            merge_stats = merge_jsonl(
                temp_product_output, product_output_path, key_cols=["product_id"]
            )
            os.remove(temp_product_output)
            logger.info("Product incremental merge stats: %s", merge_stats)
        else:
            os.replace(temp_product_output, product_output_path)
        logger.info(f"Saved enriched products to {product_output_path}")

        # --------------------------------------------------------------
        # Save review_featured.jsonl (pass-through copy)
        # No incremental merge needed here — the merge already happened
        # at the preprocessing step. This file is just a copy.
        # --------------------------------------------------------------
        logger.info(f"Copying reviews to {review_output_path} (pass-through)...")
        ensure_output_dir(review_output_path)
        shutil.copyfile(reviews_path, review_output_path)
        logger.info(f"Saved reviews copy to {review_output_path}")

    except Exception as e:
        logger.error(f"Failed to process review features: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    run_review_features(
        reviews_path=os.path.join(BASE_DIR, "data/processed/review_preprocessed.jsonl"),
        products_path=os.path.join(BASE_DIR, "data/processed/product_preprocessed.jsonl"),
        product_output_path=os.path.join(BASE_DIR, "data/features/product_featured.jsonl"),
        review_output_path=os.path.join(BASE_DIR, "data/features/review_featured.jsonl"),
    )