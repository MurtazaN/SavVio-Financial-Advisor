"""Shared preprocessing pipeline components for model training and inference."""

import os
import logging

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from config import Config

logger = logging.getLogger(__name__)


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """Impute missing values using project-specific defaults."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        for col in Config.FINANCIAL_FEATURES:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        for col in Config.PRODUCT_FEATURES:
            source_col = col
            if col == "price" and "product_price" in df.columns and "price" not in df.columns:
                source_col = "product_price"
            if source_col in df.columns and df[source_col].isnull().any():
                if source_col == "rating_variance":
                    df[source_col] = df[source_col].fillna(0.0)
                else:
                    df[source_col] = df[source_col].fillna(df[source_col].median())

        computed_features = [
            "affordability_score", "price_to_income_ratio", "residual_utility_score",
            "savings_to_price_ratio", "net_worth_indicator", "credit_risk_indicator",
        ] + Config.PRODUCT_COMPUTED_FEATURES + Config.REVIEW_COMPUTED_FEATURES
        for col in computed_features:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        for col in Config.CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")

        return df


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Ordinal-encode categorical columns, mapping unknown categories to -1."""

    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.existing_cat_cols = []

    def fit(self, X: pd.DataFrame, y=None):
        self.existing_cat_cols = [c for c in Config.CATEGORICAL_FEATURES if c in X.columns]
        if self.existing_cat_cols:
            self.encoder.fit(X[self.existing_cat_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        cols = [c for c in self.existing_cat_cols if c in df.columns]
        if cols:
            df[cols] = self.encoder.transform(df[cols])
        return df


class NumericScaler(BaseEstimator, TransformerMixin):
    """Scale numeric features to zero mean and unit variance."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_cols = []

    def fit(self, X: pd.DataFrame, y=None):
        numeric_candidates = []
        for col in Config.NUMERIC_FEATURES:
            if col in X.columns:
                numeric_candidates.append(col)
            elif col == "price" and "product_price" in X.columns:
                numeric_candidates.append("product_price")
        self.numeric_cols = numeric_candidates
        if self.numeric_cols:
            self.scaler.fit(X[self.numeric_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        cols = [c for c in self.numeric_cols if c in df.columns]
        if cols:
            df[cols] = self.scaler.transform(df[cols])
        return df


class FeatureDropper(BaseEstimator, TransformerMixin):
    """Drop non-model columns such as IDs, free text, and labels."""

    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        cols_to_drop = [c for c in Config.COLUMNS_TO_DROP + [Config.LABEL_COL] if c in df.columns]
        if "product_price" in df.columns:
            cols_to_drop.append("product_price")
        return df.drop(columns=cols_to_drop, errors="ignore")


class FeaturePipeline:
    """Unified preprocessing pipeline persisted as a single artifact."""

    def __init__(self):
        self.pipeline = Pipeline([
            ("imputer", MissingValueImputer()),
            ("encoder", CategoricalEncoder()),
            ("scaler", NumericScaler()),
            ("dropper", FeatureDropper()),
        ])

    def save(self, path=None):
        path = path or os.path.join(Config.MODEL_SAVE_DIR, "feature_pipeline.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info("Feature pipeline saved to %s", path)

    def load(self, path=None):
        path = path or os.path.join(Config.MODEL_SAVE_DIR, "feature_pipeline.pkl")
        self.pipeline = joblib.load(path)
        logger.info("Feature pipeline loaded from %s", path)

    def fit_transform(self, X: pd.DataFrame, save_pipeline: bool = True) -> pd.DataFrame:
        transformed = self.pipeline.fit_transform(X)
        if save_pipeline:
            self.save()
        return transformed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.load()
        return self.pipeline.transform(X)
