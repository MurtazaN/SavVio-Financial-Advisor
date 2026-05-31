"""
SavVio Model Wrapper — Bundles classifier + feature pipeline + label encoder
into a single MLflow pyfunc artifact.

Computes affordability features, preprocesses, predicts, and decodes the label.
"""

import joblib
import pandas as pd
import mlflow.pyfunc


class SavVioModelWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.pipeline = joblib.load(context.artifacts["feature_pipeline"])
        self.encoder = joblib.load(context.artifacts["label_encoder"])
        self.model = joblib.load(context.artifacts["classifier"])

    def predict(self, context, model_input, params=None):
        """Raw user + product data in → label, confidence, financial features out.

        Computes affordability features internally. Product/review computed
        features are expected as input columns (computed by API from DB data).
        """
        from features.financial_features import compute_affordability

        row = model_input.iloc[0]
        product_price = float(row.get("product_price", 0) or 0)

        # Compute affordability
        user_profile = {k: row.get(k) for k in [
            "monthly_income", "monthly_expenses", "savings_balance",
            "has_loan", "loan_amount", "monthly_emi", "loan_interest_rate",
            "loan_term_months", "credit_score", "liquid_savings",
            "discretionary_income", "debt_to_income_ratio",
        ]}
        aff = compute_affordability(user_financial_profile=user_profile, product_price=product_price)
        aff_dict = aff.to_dict()
        aff_dict.pop("affordability_score_unreliable", None)

        # Merge affordability into input, add downgraded=0, run pipeline+model+encoder
        feature_row = dict(row)
        feature_row.update(aff_dict)
        feature_row["downgraded"] = 0

        df = pd.DataFrame([feature_row])
        X = self.pipeline.transform(df)
        # Reorder columns to match the order the model was trained on
        # (must happen AFTER pipeline — NumericScaler needs product_price,
        #  FeatureDropper removes it, then we align to model's feature order)
        trained_cols = self.model.get_booster().feature_names
        X = X[[c for c in trained_cols if c in X.columns]]
        pred = self.model.predict(X)
        proba = self.model.predict_proba(X)

        return pd.DataFrame({
            "label": self.encoder.inverse_transform(pred),
            "confidence": proba.max(axis=1),
            "affordability_score": [float(aff_dict.get("affordability_score", 0) or 0)],
            "affordability_score_unreliable": [bool(aff.affordability_score_unreliable)],
            "price_to_income_ratio": [float(aff_dict.get("price_to_income_ratio", 0) or 0)],
            "residual_utility_score": [float(aff_dict.get("residual_utility_score", 0) or 0)],
            "savings_to_price_ratio": [float(aff_dict.get("savings_to_price_ratio", 0) or 0)],
            "net_worth_indicator": [float(aff_dict.get("net_worth_indicator", 0) or 0)],
            "credit_risk_indicator": [float(aff_dict.get("credit_risk_indicator", 0) or 0)],
        })
