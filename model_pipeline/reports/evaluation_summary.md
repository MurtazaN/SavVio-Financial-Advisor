# Evaluation Summary

Generated at: 2026-03-26 14:36:07

## Baseline Models (Validation Metrics)

### xgboost
- run_id: 189218a2c99644249ce3e0c1edbc04a5
- accuracy: 0.9979
- f1_score: 0.9979
- roc_auc: 0.9999
- pr_auc: 0.9998

### lightgbm
- run_id: 43a73410d8e24a4d90c1356123ae9f6f
- accuracy: 0.9977
- f1_score: 0.9977
- roc_auc: 1.0
- pr_auc: 0.9999

### xgb_linear
- run_id: 8980d5ab6a43497fa976ed9d1c1d30f6
- accuracy: 0.9027
- f1_score: 0.8996
- roc_auc: 0.9836
- pr_auc: 0.9614

## Champion Model (Final Test Metrics)

- model_name: xgboost_tuned
- source_run_id: a4333656468846649f074bb40974a630
- accuracy: 0.9972
- f1_score: 0.9972
- roc_auc: 1.0
- pr_auc: 1.0

## Hyperparameter Sensitivity (Tuned Champion)

- study_name: xgboost_tuning
- completed_trials: 50
- top_hyperparameters:
  - n_estimators: 0.390676
  - reg_lambda: 0.161187
  - max_depth: 0.158082
  - reg_alpha: 0.090782
  - learning_rate: 0.085652
- artifact_paths:
  - /app/reports/sensitivity/xgboost_tuned_param_importance.json
  - /app/reports/sensitivity/xgboost_tuned_param_importance.png
  - /app/reports/sensitivity/xgboost_tuned_param_sensitivity.png
