from __future__ import annotations

import sys
from pathlib import Path
from src.ingestion.run_ingestion import ingest_financial_task, ingest_product_task, ingest_review_task
from src.preprocess.run_preprocessing import preprocess_financial_task, preprocess_product_task, preprocess_review_task
from src.features.run_features import feature_financial_task, feature_review_task
from src.database.run_database import setup_database_task, load_financial_task, load_products_task, load_reviews_task, generate_and_load_embedding_task
from src.validation.run_validation import validate_raw, validate_processed, validate_features, validate_raw_anomalies

from datetime import datetime, timedelta
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
# from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator  # TODO: uncomment when Slack is configured
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
# from airflow.task.trigger_rule import TriggerRule new one, need to verify


# ==================== BRANCHING HELPER ====================
# Generic factory: returns a callable that checks upstream task states.
# If ALL upstream tasks succeeded → route to success_ids.
# If ANY upstream task failed  → route to failure_ids.
# If ANY upstream task was skipped/upstream_failed (cascade from earlier branch)
#   → return [] to skip all downstream (no duplicate alerts).
#
# The BranchPythonOperator uses trigger_rule=ALL_DONE so it always runs,
# even when some upstream tasks fail.  The branching logic inside decides
# whether to continue, alert, or silently stop.

def make_branch_check(upstream_ids, success_ids, failure_ids):
    """
    Create a branch callable for BranchPythonOperator.

    Args:
        upstream_ids:  task_id(s) to inspect
        success_ids:   task_id(s) to run when all upstreams succeeded
        failure_ids:   task_id(s) to run when any upstream failed
    """
    def _check(**context):
        import logging
        from airflow.sdk.execution_time.task_runner import RuntimeTaskInstance

        logger = logging.getLogger(__name__)
        dag_run = context["dag_run"]

        try:
            # Airflow 3.x SDK — returns nested dict: {run_id: {task_id: TaskInstanceState}}
            all_states = RuntimeTaskInstance.get_task_states(
                dag_id=dag_run.dag_id,
                run_ids=[dag_run.run_id],
                task_ids=upstream_ids,
            )
            # Extract the inner dict for our specific run
            states = all_states.get(dag_run.run_id, {})
        except Exception as e:
            logger.error("get_task_states failed: %s — routing to failure", e, exc_info=True)
            return failure_ids

        logger.info("Branch check — upstream states: %s", states)

        for tid in upstream_ids:
            state = str(states.get(tid, ""))

            # Upstream was skipped by a previous branch — stop cascade silently
            if state in ("skipped", "upstream_failed", ""):
                logger.info("Task %s is %s — skipping cascade (no alert)", tid, state)
                return []  # skip ALL downstream tasks

            # Upstream actually ran and failed — route to error alerts
            if state != "success":
                logger.info("Task %s is %s — routing to failure alerts", tid, state)
                return failure_ids

        logger.info("All upstream tasks succeeded — continuing pipeline")
        return success_ids

    return _check


# ---------- Constants ----------
ALERT_EMAIL = "murtaza.sn786@gmail.com"
# SLACK_CHANNEL = "#group-34"  # TODO: uncomment when Slack is configured

# ---------- Default args ----------
default_args = {
    'owner': 'Murtaza Nipplewala',
    'start_date': datetime(2026, 2, 21),
    'retries': 0,
    'retry_delay': timedelta(minutes=0.3),
    'email': 'murtaza.sn786@gmail.com',
    'email_on_failure': True,
    'email_on_retry': False,
}

# ---------- DAG ----------
dag = DAG(
    dag_id="Data_pipeline_airflow",
    default_args=default_args,
    description="Data Pipeline Airflow DAG",
    schedule="@daily",
    catchup=False,
    tags=["example"],
    owner_links={"Murtaza Nipplewala": "https://github.com/MurtazaNipplewala/SavVio"},
    max_active_runs=1,
)


# ==================== ERROR ALERT OPERATORS ====================
# Each pair (email + slack) is a terminal leaf — pipeline stops here.

# --- Ingestion Errors ---
email_error_at_ingestion = EmailOperator(
    task_id="send_email_at_ingestion_error",
    to=ALERT_EMAIL,
    cc="nirajmehta960@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Ingestion",
    html_content="<p>Something went wrong at Ingestion stage.</p>",
    dag=dag,
)

# slack_error_at_ingestion = SlackWebhookOperator(
#     task_id="send_slack_at_ingestion_error",
#     slack_webhook_conn_id="slack_webhook",
#     message=":red_circle: *SavVio Data Pipeline* — Error at *Ingestion* stage.",
#     channel=SLACK_CHANNEL,
#     dag=dag,
# )

# --- Raw Validation Errors ---
email_error_at_raw_validation = EmailOperator(
    task_id="send_email_at_raw_validation_error",
    to=ALERT_EMAIL,
    cc="nirajmehta960@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Raw Validation",
    html_content="<p>Something went wrong at Raw Data Validation stage.</p>",
    dag=dag,
)

# slack_error_at_raw_validation = SlackWebhookOperator(
#     task_id="send_slack_at_raw_validation_error",
#     slack_webhook_conn_id="slack_webhook",
#     message=":red_circle: *SavVio Data Pipeline* — Error at *Raw Validation* stage.",
#     channel=SLACK_CHANNEL,
#     dag=dag,
# )

# --- Preprocessing Errors ---
email_error_at_preprocessing = EmailOperator(
    task_id="send_email_at_preprocessing_error",
    to=ALERT_EMAIL,
    cc="nirajmehta960@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Preprocessing",
    html_content="<p>Something went wrong at Preprocessing stage.</p>",
    dag=dag,
)

# slack_error_at_preprocessing = SlackWebhookOperator(
#     task_id="send_slack_at_preprocessing_error",
#     slack_webhook_conn_id="slack_webhook",
#     message=":red_circle: *SavVio Data Pipeline* — Error at *Preprocessing* stage.",
#     channel=SLACK_CHANNEL,
#     dag=dag,
# )

# --- Processed Validation Errors ---
email_error_at_processed_validation = EmailOperator(
    task_id="send_email_at_processed_validation_error",
    to=ALERT_EMAIL,
    cc="nirajmehta960@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Processed Validation",
    html_content="<p>Something went wrong at Processed Data Validation stage.</p>",
    dag=dag,
)

# slack_error_at_processed_validation = SlackWebhookOperator(
#     task_id="send_slack_at_processed_validation_error",
#     slack_webhook_conn_id="slack_webhook",
#     message=":red_circle: *SavVio Data Pipeline* — Error at *Processed Validation* stage.",
#     channel=SLACK_CHANNEL,
#     dag=dag,
# )

# --- Feature Engineering Errors ---
email_error_at_feature_engineering = EmailOperator(
    task_id="send_email_at_feature_engineering_error",
    to=ALERT_EMAIL,
    cc="nirajmehta960@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Feature Engineering",
    html_content="<p>Something went wrong at Feature Engineering stage.</p>",
    dag=dag,
)

# slack_error_at_feature_engineering = SlackWebhookOperator(
#     task_id="send_slack_at_feature_engineering_error",
#     slack_webhook_conn_id="slack_webhook",
#     message=":red_circle: *SavVio Data Pipeline* — Error at *Feature Engineering* stage.",
#     channel=SLACK_CHANNEL,
#     dag=dag,
# )

# --- Featured Validation Errors ---
email_error_at_featured_validation = EmailOperator(
    task_id="send_email_at_featured_validation_error",
    to=ALERT_EMAIL,
    cc="nirajmehta960@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at Featured Validation",
    html_content="<p>Something went wrong at Featured Data Validation stage.</p>",
    dag=dag,
)

# slack_error_at_featured_validation = SlackWebhookOperator(
#     task_id="send_slack_at_featured_validation_error",
#     slack_webhook_conn_id="slack_webhook",
#     message=":red_circle: *SavVio Data Pipeline* — Error at *Featured Validation* stage.",
#     channel=SLACK_CHANNEL,
#     dag=dag,
# )

# --- DB Loading Errors ---
email_error_at_DB_loading = EmailOperator(
    task_id="send_email_at_DB_loading_error",
    to=ALERT_EMAIL,
    cc="nirajmehta960@gmail.com",
    subject="SavVio Data Pipeline Airflow - Error at DB Loading",
    html_content="<p>Something went wrong at DB Loading stage.</p>",
    dag=dag,
)

# slack_error_at_DB_loading = SlackWebhookOperator(
#     task_id="send_slack_at_DB_loading_error",
#     slack_webhook_conn_id="slack_webhook",
#     message=":red_circle: *SavVio Data Pipeline* — Error at *DB Loading* stage.",
#     channel=SLACK_CHANNEL,
#     dag=dag,
# )

# --- Bias Analysis Errors (placeholder) ---
# email_error_at_bias_analysis = EmailOperator(
#     task_id="send_email_at_bias_analysis_error",
#     to=ALERT_EMAIL,
#     subject="SavVio Data Pipeline Airflow - Error at Bias Analysis",
#     html_content="<p>Something went wrong at Bias Analysis stage.</p>",
#     trigger_rule=TriggerRule.ONE_FAILED,
#     dag=dag,
# )

# slack_error_at_bias_analysis = SlackWebhookOperator(
#     task_id="send_slack_at_bias_analysis_error",
#     slack_webhook_conn_id="slack_webhook",
#     message=":red_circle: *SavVio Data Pipeline* — Error at *Bias Analysis* stage.",
#     channel=SLACK_CHANNEL,
#     trigger_rule=TriggerRule.ONE_FAILED,
#     dag=dag,
# )


# ==================== SUCCESS ALERT OPERATORS ====================

email_pipeline_success = EmailOperator(
    task_id="send_email_pipeline_success",
    to=ALERT_EMAIL,
    cc="nirajmehta960@gmail.com",
    subject="SavVio Data Pipeline Airflow - Pipeline Completed Successfully",
    html_content="<p>All data has been successfully ingested, validated, processed, featured, and loaded into the database.</p>",
    dag=dag,
)

# slack_pipeline_success = SlackWebhookOperator(
#     task_id="send_slack_pipeline_success",
#     slack_webhook_conn_id="slack_webhook",
#     message=":large_green_circle: *SavVio Data Pipeline* — Pipeline completed *successfully*. All data loaded into DB.",
#     channel=SLACK_CHANNEL,
#     dag=dag,
# )


# ═══════════════════════════════════════════════════════════════════
# 1. DATA INGESTION  (runs in PARALLEL)
# ═══════════════════════════════════════════════════════════════════

ingest_financial = PythonOperator(
    task_id='ingest_financial_data',
    python_callable=ingest_financial_task,
    dag=dag,
)

ingest_products = PythonOperator(
    task_id='ingest_product_data',
    python_callable=ingest_product_task,
    dag=dag,
)

ingest_reviews = PythonOperator(
    task_id='ingest_review_data',
    python_callable=ingest_review_task,
    dag=dag,
)

# --- BRANCH: all ingestion succeeded? ---
check_ingestion = BranchPythonOperator(
    task_id='check_ingestion',
    python_callable=make_branch_check(
        upstream_ids=['ingest_financial_data', 'ingest_product_data', 'ingest_review_data'],
        success_ids=['validate_raw_data', 'validate_raw_anomalies'],
        # failure_ids=['send_email_at_ingestion_error', 'send_slack_at_ingestion_error'],
        failure_ids=['send_email_at_ingestion_error'],
    ),
    trigger_rule=TriggerRule.ALL_DONE,   # branch must always run so it can inspect results and route to errors
    dag=dag,
)

[ingest_financial, ingest_products, ingest_reviews] >> check_ingestion
# check_ingestion >> [email_error_at_ingestion, slack_error_at_ingestion]
check_ingestion >> email_error_at_ingestion


# ═══════════════════════════════════════════════════════════════════
# 2. RAW DATA VALIDATION  (branched from ingestion success path)
# ═══════════════════════════════════════════════════════════════════

validate_raw_data = PythonOperator(
    task_id='validate_raw_data',
    python_callable=validate_raw,
    dag=dag,
)

validate_raw_anomaly = PythonOperator(
    task_id='validate_raw_anomalies',
    python_callable=validate_raw_anomalies,
    dag=dag,
)

check_ingestion >> [validate_raw_data, validate_raw_anomaly]

# --- BRANCH: raw validation succeeded? ---
check_raw_validation = BranchPythonOperator(
    task_id='check_raw_validation',
    python_callable=make_branch_check(
        upstream_ids=['validate_raw_data'],
        success_ids=['preprocess_financial_data', 'preprocess_product_data', 'preprocess_review_data'],
        # failure_ids=['send_email_at_raw_validation_error', 'send_slack_at_raw_validation_error'],
        failure_ids=['send_email_at_raw_validation_error'],
    ),
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

validate_raw_data >> check_raw_validation
# check_raw_validation >> [email_error_at_raw_validation, slack_error_at_raw_validation]
check_raw_validation >> email_error_at_raw_validation


# ═══════════════════════════════════════════════════════════════════
# 3. DATA PREPROCESSING  (branched from raw validation success path)
# ═══════════════════════════════════════════════════════════════════

preprocess_financial = PythonOperator(
    task_id='preprocess_financial_data',
    python_callable=preprocess_financial_task,
    dag=dag,
)

preprocess_products = PythonOperator(
    task_id='preprocess_product_data',
    python_callable=preprocess_product_task,
    dag=dag,
)

preprocess_reviews = PythonOperator(
    task_id='preprocess_review_data',
    python_callable=preprocess_review_task,
    dag=dag,
)

check_raw_validation >> [preprocess_financial, preprocess_products, preprocess_reviews]

# --- BRANCH: all preprocessing succeeded? ---
check_preprocessing = BranchPythonOperator(
    task_id='check_preprocessing',
    python_callable=make_branch_check(
        upstream_ids=['preprocess_financial_data', 'preprocess_product_data', 'preprocess_review_data'],
        success_ids=['validate_processed_data'],
        # failure_ids=['send_email_at_preprocessing_error', 'send_slack_at_preprocessing_error'],
        failure_ids=['send_email_at_preprocessing_error'],
    ),
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

[preprocess_financial, preprocess_products, preprocess_reviews] >> check_preprocessing
# check_preprocessing >> [email_error_at_preprocessing, slack_error_at_preprocessing]
check_preprocessing >> email_error_at_preprocessing


# ═══════════════════════════════════════════════════════════════════
# 4. PROCESSED DATA VALIDATION  (branched from preprocessing success)
# ═══════════════════════════════════════════════════════════════════

validate_processed_data = PythonOperator(
    task_id='validate_processed_data',
    python_callable=validate_processed,
    dag=dag,
)

check_preprocessing >> validate_processed_data

# --- BRANCH: processed validation succeeded? ---
check_processed_validation = BranchPythonOperator(
    task_id='check_processed_validation',
    python_callable=make_branch_check(
        upstream_ids=['validate_processed_data'],
        success_ids=['feature_financial_data', 'feature_review_data'],
        # failure_ids=['send_email_at_processed_validation_error', 'send_slack_at_processed_validation_error'],
        failure_ids=['send_email_at_processed_validation_error'],
    ),
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

validate_processed_data >> check_processed_validation
# check_processed_validation >> [email_error_at_processed_validation, slack_error_at_processed_validation]
check_processed_validation >> email_error_at_processed_validation


# ═══════════════════════════════════════════════════════════════════
# 5. FEATURE ENGINEERING  (branched from processed validation success)
# ═══════════════════════════════════════════════════════════════════

feature_financial = PythonOperator(
    task_id='feature_financial_data',
    python_callable=feature_financial_task,
    dag=dag,
)

feature_reviews = PythonOperator(
    task_id='feature_review_data',
    python_callable=feature_review_task,
    dag=dag,
)

check_processed_validation >> [feature_financial, feature_reviews]

# --- BRANCH: all feature engineering succeeded? ---
check_feature_engineering = BranchPythonOperator(
    task_id='check_feature_engineering',
    python_callable=make_branch_check(
        upstream_ids=['feature_financial_data', 'feature_review_data'],
        success_ids=['validate_featured_data'],
        # failure_ids=['send_email_at_feature_engineering_error', 'send_slack_at_feature_engineering_error'],
        failure_ids=['send_email_at_feature_engineering_error'],
    ),
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

[feature_financial, feature_reviews] >> check_feature_engineering
# check_feature_engineering >> [email_error_at_feature_engineering, slack_error_at_feature_engineering]
check_feature_engineering >> email_error_at_feature_engineering


# ═══════════════════════════════════════════════════════════════════
# 6. FEATURED DATA VALIDATION  (branched from feature eng success)
# ═══════════════════════════════════════════════════════════════════

validate_featured_data = PythonOperator(
    task_id='validate_featured_data',
    python_callable=validate_features,
    dag=dag,
)

check_feature_engineering >> validate_featured_data

# --- BRANCH: featured validation succeeded? ---
check_featured_validation = BranchPythonOperator(
    task_id='check_featured_validation',
    python_callable=make_branch_check(
        upstream_ids=['validate_featured_data'],
        success_ids=['setup_database'],
        # failure_ids=['send_email_at_featured_validation_error', 'send_slack_at_featured_validation_error'],
        failure_ids=['send_email_at_featured_validation_error'],
    ),
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

validate_featured_data >> check_featured_validation
# check_featured_validation >> [email_error_at_featured_validation, slack_error_at_featured_validation]
check_featured_validation >> email_error_at_featured_validation


# ═══════════════════════════════════════════════════════════════════
# 7. LOADING DATA INTO POSTGRESQL  (branched from featured validation)
# ═══════════════════════════════════════════════════════════════════

setup_database = PythonOperator(
    task_id='setup_database',
    python_callable=setup_database_task,
    dag=dag,
)

load_financial = PythonOperator(
    task_id='load_financial_profiles',
    python_callable=load_financial_task,
    dag=dag,
)

load_product = PythonOperator(
    task_id='load_products',
    python_callable=load_products_task,
    dag=dag,
)

load_review = PythonOperator(
    task_id='load_reviews',
    python_callable=load_reviews_task,
    dag=dag,
)

generate_load_embeddings = PythonOperator(
    task_id='generate_load_embeddings',
    python_callable=generate_and_load_embedding_task,
    dag=dag,
)

check_featured_validation >> setup_database
setup_database >> [load_financial, load_product] >> load_review >> generate_load_embeddings

# --- BRANCH: all DB loading succeeded? ---
check_db_loading = BranchPythonOperator(
    task_id='check_db_loading',
    python_callable=make_branch_check(
        upstream_ids=['load_financial_profiles', 'load_products', 'load_reviews', 'generate_load_embeddings'],
        # success_ids=['send_email_pipeline_success', 'send_slack_pipeline_success'],
        success_ids=['send_email_pipeline_success'],
        # failure_ids=['send_email_at_DB_loading_error', 'send_slack_at_DB_loading_error'],
        failure_ids=['send_email_at_DB_loading_error'],
    ),
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

generate_load_embeddings >> check_db_loading
# check_db_loading >> [email_error_at_DB_loading, slack_error_at_DB_loading]
check_db_loading >> email_error_at_DB_loading
# check_db_loading >> [email_pipeline_success, slack_pipeline_success]
check_db_loading >> email_pipeline_success

# ═══════════════════════════════════════════════════════════════════
# SENTINEL — marks the DAG run as failed if any stage was skipped/failed
# ═══════════════════════════════════════════════════════════════════
def _check_pipeline_complete(**context):
    """Fail the DAG run if the success email was not sent (i.e. pipeline didn't fully complete)."""
    import logging
    from airflow.sdk.execution_time.task_runner import RuntimeTaskInstance
    
    logger = logging.getLogger(__name__)
    dag_run = context['dag_run']
    
    try:
        # Airflow 3 SDK uses get_task_states instead of context['dag_run'].get_task_instance
        all_states = RuntimeTaskInstance.get_task_states(
            dag_id=dag_run.dag_id,
            run_ids=[dag_run.run_id],
            task_ids=['send_email_pipeline_success'],
        )
        states = all_states.get(dag_run.run_id, {})
    except Exception as e:
        logger.error("get_task_states failed: %s", e, exc_info=True)
        raise AirflowException(f"Failed to query task states: {e}")
        
    state = str(states.get('send_email_pipeline_success', 'missing'))
    
    # Handle string 'success' or TaskInstanceState enum strings
    if state not in ('success', 'TaskInstanceState.SUCCESS') and 'success' not in state.lower():
        raise AirflowException(
            f"Pipeline did not complete successfully — "
            f"'send_email_pipeline_success' state is '{state}'"
        )

pipeline_sentinel = PythonOperator(
    task_id='pipeline_sentinel',
    python_callable=_check_pipeline_complete,
    trigger_rule=TriggerRule.ALL_DONE,   # always runs, checks success email state
    dag=dag,
)

email_pipeline_success >> pipeline_sentinel


# ═══════════════════════════════════════════════════════════════════
# Airflow DAG for Bias Analysis (commented out)
# ═══════════════════════════════════════════════════════════════════

# bias_financial = PythonOperator(
#     task_id='bias_analysis_financial',
#     python_callable=bias.analyze_financial_bias,
#     dag=dag
# )

# bias_products = PythonOperator(
#     task_id='bias_analysis_products',
#     python_callable=bias.analyze_product_bias,
#     dag=dag
# )

# # Run in parallel since they analyze independent tracks.
# [bias_financial, bias_products] >> complete