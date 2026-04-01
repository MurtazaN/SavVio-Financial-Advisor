# SavVio

An AI-driven financial advocacy tool designed to bridge the gap between e-commerce and personal finance. SavVio serves as a "Financial Fiduciary" that evaluates whether a user should make a purchase based on their real-time financial health and the product's actual utility.

## Project Overview

The SavVio is an MLOps project that will integrate real-time product data with sensitive financial streams to provide responsible, conversational shopping guidance. Unlike traditional shopping assistants that focus on maximizing conversion, SavVio will evaluate purchases based on:

- **Financial Health**: User's income, expenses, savings, and debt obligations
- **Product Utility**: Analysis of product specifications and real-world usefulness
- **Affordability Metrics**: Calculated discretionary budget and residual utility

The system will provide **Green/Yellow/Red** light recommendations before users complete their purchase.

**Note**: This is the initial project setup phase. The project structure and documentation will evolve as development progresses.

## Team Members

- Murtaza Nipplewala
- Niraj Mehta
- Pranathi Bombay
- Rishabh Joshi
- Sanjana Patnam
- Wen-Hsin Su

## Project Structure

```
SavVio/
│
├── data_pipeline/              # Data ingestion and preprocessing pipeline
│   ├── dags/                   # Airflow DAG definitions
│   ├── data/                   # Data storage (versioned with DVC)
│   │   ├── raw/                # Raw data from sources
│   │   ├── processed/          # Processed/cleaned data
│   │   └── validated/          # Validated data ready for model training
│   ├── scripts/                # Data processing scripts
│   ├── tests/                  # Unit tests for data pipeline
│   ├── logs/                   # Pipeline execution logs
│   └── config/                 # Pipeline configuration files
│
├── model_pipeline/          # ML model development and training
│   ├── src/                    # Source code for model training
│   ├── notebooks/              # Jupyter notebooks for experimentation
│   ├── models/                 # Trained model artifacts
│   ├── experiments/            # Experiment tracking outputs (MLflow)
│   ├── config/                 # Model configuration files
│   └── tests/                  # Model testing scripts
│
├── deployment/                 # Model deployment and serving
│   ├── scripts/                # Deployment automation scripts
│   ├── config/                 # Deployment configuration
│   ├── monitoring/             # Monitoring and alerting scripts
│   └── docker/                 # Dockerfiles for containerization
│
├── frontend/                   # Web application interface
│   ├── src/                    # Frontend source code (Streamlit)
│   ├── static/                 # Static assets
│   └── templates/              # HTML templates (if needed)
│
├── infrastructure/             # Infrastructure as Code
│   ├── terraform/              # Terraform configurations
│   ├── kubernetes/             # Kubernetes manifests
│   └── cloud-build/            # Cloud Build configurations
│
├── docs/                       # Documentation
│   ├── api/                    # API documentation
│   ├── architecture/           # System architecture diagrams
│   └── user-guide/             # User guides
│
├── .github/                    # GitHub configurations
│   └── workflows/              # CI/CD pipeline definitions
│
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Getting Started

### Tools and Services

- Python 3.12+
- GitHub Actions
- Docker
- Apache Airflow (for data pipeline orchestration)
- MLflow
- LLM Models - GPT 4.1, Claude 4.5, Gemini 3
- DVC (Data Version Control) for data versioning
- PostgreSQL with pgvector extension or Vertex AI Vector Search
- Google Cloud Platform (GCP):
      Cloud Run, Cloud Storage, BigQuery, Vertex AI, Cloud Build, Cloud Monitoring, Cloud Logging, GCP Billing & Quotas
- Great Expectations for Data Quality Monitoring
- Evidently AI for Model Drift Monitoring
- Prometheus/Grafana for Latency & Throughput Monitoring

### Installation Instructions

Installation instructions and setup guides will be added as the project development progresses. The project will include:

- Python dependencies (requirements.txt)
- Environment configuration files
- Docker Image and compose files
- GCP deployment configurations
- Data pipeline setup with Airflow
- Model development environment setup

## Dataset Information

### Dataset Overview

SavVio relies on two primary categories of data:

   1. For the scope of this project, data will be sourced as follows: 
Financial Data: User financial data is sourced from the Personal Finance ML Dataset (Kaggle), which contains synthetic records of income, expenses, savings, and financial behavior. This dataset is used to simulate different financial situations and affordability scenarios.

   2. Product Data: Product data is sourced from the Amazon Products Dataset (Kaggle), which includes product names, categories, prices, and basic descriptions. This dataset is used to represent realistic product choices during purchase evaluation.


### Data Sources

- **Financial Data**: https://www.kaggle.com/datasets/miadul/personal-finance-ml-dataset
- **Product Data**: https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset

**Note**: All datasets are publicly available, read-only, and used strictly for academic experimentation and evaluation


### Data Privacy

- No real personally identifiable information (PII) is used
- All financial data is synthetic or anonymized
- User identifiers are masked or hashed before storage
- Financial snapshots are encrypted
- System operates in read-only mode

## Planned Usage

### Data Pipeline

The data pipeline will be orchestrated using Apache Airflow:
- Airflow webserver will run on port 8080
- DAGs will handle data ingestion, preprocessing, validation, and versioning
- Data will be versioned using DVC with GCP Cloud Storage backend

### Model Development

Model training and development will include:
- Loading versioned data from the data pipeline
- Training baseline models (Logistic Regression, XGBoost)
- Experiment tracking with MLflow
- Bias detection and mitigation
- Model validation and selection

### Running the Application

Once implemented:
- **Backend API**: Will run on port 4000 (FastAPI)
- **Frontend**: Will run on port 3000 (Streamlit)

### Deployment

Deployment will be handled on Google Cloud Platform (GCP):
- Cloud Run for serverless containerized deployment
- Vertex AI for model serving and versioning
- Cloud Storage for data and model artifacts
- See `deployment/README.md` for detailed deployment plans (to be added)

## Planned Testing

Testing will be implemented for each component:
- Unit tests for data pipeline components
- Model development tests
- Integration tests
- End-to-end testing

## Planned Monitoring

The system will include monitoring for:

- **Data Quality**: Schema validation, anomaly detection, data drift
- **Model Performance**: Accuracy, precision, recall, F1-score
- **Bias Detection**: Performance across different data slices
- **System Health**: API latency, error rates, resource usage

Monitoring will use:
- MLflow for experiment tracking
- GCP Cloud Monitoring for production monitoring
- Prometheus/Grafana for latency and throughput monitoring

## Configuration

Configuration files will be added as the project develops:

- `data_pipeline/config/pipeline_config.yaml` - Data pipeline settings (to be added)
- `model_pipeline/config/training_config.yaml` - Model training parameters (to be added)
- `deployment/config/deployment_config.yaml` - Deployment settings (to be added)
- Environment variables and secrets management (to be added)

## Technology Stack

| Category | Tools Used |
|----------|-----------|
| Cloud Platform | Google Cloud Platform (GCP) - Cloud Run, Cloud Storage, BigQuery, Vertex AI, Cloud Build, Cloud Monitoring, Cloud Logging, GCP Billing & Quotas |
| Backend Framework | FastAPI (Python) |
| Frontend Interface | Streamlit (Web Application) |
| Data Orchestration | Apache Airflow |
| Data Versioning | DVC (Data Version Control) with GCP Cloud Storage backend |
| Decision Engine | Python (Deterministic rule-based financial logic) |
| Machine Learning Models | Scikit-learn (Logistic Regression), XGBoost |
| LLM Models | GPT 4.1, Claude 4.5, Gemini 3 |
| Safety & Guardrails | Vertex AI Safety Filters, NVIDIA NeMo Guardrails |
| Database (Relational) | PostgreSQL |
| Database (Vector) | pgvector (extension for PostgreSQL) or Vertex AI Vector Search |
| Model Tracking | MLflow |
| Model Registry | GCP Artifact Registry, Vertex AI Model Registry |
| Data Quality Monitoring | Great Expectations (triggered via Airflow) |
| Model Drift Monitoring | Evidently AI (Open Source), Vertex AI Model Monitoring |
| Latency & Throughput Monitoring | Prometheus, Grafana, GCP Cloud Monitoring |
| Logging & Containerization | Cloud Logging, Docker |
| CI/CD | GitHub Actions |
| Deployment | Cloud Run (Serverless Containers) |

## Planned Features

- **Data Pipeline**: Automated data ingestion, preprocessing, validation, and versioning using Airflow
- **Bias Detection**: Data slicing and bias mitigation across demographic groups
- **Model Registry**: Version-controlled model storage and management on GCP Artifact Registry/Vertex AI
- **CI/CD Automation**: Automated training, validation, and deployment pipelines using GitHub Actions and GCP Cloud Build
- **Monitoring & Alerts**: Real-time monitoring of model performance and data drift using GCP Cloud Monitoring
- **Automated Retraining**: Trigger retraining when model decay or data drift is detected

## Metrics and Objectives

### Metrics

- **Residual Utility Score (RUS)**: Impact of purchase on emergency fund/savings
- **Friction Success Rate**: Percentage of "Red Light" recommendations that prevented impulse purchases
- **Categorization Accuracy**: Ability to correctly identify recurring bills and essential spending

### Objectives

- Calculate "break-even point" for high-utility items
- Promote long-term user financial health
- Provide transparent, responsible purchase recommendations

## Security & Privacy

The following security measures are planned:
- All financial data will be encrypted at rest
- User identifiers will be masked/hashed
- Read-only data access
- No transactional capabilities
- Compliance with data privacy principles

## Documentation

Documentation will be added as the project progresses:
- API Documentation (to be added)
- Architecture Diagrams (to be added)
- User Guide (to be added)

---

**Note**: This is an academic project. All data is synthetic or publicly available. No real user financial information is processed. This README reflects the initial project setup and will be updated as development progresses.
