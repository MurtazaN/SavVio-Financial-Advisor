# Deployment

This directory contains deployment scripts, configurations, and monitoring setup for the Purchase Guardrail Agent.

## Structure

- `scripts/` - Deployment automation scripts (Terraform, Cloud Build, etc.)
- `config/` - Deployment configuration files
- `monitoring/` - Monitoring and alerting scripts
- `docker/` - Dockerfiles for containerization

## Deployment Strategy

### Cloud Deployment on Google Cloud Platform (GCP)

The model is deployed exclusively on Google Cloud Platform using the following GCP services:
- **Cloud Run**: Serverless containerized deployment for the API and frontend
- **Vertex AI**: Model serving, versioning, and model registry
- **Cloud Storage**: Model artifact storage and DVC remote storage
- **Cloud Monitoring**: Performance and health monitoring
- **Cloud Logging**: Centralized logging and log analysis
- **Cloud Build**: CI/CD pipeline for automated builds and deployments
- **Artifact Registry**: Container image storage and versioning
- **Cloud SQL**: Managed PostgreSQL database for production

### Deployment Automation

- **Terraform**: Infrastructure as Code for provisioning GCP resources
- **Cloud Build**: GCP-native CI/CD pipeline for automated deployment
- **GitHub Actions**: Triggers Cloud Build on code commits and model registry updates

## Deployment Steps

1. **Build Docker image**:
   ```bash
   docker build -f docker/Dockerfile.api -t pga-api:latest .
   ```

2. **Deploy to GCP Cloud Run** (using gcloud CLI):
   ```bash
   # Set your GCP project
   gcloud config set project YOUR_GCP_PROJECT_ID
   
   # Build and push image to GCP Artifact Registry
   gcloud builds submit --tag gcr.io/YOUR_GCP_PROJECT_ID/pga-api:latest
   
   # Deploy to Cloud Run
   gcloud run deploy pga-api \
     --image gcr.io/YOUR_GCP_PROJECT_ID/pga-api:latest \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

3. **Or use Terraform for GCP infrastructure**:
   ```bash
   cd infrastructure/terraform
   # Initialize Terraform with GCP backend
   terraform init
   # Review planned changes
   terraform plan
   # Apply infrastructure changes to GCP
   terraform apply
   ```

## Monitoring

### Model Monitoring
- **Performance Metrics**: Accuracy, precision, recall, F1-score over time
- **Data Drift Detection**: Monitor input data distribution changes
- **Model Decay Detection**: Track model performance degradation

### System Monitoring
- **API Latency**: Response time monitoring
- **Error Rates**: Track API errors and failures
- **Resource Usage**: CPU, memory, request rates

### Alerting
- Email/Slack notifications for:
  - Model performance below threshold
  - Data drift detected
  - System errors
  - Retraining triggered

## Automated Retraining

When model decay or data drift is detected:
1. System automatically triggers retraining pipeline
2. Pulls latest data from data pipeline
3. Retrains model with new data
4. Validates new model performance
5. Deploys if performance improved, otherwise maintains current model

## Configuration

Edit `config/deployment_config.yaml` to configure:
- Deployment platform settings
- Monitoring thresholds
- Retraining triggers
- Alert configurations
