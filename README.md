# News Classification MLOps Demo

This project demonstrates a simple MLOps workflow for training and serving a news classifier.  
It uses **Kubeflow Pipelines** for orchestration, **MLflow** for experiment tracking, **MinIO** for
artifact storage, **BentoML** for model serving and a **Streamlit** front‑end.

## Repository Layout

```
front_end/               # Streamlit UI
kubeflow_pipeline/       # Kubeflow components and pipeline definition
minio/                   # MinIO manifests
mlflow-yamls/            # MLflow + PostgreSQL manifests
serving/                 # BentoML service manifests
trigger/                 # CronJob to trigger pipeline runs
ingress/                 # Ingress rules
news_pipeline.yaml       # Compiled pipeline
port-forward-all.sh      # Helper to port-forward services
```

## Prerequisites

- Docker with access to the local Minikube registry
- Minikube and kubectl
- python, kfp, bentoml packages

## Cluster Setup

Start Minikube and create the namespaces used by the application:

```bash
minikube start
kubectl create namespace mlops
kubectl create namespace kubeflow
kubectl create namespace mlflow
```

### Deploy supporting services

```bash
# MinIO
kubectl apply -f minio/minio-deployment.yaml
kubectl apply -f minio/minio-namespace.yaml
kubectl apply -f minio/minio-pvc.yaml
kubectl apply -f minio/minio-secret.yaml
kubectl apply -f minio/minio-service.yaml

# MLflow & PostgreSQL
kubectl apply -f mlflow-yamls/postgresql-deployment.yaml
kubectl apply -f mlflow-yamls/mlflow-deployment.yaml
```

### Install Kubeflow Pipelines

```bash
export PIPELINE_VERSION=2.4.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
```

## Build Container Images

Use the Minikube Docker daemon so the images are available to the cluster:

```bash
eval $(minikube docker-env)

# Streamlit front end
docker build -t streamlit-news-app:latest front_end

# Pipeline trigger job
docker build -t trigger-pipeline:latest trigger

# BentoML service
bentoml build serving/service.py -n news_classifier:latest
bentoml build serving
bentoml containerize news_classifier:latest -t news_classifier:latest
```

## Deploy Application Components

```bash
# BentoML deployment
kubectl apply -f serving/bentoml-pvc.yaml
kubectl apply -f serving/bentoml-deployment.yaml
kubectl apply -f serving/bento-service.yaml

# Streamlit UI
kubectl apply -f front_end/streamlit-app.yaml

# Scheduled pipeline trigger
kubectl apply -f trigger/cronjob.yaml

# Ingress rules
kubectl apply -f ingress/mlops-ingress.yaml
kubectl apply -f ingress/kubeflow-ingress.yaml
```

## Running the Pipeline

The Kubeflow pipeline definition is compiled into `news_pipeline.yaml`. You can trigger
runs manually or via the CronJob.  Secrets with API keys are expected in the
`pipeline-trigger-secrets` secret, for example:

```bash
kubectl -n kubeflow create secret generic pipeline-trigger-secrets \
  --from-literal=NEWS_API_KEY=<news_api_key> \
  --from-literal=MINIO_ENDPOINT_KUBEFLOW=http://minio.mlops.svc.cluster.local:9000 \
  --from-literal=GEMINI_API_KEY=<gemini_api_key> \
  --from-literal=MEDIASTACK_ACCESS_KEY=<mediastack_key> \
  --from-literal=NEWSAPI_API_KEY=<newsapi_key> \
  --from-literal=GNEWS_API_KEY=<gnews_key> \
  --from-literal=KFP_HOST=http://ml-pipeline.kubeflow.svc.cluster.local:8888
```

You can also trigger a run locally using `makerun.py` once the services are
running and port-forwarded.

## Accessing the Services

Use the provided script to port‑forward all services to your localhost:

```bash
./port-forward-all.sh
```

After port-forwarding:

- MLflow UI: <http://localhost:5000>
- MinIO console: <http://localhost:9001>
- Streamlit app: <http://localhost:8900>
- BentoML service: <http://localhost:10000>
- Kubeflow Pipelines UI: <http://localhost:3000>

