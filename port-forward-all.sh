#!/bin/bash

minikube status | grep -q "Running" || minikube start

kubectl -n mlops port-forward svc/mlflow 5000:5000 &
kubectl -n mlops port-forward svc/minio-console 9001:9001 &
kubectl -n mlops port-forward svc/minio 9000:9000 &
kubectl -n kubeflow port-forward svc/ml-pipeline-ui 3000:80 &
kubectl -n kubeflow port-forward svc/streamlit-news-app 8900:8501 &
kubectl -n kubeflow port-forward svc/bentoml-news 10000:3000 &

echo "All port-forwarding started. Press Ctrl+C to stop."
wait
