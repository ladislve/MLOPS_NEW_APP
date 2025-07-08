import os
import mlflow
import bentoml

# Configure MLflow (remote or local)
# mlflow.set_tracking_uri("http://localhost:5000")  # or your MLflow URI

# # S3-based artifact storage (MinIO)
# os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

mlflow.set_tracking_uri("http://mlflow.mlops.svc.cluster.local:5000")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio.mlops.svc.cluster.local:9000"

os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin123"

# Find the latest run from your experiment
experiment_name = "news-classification"
client = mlflow.MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"])
run_id = runs[0].info.run_id

# Load MLflow PyFunc model
model_uri = f"runs:/{run_id}/model"
print(f"Loading model from: {model_uri}")
# pyfunc_model = mlflow.pytorch.load_model(model_uri)

# Save into BentoML
bento_model = bentoml.mlflow.import_model(
    name="news_classifier",
    model_uri=model_uri,
    signatures={"predict": {"batchable": True}},  # required for serving
    labels={"mlflow_uri": model_uri},
)
print(f"Model saved to BentoML: {bento_model}")
