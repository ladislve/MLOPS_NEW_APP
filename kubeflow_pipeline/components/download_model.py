from kfp import dsl

@dsl.component(
    base_image="python:3.10",
    packages_to_install=["mlflow==2.13.0", "bentoml", "boto3"]
)
def download_model_op(endpoint: str):
    import os
    import mlflow
    import bentoml

    mlflow.set_tracking_uri("http://mlflow.mlops.svc.cluster.local:5000")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = endpoint
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin123"

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name("news-classification")
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"])
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"

    print(f"Importing model from {model_uri} into BentoML...")
    bentoml.mlflow.import_model(
        name="news_classifier",
        model_uri=model_uri,
        signatures={"predict": {"batchable": False}},
        labels={"mlflow_uri": model_uri},
    )
