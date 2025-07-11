from kfp import dsl

@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        'transformers',
        'torch',
        "mlflow==2.13.0",
        "bentoml",
        "boto3",
        "pandas",
        "scikit-learn"
    ]
)
def download_model_op(endpoint: str) -> bool:
    import os, time
    import pandas as pd
    import mlflow
    from mlflow.exceptions import MlflowException
    from sklearn.metrics import f1_score
    import boto3
    import bentoml


    os.environ["MLFLOW_S3_ENDPOINT_URL"] = endpoint
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin123"
    mlflow.set_tracking_uri("http://mlflow.mlops.svc.cluster.local:5000")
    os.environ["BENTOML_HOME"] = "/bentoml_storage"
    client = mlflow.MlflowClient()

    s3 = boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id="minioadmin", aws_secret_access_key="minioadmin123"
    )
    s3.download_file("mlops", "data/train.csv", "train.csv")
    s3.download_file("mlops", "data/val.csv",   "val.csv")

    train_df = pd.read_csv("train.csv")
    val_df   = pd.read_csv("val.csv")

    s3.download_file('mlops', 'data/labels.json', 'labels.json')
    import json
    with open('labels.json') as f:
        label_list = json.load(f)
    label2id = {lbl: i for i, lbl in enumerate(label_list)}

    true_codes = val_df['category'].map(label2id).astype(int).tolist()

    prod_f1 = -float("inf")
    try:
        prod_pyfunc = mlflow.pyfunc.load_model("models:/news_classifier/Production")
        prod_preds  = prod_pyfunc.predict(val_df)
        prod_codes  = pd.Series(prod_preds).astype(int).tolist()
        prod_f1     = f1_score(true_codes, prod_codes, average="weighted")
    except MlflowException:
        print("No existing Production model; setting prod_f1 = -inf")
    print(f"Production model eval F1 = {prod_f1:.4f}")

    exp  = client.get_experiment_by_name("news-classification")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    if not runs:
        raise RuntimeError("No runs found for 'news-classification'.")
    challenger = runs[0]
    run_id     = challenger.info.run_id
    challenger_f1 = challenger.data.metrics.get("final_val_f1")
    if challenger_f1 is None:
        raise RuntimeError(f"Latest run {run_id} missing final_val_f1 metric.")
    print(f"Challenger run {run_id} logged F1 = {challenger_f1:.4f}")

    should_reload = False
    if challenger_f1 > prod_f1:
        print("Promoting new model to Production.")
        reg = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name="news_classifier")
        version = reg.version
        for _ in range(20):
            mv = client.get_model_version("news_classifier", version)
            if mv.status == "READY":
                break
            time.sleep(1)
        client.transition_model_version_stage(
            name="news_classifier",
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        bentoml.mlflow.import_model(
            name="news_classifier",
            model_uri=f"runs:/{run_id}/model",
            signatures={"predict": {"batchable": False}},
            labels={
                "mlflow_run_id": run_id,
                "final_val_f1": f"{challenger_f1:.4f}",
                "mlflow_version": str(version)
            }
        )
        should_reload = True
    else:
        print("No promotion needed; Production remains best or equal.")




    return should_reload

