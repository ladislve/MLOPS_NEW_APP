from kfp import dsl

@dsl.component(
    base_image="python:3.10",
    packages_to_install=["mlflow==2.13.0", "bentoml", "boto3", "kubernetes"]
)
def download_model_op(endpoint: str):
    import os
    import mlflow
    import bentoml
    import requests
    from kubernetes import client, config

    # Load Kubernetes config
    config.load_incluster_config()
    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()

    # --- MLflow model export ---
    mlflow.set_tracking_uri("http://mlflow.mlops.svc.cluster.local:5000")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = endpoint
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin123"

    client_mlflow = mlflow.MlflowClient()
    experiment = client_mlflow.get_experiment_by_name("news-classification")
    runs = client_mlflow.search_runs(experiment.experiment_id, order_by=["start_time DESC"])
    new_run = runs[0]
    run_id = new_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    new_loss = new_run.data.metrics.get("train_loss", float("inf"))

    # --- Get current model's loss ---
    all_models = bentoml.models.list()
    current_models = [m for m in all_models if m.tag.name == "news_classifier" and "mlflow_uri" in m.info.labels]
    current_loss = float("inf")
    for model in current_models:
        try:
            uri = model.info.labels["mlflow_uri"]
            parts = uri.split("/")
            run_id_candidate = parts[1] if len(parts) > 1 else ""
            if run_id_candidate:
                old_run = client_mlflow.get_run(run_id_candidate)
                loss = old_run.data.metrics.get("train_loss", float("inf"))
                if loss < current_loss:
                    current_loss = loss
        except Exception as e:
            print(f"Warning: couldn't retrieve loss from label or run ID: {e}")

    print(f"New model loss: {new_loss}, Current model loss: {current_loss}")
    print("Found models:")
    for model in current_models:
        print(f"- Tag: {model.tag}, Label: {model.info.labels.get('mlflow_uri')}")


    if new_loss >= current_loss:
        print("New model did not improve. Skipping deployment.")
        return

    # --- Import model into BentoML ---
    print(f"Importing model from {model_uri} into BentoML...")
    bentoml.mlflow.import_model(
        name="news_classifier",
        model_uri=model_uri,
        signatures={"predict": {"batchable": False}},
        labels={"mlflow_uri": model_uri},
    )

    # --- Determine current and next versions ---
    try:
        svc = core_v1.read_namespaced_service(name="news-classifier", namespace="mlops")
        current_version = svc.spec.selector.get("version", "v1")
    except:
        current_version = "v1"

    next_version = "v2" if current_version == "v1" else "v1"

    # --- Create new deployment ---
    new_deployment = client.V1Deployment(
        metadata=client.V1ObjectMeta(name=f"news-classifier-{next_version}", namespace="mlops"),
        spec=client.V1DeploymentSpec(
            replicas=1,
            selector=client.V1LabelSelector(
                match_labels={"app": "news-classifier", "version": next_version}
            ),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={"app": "news-classifier", "version": next_version}),
                spec=client.V1PodSpec(containers=[
                    client.V1Container(
                        name="news-classifier",
                        image="news_classification:latest",
                        image_pull_policy="Never",
                        ports=[client.V1ContainerPort(container_port=3000)]
                    )
                ])
            )
        )
    )

    try:
        apps_v1.create_namespaced_deployment(namespace="mlops", body=new_deployment)
    except client.exceptions.ApiException as e:
        if e.status == 409:
            apps_v1.replace_namespaced_deployment(name=f"news-classifier-{next_version}", namespace="mlops", body=new_deployment)
        else:
            raise

    # --- Wait for new pod to be ready ---
    import time
    while True:
        pods = core_v1.list_namespaced_pod(namespace="mlops", label_selector=f"version={next_version},app=news-classifier")
        if all(p.status.phase == "Running" and all(c.ready for c in p.status.container_statuses) for p in pods.items):
            break
        time.sleep(2)

    # --- Patch service to point to new version ---
    svc.spec.selector["version"] = next_version
    core_v1.patch_namespaced_service(name="news-classifier", namespace="mlops", body=svc)

    print(f"Deployment switched to version {next_version}.")

    # --- Delete old deployment ---
    old_deployment_name = f"news-classifier-{current_version}"
    try:
        apps_v1.delete_namespaced_deployment(name=old_deployment_name, namespace="mlops")
        print(f"Old deployment {old_deployment_name} deleted.")
    except client.exceptions.ApiException as e:
        print(f"Warning: could not delete old deployment {old_deployment_name}: {e}")