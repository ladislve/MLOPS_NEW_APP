from kubernetes import client, config

config.load_kube_config()

api = client.CustomObjectsApi()

experiment = api.get_namespaced_custom_object(
    group="kubeflow.org",
    version="v1beta1",
    namespace="kubeflow",
    plural="experiments",
    name="news-classifier-tuning"
)

best_params = experiment['status']['currentOptimalTrial']['parameterAssignments']

print(best_params)