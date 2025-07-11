from kfp import dsl

@dsl.component(
    base_image="python:3.10",
    packages_to_install=["requests"]
)
def reload_bentoml_model_op(bentoml_url: str):
    import requests
    resp = requests.post(f"{bentoml_url}/reload_model")
    print("Status:", resp.status_code)
    print("Body:", resp.text)
