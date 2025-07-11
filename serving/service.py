# import bentoml
# from bentoml.models import BentoModel
# from pydantic import Field
# from typing import Annotated
# import pandas as pd
# import threading
# # Match your label order from training
# TARGET_CATEGORIES = ["world", "sports", "business"]

# demo_image = bentoml.images.PythonImage(python_version="3.10") \
#     .python_packages("mlflow==2.13.0", "torch", "transformers", "pandas")

# @bentoml.service(
#     image=demo_image,
#     traffic={"timeout": 10},
#     resources={"cpu": "2"},
#      http={
#         "cors": {
#             "enabled": True,
#             "access_control_allow_origins": ["*"],  # Allow from all (or replace with specific frontend domain)
#             "access_control_allow_methods": ["GET", "POST", "OPTIONS"],
#             "access_control_allow_credentials": True,
#             "access_control_allow_headers": ["*"],
#         }
#     }
# )




# class NewsClassifier:
#     # bento_model = BentoModel("news_classifier:latest")

#     def __init__(self):
#         # self.model = bentoml.mlflow.load_model(self.bento_model)  # This is a PyFuncModel
#         self.model = self.load_latest_model()
#         self.model_lock = threading.Lock()
#         import os
#         print("BENTOML_HOME:", os.environ.get("BENTOML_HOME"))

#     def load_latest_model(self):
#         filtered = [m for m in bentoml.models.list() if m.tag.name == "news_classifier"]
#         if not filtered:
#             raise RuntimeError("No models found with tag 'news_classifier'")
#         # Sort by creation_time (most recent first)
#         filtered.sort(key=lambda m: m.creation_time, reverse=True)
#         print(filtered[0])
#         print(filtered[1])
#         latest_model = filtered[0]
#         print("Model loaded:", latest_model.tag, "created at:", latest_model.creation_time)
#         return bentoml.mlflow.load_model(latest_model.tag)
    
#     @bentoml.api
#     def predict(
#         self,
#         text: Annotated[str, Field(..., description="News article text")]
#     ) -> str:
#         input_df = pd.DataFrame({"text": [text]})  # Pass raw text
#         with self.model_lock:
#             prediction = self.model.predict(input_df)
#         return TARGET_CATEGORIES[int(prediction[0])]
#         # return 'text'
    
#     @bentoml.api
#     def reload_model(self) -> str:
#         with self.model_lock:
#             self.model = self.load_latest_model()
#         return "Model reloaded"
    
#     @bentoml.api
#     def current_model_info(self) -> dict:
#         from bentoml.models import get
#         model = get("news_classifier:latest")
#         return {
#             "tag": str(model.tag),
#             "labels": model.info.labels,
#             "created_at": str(model.info.creation_time),
#         }


import bentoml
from bentoml.models import BentoModel
from pydantic import Field
from typing import Annotated
import pandas as pd
import threading
import json
import os
import boto3
from botocore.client import Config

demo_image = bentoml.images.PythonImage(python_version="3.10") \
    .python_packages("mlflow==2.13.0", "torch", "transformers", "pandas", "boto3")

@bentoml.service(
    image=demo_image,
    traffic={"timeout": 10},
    resources={"cpu": "2"},
    http={
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["*"],
            "access_control_allow_methods": ["GET", "POST", "OPTIONS"],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
        }
    }
)
class NewsClassifier:
    def __init__(self):
        self.categories = self.load_labels_from_minio()
        self.model = self.load_latest_model()
        self.model_lock = threading.Lock()

    def load_labels_from_minio(self):
        endpoint   = os.environ.get("MINIO_ENDPOINT")
        access_key = os.environ.get("MINIO_ACCESS_KEY")
        secret_key = os.environ.get("MINIO_SECRET_KEY")
        bucket     = os.environ.get("MINIO_BUCKET")
        object_key = os.environ.get("LABELS_OBJECT", "data/labels.json")

        if not all([endpoint, access_key, secret_key, bucket, object_key]):
            raise RuntimeError("MinIO config missing from env variables")

        endpoint_url = endpoint
        if endpoint_url.startswith("http://"):
            use_ssl = False
        elif endpoint_url.startswith("https://"):
            use_ssl = True
        else:
            raise RuntimeError("MINIO_ENDPOINT must start with http:// or https://")

        s3 = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4'),
            region_name="us-east-1",
            use_ssl=use_ssl
        )

        obj = s3.get_object(Bucket=bucket, Key=object_key)
        labels_data = obj["Body"].read().decode("utf-8")
        categories = json.loads(labels_data)
        print("Loaded categories from MinIO:", categories)
        return categories

    def load_latest_model(self):
        filtered = [m for m in bentoml.models.list() if m.tag.name == "news_classifier"]
        if not filtered:
            raise RuntimeError("No models found with tag 'news_classifier'")
        filtered.sort(key=lambda m: m.creation_time, reverse=True)
        latest_model = filtered[0]
        print("Model loaded:", latest_model.tag, "created at:", latest_model.creation_time)
        return bentoml.mlflow.load_model(latest_model.tag)

    @bentoml.api
    def predict(
        self,
        text: Annotated[str, Field(..., description="News article text")]
    ) -> str:
        input_df = pd.DataFrame({"text": [text]})
        with self.model_lock:
            prediction = self.model.predict(input_df)
        idx = int(prediction[0])
        if idx < 0 or idx >= len(self.categories):
            return "UNKNOWN"
        return self.categories[idx]

    @bentoml.api
    def reload_model(self) -> str:
        with self.model_lock:
            self.model = self.load_latest_model()
            self.categories = self.load_labels_from_minio()
        return "Model and labels reloaded"

    @bentoml.api
    def get_categories(self) -> list:
        return self.categories

    @bentoml.api
    def current_model_info(self) -> dict:
        from bentoml.models import get
        model = get("news_classifier:latest")
        return {
            "tag": str(model.tag),
            "labels": model.info.labels,
            "created_at": str(model.info.creation_time),
            "categories": self.categories
        }
