import bentoml
from bentoml.models import BentoModel
from pydantic import Field
from typing import Annotated
import pandas as pd

# Match your label order from training
TARGET_CATEGORIES = ["world", "sports", "business"]

demo_image = bentoml.images.PythonImage(python_version="3.10") \
    .python_packages("mlflow==2.13.0", "torch", "transformers", "pandas")

@bentoml.service(
    image=demo_image,
    traffic={"timeout": 10},
    resources={"cpu": "2"},
)
class NewsClassifier:
    bento_model = BentoModel("news_classifier:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)  # This is a PyFuncModel

    @bentoml.api
    def predict(
        self,
        text: Annotated[str, Field(..., description="News article text")]
    ) -> str:
        input_df = pd.DataFrame({"text": [text]})  # Pass raw text
        prediction = self.model.predict(input_df)
        return TARGET_CATEGORIES[int(prediction[0])]
