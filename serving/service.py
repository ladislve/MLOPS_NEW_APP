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
     http={
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["*"],  # Allow from all (or replace with specific frontend domain)
            "access_control_allow_methods": ["GET", "POST", "OPTIONS"],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
        }
    }
)




class NewsClassifier:
    bento_model = BentoModel("news_classifier:latest")

    def __init__(self):
        # self.model = bentoml.mlflow.load_model(self.bento_model)  # This is a PyFuncModel
        self.model = self.load_latest_model()

    def load_latest_model(self):
        models = bentoml.models.list()
        print('the models:\n\n\n')
        
        for m in models:
            print(m.tag.name)
        filtered = [m for m in models if m.tag.name == "news_classifier"]
        if not filtered:
            raise RuntimeError("No models found with tag 'news_classifier'")
        latest_model = filtered[-1]  # or sorted if needed
        print(" Model loaded:", latest_model.tag)
        return bentoml.mlflow.load_model(latest_model.tag)
    
    @bentoml.api
    def predict(
        self,
        text: Annotated[str, Field(..., description="News article text")]
    ) -> str:
        input_df = pd.DataFrame({"text": [text]})  # Pass raw text
        prediction = self.model.predict(input_df)
        return TARGET_CATEGORIES[int(prediction[0])]
        # return 'text'
    
    @bentoml.api
    def reload_model(self) -> str:
        self.model = self.load_latest_model()
        return "Model reloaded"

