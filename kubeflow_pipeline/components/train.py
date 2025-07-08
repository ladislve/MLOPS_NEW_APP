from kfp import dsl
from kfp.dsl import Input, Dataset
from typing import Annotated

@dsl.component(
    base_image='python:3.10',
    packages_to_install=[
        'pandas', 'torch', 'transformers', 'mlflow==2.13.0', 'boto3', 'kfp'
    ]
)
def train_op(
    train_data: Annotated[Input[Dataset], "train_data"],
    val_data: Annotated[Input[Dataset], "val_data"],
    endpoint: str,
    epochs: int = 1
):
    import os
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import mlflow
    import mlflow.pyfunc
    import boto3
    from torch.serialization import safe_globals

    # --- Environment Setup ---
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = endpoint
    os.environ["AWS_ACCESS_KEY_ID"] = 'minioadmin'
    os.environ["AWS_SECRET_ACCESS_KEY"] = 'minioadmin123'
    mlflow.set_tracking_uri("http://mlflow.mlops.svc.cluster.local:5000")
    mlflow.set_experiment("news-classification")

    # --- Upload datasets to MinIO (optional) ---
    boto_session = boto3.session.Session()
    s3 = boto_session.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin123'
    )
    s3.upload_file(train_data.path, 'mlops', 'data/train.csv')
    s3.upload_file(val_data.path, 'mlops', 'data/val.csv')

    # --- Load data ---
    train_df = pd.read_csv(train_data.path)
    val_df = pd.read_csv(val_data.path)

    # --- Prepare model and tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    num_categories = train_df['category'].nunique()

    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=num_categories
    )

    train_enc = tokenizer(train_df['text'].tolist(), padding=True, truncation=True, return_tensors="pt")
    train_labels = torch.tensor(pd.Categorical(train_df['category']).codes, dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(train_enc['input_ids'], train_enc['attention_mask'], train_labels)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    # --- Training ---
    mlflow.start_run()
    mlflow.log_param("epochs", epochs)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask, labels in train_loader:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

    # --- Save raw PyTorch model locally ---
    torch.save(model, "model.pt")

    # --- Define custom PythonModel wrapper ---
    class NewsClassifierWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import torch
            from transformers import AutoTokenizer
            self.model = torch.load(context.artifacts["model_path"],
                                    weights_only=False)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        def predict(self, context, model_input):
            import torch
            texts = model_input["text"].tolist()
            tokens = self.tokenizer(
                texts, padding=True, truncation=True,
                return_tensors="pt", max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**tokens)
                preds = torch.argmax(outputs.logits, dim=1).numpy()
            return preds

    # --- Log as PyFunc with preprocessing included ---
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=NewsClassifierWrapper(),
        artifacts={"model_path": "model.pt"}
    )

    mlflow.end_run()
