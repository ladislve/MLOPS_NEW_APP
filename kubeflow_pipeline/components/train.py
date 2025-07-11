from kfp import dsl
from kfp.dsl import Input, Dataset
from typing import Annotated

@dsl.component(
    base_image='python:3.10-slim',
    packages_to_install=[
        'pandas', 'torch', 'transformers', 'mlflow==2.13.0', 'boto3', 'kfp', 'scikit-learn'
    ]
)
def train_op(
    train_data: Annotated[Input[Dataset], "train_data"],
    val_data:   Annotated[Input[Dataset], "val_data"],
    endpoint: str,
    epochs:    int = 1
):
    import os
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import mlflow
    import boto3
    import json
    from sklearn.metrics import accuracy_score, f1_score
    from torch.serialization import add_safe_globals
    from transformers.models.distilbert.modeling_distilbert import (
    DistilBertModel,
    DistilBertForSequenceClassification,
    )   

    os.environ["MLFLOW_S3_ENDPOINT_URL"]    = endpoint
    os.environ["AWS_ACCESS_KEY_ID"]         = 'minioadmin'
    os.environ["AWS_SECRET_ACCESS_KEY"]     = 'minioadmin123'
    mlflow.set_tracking_uri("http://mlflow.mlops.svc.cluster.local:5000")
    mlflow.set_experiment("news-classification")

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin123'
    )
    s3.upload_file(train_data.path, 'mlops', 'data/train.csv')
    s3.upload_file(val_data.path,   'mlops', 'data/val.csv')

    train_df = pd.read_csv(train_data.path)
    val_df   = pd.read_csv(val_data.path)

    all_labels = pd.concat([train_df['category'], val_df['category']], ignore_index=True).unique()
    label_list = sorted(all_labels)
    label2id   = {lbl: i for i, lbl in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model     = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_labels
    )

    def df_to_loader(df: pd.DataFrame, batch_size: int, shuffle: bool):
        enc = tokenizer(
            df['text'].tolist(),
            padding=True, truncation=True, return_tensors="pt", max_length=512
        )

        ids = df['category'].map(lambda c: label2id.get(c, -1)).astype(int).tolist()
        labels = torch.tensor(ids, dtype=torch.long)
        ds = TensorDataset(enc['input_ids'], enc['attention_mask'], labels)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = df_to_loader(train_df, batch_size=8, shuffle=True)
    val_loader   = df_to_loader(val_df,   batch_size=16, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    def evaluate(loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for input_ids, attn_mask, labels in loader:
                out = model(input_ids=input_ids, attention_mask=attn_mask)
                batch_preds = torch.argmax(out.logits, dim=1).cpu().numpy()
                preds.extend(batch_preds)
                trues.extend(labels.cpu().numpy())
        return accuracy_score(trues, preds), f1_score(trues, preds, average='weighted')

    mlflow.start_run()
    mlflow.log_param("epochs", epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for input_ids, attn_mask, labels in train_loader:
            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        tr_acc, tr_f1 = evaluate(train_loader)
        vl_acc, vl_f1 = evaluate(val_loader)
        mlflow.log_metric("train_accuracy", tr_acc, step=epoch)
        mlflow.log_metric("train_f1",       tr_f1, step=epoch)
        mlflow.log_metric("val_accuracy",   vl_acc, step=epoch)
        mlflow.log_metric("val_f1",         vl_f1, step=epoch)

    f_tr_acc, f_tr_f1 = evaluate(train_loader)
    f_val_acc, f_val_f1 = evaluate(val_loader)
    mlflow.log_metric("final_train_accuracy", f_tr_acc)
    mlflow.log_metric("final_train_f1",       f_tr_f1)
    mlflow.log_metric("final_val_accuracy",   f_val_acc)
    mlflow.log_metric("final_val_f1",         f_val_f1)

    torch.save(model, "model.pt")
    with open('labels.json', 'w') as f:
        json.dump(label_list, f)
    s3.upload_file('labels.json', 'mlops', 'data/labels.json')

    class NewsClassifierWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import torch
            from transformers import AutoTokenizer
            add_safe_globals([
                DistilBertModel,
                DistilBertForSequenceClassification,
            ])
            
            self.model = torch.load(
                context.artifacts["model_path"],
                weights_only=False
            )
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')


        def predict(self, context, model_input):
            import torch
            texts = model_input["text"].tolist()
            toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad():
                out = self.model(**toks)
                return torch.argmax(out.logits, dim=1).cpu().numpy()

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=NewsClassifierWrapper(),
        artifacts={"model_path": "model.pt"}
    )


    
    mlflow.end_run()