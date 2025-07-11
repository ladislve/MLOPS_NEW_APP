import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import boto3
from sklearn.metrics import accuracy_score, f1_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--endpoint', type=str, required=True)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=8)
args = parser.parse_args()

os.environ["AWS_ACCESS_KEY_ID"] = 'minioadmin'
os.environ["AWS_SECRET_ACCESS_KEY"] = 'minioadmin123'

s3 = boto3.client('s3', endpoint_url=args.endpoint,
                  aws_access_key_id='minioadmin',
                  aws_secret_access_key='minioadmin123')

s3.download_file('mlops', 'data/train.csv', 'train.csv')
s3.download_file('mlops', 'data/val.csv', 'val.csv')

train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')

all_labels = pd.concat([train_df['category'], val_df['category']], ignore_index=True).unique()
label_list = sorted(all_labels)
label2id = {lbl: i for i, lbl in enumerate(label_list)}
num_labels = len(label_list)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

def df_to_loader(df, batch_size, shuffle):
    enc = tokenizer(df['text'].tolist(), padding=True, truncation=True, return_tensors="pt", max_length=512)
    ids = df['category'].map(lambda c: label2id.get(c, -1)).astype(int).tolist()
    labels = torch.tensor(ids, dtype=torch.long)
    ds = TensorDataset(enc['input_ids'], enc['attention_mask'], labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = df_to_loader(train_df, batch_size=args.batch_size, shuffle=True)
val_loader = df_to_loader(val_df, batch_size=args.batch_size, shuffle=False)

optimizer = AdamW(model.parameters(), lr=args.learning_rate)

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

for epoch in range(args.epochs):
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

    tr_acc, tr_f1 = evaluate(train_loader)
    vl_acc, vl_f1 = evaluate(val_loader)

    print(f"epoch={epoch}; train_loss={avg_loss}; train_accuracy={tr_acc}; train_f1={tr_f1}; val_accuracy={vl_acc}; val_f1={vl_f1}")

final_val_acc, final_val_f1 = evaluate(val_loader)
print(f"final_val_accuracy={final_val_acc}")
print(f"final_val_f1={final_val_f1}")
import sys
sys.stdout.flush()
sys.exit(0)