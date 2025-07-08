from kfp import dsl
from kfp.dsl import Input, Output, Dataset
from typing import Annotated

@dsl.component(
    base_image='python:3.10',
    packages_to_install=['pandas', 'boto3', 'scikit-learn','kfp']
)
def preprocess_op(
    raw_data: Annotated[Input[Dataset], "raw_data"],
    train_data: Annotated[Output[Dataset], "train_data"],
    val_data: Annotated[Output[Dataset], "val_data"],
    endpoint:str
):
    import pandas as pd
    import boto3
    import os
    from sklearn.model_selection import train_test_split
    
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin123'
    )

    local_raw_file = 'news_raw.csv'
    local_train_file = 'train.csv'
    local_val_file = 'val.csv'

    s3.download_file('mlops', 'data/news_raw.csv', local_raw_file)

    df = pd.read_csv(local_raw_file)
    df = df.dropna(subset=['title', 'description'])
    df['text'] = df['title'] + ". " + df['description']

    x_train, x_val, y_train, y_val = train_test_split(
        df['text'], df['category'], test_size=0.2, random_state=42, stratify=df['category']
    )

    train_df = pd.DataFrame({'text': x_train, 'category': y_train})
    val_df = pd.DataFrame({'text': x_val, 'category': y_val})

    train_df.to_csv(train_data.path, index=False)  # KFP-managed path
    val_df.to_csv(val_data.path, index=False)

    # Upload for downstream persistence
    s3.upload_file(train_data.path, 'mlops', 'data/train.csv')
    s3.upload_file(val_data.path, 'mlops', 'data/val.csv')

    for file in [local_raw_file]:
        try:
            os.remove(file)
        except OSError as e:
            print(f"Error removing file {file}: {e}")
