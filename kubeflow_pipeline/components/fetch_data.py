from kfp import dsl
from kfp.dsl import Output, Dataset
from typing import Annotated

@dsl.component(
    base_image='python:3.10',
    packages_to_install=['pandas', 'requests', 'boto3','kfp']
)
def fetch_op(raw_data: Annotated[Output[Dataset], "raw_data"],
             news_api_key: str,
             endpoint: str):
    import os
    import requests
    import pandas as pd
    import boto3


    categories = ['business', 'sports', 'technology']
    articles = []

    for category in categories:
        url = f'https://newsapi.org/v2/top-headlines?category={category}&apiKey={news_api_key}&pageSize=50'
        resp = requests.get(url)
        print(resp)
        data = resp.json()
        for article in data['articles']:
            articles.append({
                'title': article['title'],
                'description': article['description'],
                'category': category
            })

    df = pd.DataFrame(articles)
    df.to_csv(raw_data.path, index=False)

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin123'
    )
    s3.upload_file(raw_data.path, 'mlops', 'data/news_raw.csv')
