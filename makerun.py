import kfp
from dotenv import load_dotenv
load_dotenv()
import os
client = kfp.Client(host="http://localhost:3000")

# Create a run directly from the pipeline YAML
run = client.create_run_from_pipeline_package(
    pipeline_file="news_pipeline.yaml",
    arguments={'news_api_key': os.getenv('NEWS_API_KEY'),
               'endpoint': os.getenv('MINIO_ENDPOINT_KUBEFLOW'),},  # If your pipeline accepts parameters, put them here
    run_name="news-classification-run"
)

# import os
# import requests
# import pandas as pd
# import boto3
# from dotenv import load_dotenv

# load_dotenv()
# NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# categories = ['business', 'sports', 'technology']
# articles = []

# for category in categories:
#     url = f'https://newsapi.org/v2/top-headlines?category={category}&apiKey={NEWS_API_KEY}&pageSize=50'
#     resp = requests.get(url)
    
#     data = resp.json()
#     print(data)
#     for article in data['articles']:
#         articles.append({
#             'title': article['title'],
#             'description': article['description'],
#             'category': category
#         })

# df = pd.DataFrame(articles)
# df.to_csv('raw_data.csv', index=False)
# endpoint = os.getenv('MINIO_ENDPOINT_LOCAL')
# if os.getenv('KUBEFLOW_RUN') == 'true':
#     endpoint = os.getenv('MINIO_ENDPOINT_KUBEFLOW')

# s3 = boto3.client(
#     's3',
#     endpoint_url=endpoint,
#     aws_access_key_id='minioadmin',
#     aws_secret_access_key='minioadmin123'
# )
# s3.upload_file('raw_data.csv', 'mlops', 'data/news_raw.csv')