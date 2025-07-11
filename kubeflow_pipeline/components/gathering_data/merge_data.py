from kfp import dsl
from kfp.dsl import Input, Output, Dataset
from typing import Annotated

@dsl.component(
    base_image='python:3.10-slim',
    packages_to_install=['pandas', 'boto3']
)
def merge_data_op(
    rss_data: Annotated[Input[Dataset], "rss_data"],
    scrape_data: Annotated[Input[Dataset], "scrape_data"],
    api_data: Annotated[Input[Dataset], "api_data"],
    merged_data: Annotated[Output[Dataset], "merged_data"],
    endpoint: str,
    aws_access_key_id: str = 'minioadmin',
    aws_secret_access_key: str = 'minioadmin123'
):
    import pandas as pd
    import boto3

    df_rss    = pd.read_csv(rss_data.path)
    df_scrape = pd.read_csv(scrape_data.path)
    df_api    = pd.read_csv(api_data.path)

    df_all  = pd.concat([df_rss, df_scrape, df_api], ignore_index=True)
    df_uniq = df_all.drop_duplicates(subset=['link'])

    df_uniq.to_csv(merged_data.path, index=False)

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    with open(merged_data.path, 'rb') as f:
        resp = s3.put_object(Bucket='mlops', Key='data/merged.csv', Body=f)
    version_id = resp.get('VersionId')
    print(f"Uploaded 'data/merged.csv' with version ID: {version_id}")
