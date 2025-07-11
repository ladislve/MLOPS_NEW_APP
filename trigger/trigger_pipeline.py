import kfp
import os

required_env_vars = [
    'NEWS_API_KEY',
    'MINIO_ENDPOINT_KUBEFLOW',
    'GNEWS_API_KEY',
    'MEDIASTACK_ACCESS_KEY',
    'NEWSAPI_API_KEY',
    'GEMINI_API_KEY',
    'KFP_HOST'
]

for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"Environment variable {var} is not set!")

client = kfp.Client(host=os.getenv("KFP_HOST"))

run = client.create_run_from_pipeline_package(
    pipeline_file="news_pipeline.yaml",
    enable_caching=True,
    arguments={
        'news_api_key': os.getenv('NEWS_API_KEY'),
        'endpoint': os.getenv('MINIO_ENDPOINT_KUBEFLOW'),
        'gnews_api_key': os.getenv('GNEWS_API_KEY'),
        'mediastack_access_key': os.getenv('MEDIASTACK_ACCESS_KEY'),
        'newsapi_api_key': os.getenv('NEWSAPI_API_KEY'),
        'gemini_api_key': os.getenv('GEMINI_API_KEY')
    },
    run_name="news-classification-run"
)
