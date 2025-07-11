import kfp
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

client = kfp.Client(host="http://localhost:3000")

run = client.create_run_from_pipeline_package(
    pipeline_file="news_pipeline.yaml",
    enable_caching=True,
    arguments={
        'news_api_key': os.getenv('NEWS_API_KEY'),
        'endpoint': os.getenv('MINIO_ENDPOINT_KUBEFLOW'),
        'gnews_api_key': os.getenv('GNEWS_API_KEY'),
        'mediastack_access_key': os.getenv('MEDIASTACK_ACCESS_KEY'),
        'newsapi_api_key': os.getenv('NEWS_API_KEY'),
        'gemini_api_key': os.getenv('GEMINI_API_KEY')
    }
)

# kubectl -n kubeflow create secret generic pipeline-trigger-secrets \
#   --from-literal=NEWS_API_KEY=a602b247767b40b79a579298b21c2fe9 \
#   --from-literal=MINIO_ENDPOINT_KUBEFLOW=http://minio.mlops.svc.cluster.local:9000 \
#   --from-literal=GEMINI_API_KEY=AIzaSyDdoIR2Tnflbe8F33ZwW-gBz95wqr3tPKY \
#   --from-literal=MEDIASTACK_ACCESS_KEY=b6a6d8cbba9c5c17bac94b4a4058a97d \
#   --from-literal=NEWSAPI_API_KEY=a602b247767b40b79a579298b21c2fe9 \
#   --from-literal=GNEWS_API_KEY=f42ffa7ab425651b1448413afecf998c \
#   --from-literal=KFP_HOST=http://ml-pipeline.kubeflow.svc.cluster.local:8888



# import kfp
# from dotenv import load_dotenv
# load_dotenv()
# import os

# client = kfp.Client(host="http://localhost:3000")

# experiment = client.create_experiment(name="News Classification")
# experiment_id = experiment.experiment_id

# cron_expression = "*/8 * * * *"

# recurring_run = client.create_recurring_run(
#     experiment_id=experiment_id,
#     job_name="news-classification-scheduled",
#     description="Run news classification pipeline every hour",
#     pipeline_package_path="news_pipeline.yaml",
#     params={
#         'news_api_key': os.getenv('NEWS_API_KEY'),
#         'endpoint': os.getenv('MINIO_ENDPOINT_KUBEFLOW'),
#     },
#     max_concurrency=1,
#     no_catchup=True,
#     cron_expression=cron_expression,
#     enable_caching=False
# )
