from kfp import dsl
from kubeflow_pipeline.components.fetch_data import fetch_op
from kubeflow_pipeline.components.preprocess import preprocess_op
from kubeflow_pipeline.components.train import train_op
from kubeflow_pipeline.components.download_model import download_model_op

@dsl.pipeline(
    name='news-classification-full'
)
def news_pipeline(news_api_key:str, endpoint:str):
    # Fetch raw data and output as artifact
    fetch_task = fetch_op(news_api_key=news_api_key, endpoint=endpoint)

    # Preprocess using fetched raw data
    preprocess_task = preprocess_op(
        raw_data=fetch_task.outputs['raw_data'],
        endpoint=endpoint
    )

    # Train using preprocessed train and validation datasets
    train_task = train_op(
    train_data=preprocess_task.outputs['train_data'],
    val_data=preprocess_task.outputs['val_data'],
    endpoint="http://minio.mlops.svc.cluster.local:9000",
    epochs=1
    )   

    download_task = download_model_op(
        endpoint="http://minio.mlops.svc.cluster.local:9000"
    ).after(train_task)

if __name__ == '__main__':
    from kfp.compiler import Compiler
    Compiler().compile(news_pipeline, 'news_pipeline.yaml')
