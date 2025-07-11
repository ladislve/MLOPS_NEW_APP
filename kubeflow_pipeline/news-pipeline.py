from kfp import dsl
from kfp import kubernetes
from kubeflow_pipeline.components.preprocess import preprocess_op
from kubeflow_pipeline.components.train import train_op
from kubeflow_pipeline.components.download_model import download_model_op
from kubeflow_pipeline.components.reload_model import reload_bentoml_model_op

# New components
from kubeflow_pipeline.components.gathering_data.fetch_rss import fetch_rss_op
from kubeflow_pipeline.components.gathering_data.fetch_scrape import fetch_scrape_op
from kubeflow_pipeline.components.gathering_data.fetch_api import fetch_api_op
from kubeflow_pipeline.components.gathering_data.merge_data import merge_data_op

@dsl.pipeline(
    name='news-classification-full'
)
def news_pipeline(
    news_api_key: str,
    endpoint: str,
    gnews_api_key: str,
    mediastack_access_key: str,
    newsapi_api_key: str,
    gemini_api_key: str
):
    # 1) Fetch from RSS
    rss_task = fetch_rss_op(
        endpoint=endpoint
    )

    # 2) Fetch by scraping
    scrape_task = fetch_scrape_op(
        endpoint=endpoint,
        max_workers=8
    )

    # 3) Fetch from open-news APIs
    api_task = fetch_api_op(
        endpoint=endpoint,
        gnews_api_key=gnews_api_key,
        mediastack_access_key=mediastack_access_key,
        newsapi_api_key=newsapi_api_key
    )

    # 4) Merge all sources on unique 'link'
    merge_task = merge_data_op(
        rss_data=rss_task.outputs['raw_data'],
        scrape_data=scrape_task.outputs['raw_data'],
        api_data=api_task.outputs['raw_data'],
        endpoint=endpoint
    )

    preprocess_task = preprocess_op(
        merged_data=merge_task.outputs["merged_data"],
        endpoint=endpoint,
        api_key=gemini_api_key
    )

    # # 5) Preprocess merged data
    # preprocess_task = preprocess_op(
    #     raw_data=merge_task.outputs['merged_data'],
    #     endpoint=endpoint
    # )

    # 6) Train model
    train_task = train_op(
        train_data=preprocess_task.outputs['train_data'],
        val_data=preprocess_task.outputs['val_data'],
        endpoint="http://minio.mlops.svc.cluster.local:9000",
        epochs=11
    )

    # 7) Download the trained Bento model to PVC
    download_task = download_model_op(
        endpoint="http://minio.mlops.svc.cluster.local:9000"
    ).after(train_task)

    kubernetes.mount_pvc(
        download_task,
        pvc_name='bentoml-pvc',
        mount_path='/bentoml_storage'
    )

    # 8) Reload in Bento if download succeeded
    with dsl.If(download_task.output == True):
        reload_bentoml_model_op(
            bentoml_url="http://bentoml-news.kubeflow.svc.cluster.local:3000"
        )


if __name__ == '__main__':
    from kfp.compiler import Compiler
    Compiler().compile(news_pipeline, 'news_pipeline.yaml')
