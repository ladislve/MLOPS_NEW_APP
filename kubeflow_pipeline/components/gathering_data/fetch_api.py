from kfp import dsl
from kfp.dsl import Output, Dataset
from typing import Annotated

@dsl.component(
    base_image='python:3.10-slim',
    packages_to_install=[
        'pandas',
        'requests',
        'newspaper3k',
        'lxml_html_clean',
        'boto3',
    ]
)
def fetch_api_op(
    raw_data: Annotated[Output[Dataset], "raw_data"],
    endpoint: str,
    gnews_api_key: str,
    mediastack_access_key: str,
    newsapi_api_key: str,
    aws_access_key_id: str = 'minioadmin',
    aws_secret_access_key: str = 'minioadmin123'
):
    import requests
    import pandas as pd
    import time
    from newspaper import Article
    import boto3

    def scrape_article_text(url):
        try:
            art = Article(url)
            art.download()
            art.parse()
            return art.text[:500]
        except:
            return ""

    def fetch_gnews_articles():
        url = f"https://gnews.io/api/v4/search?q=world&lang=en&max=7&token={gnews_api_key}"
        resp = requests.get(url)
        data = resp.json()
        arts = []
        for item in data.get("articles", []):
            title = item.get("title", "").strip()
            link = item.get("url", "").strip()
            text = scrape_article_text(link)
            if title and text:
                arts.append({
                    "source": "GNews",
                    "title": title,
                    "text": text,
                    "link": link
                })
            time.sleep(0.05)
        return arts

    def fetch_mediastack_articles():
        url = f"http://api.mediastack.com/v1/news?access_key={mediastack_access_key}&languages=en&limit=7"
        resp = requests.get(url)
        data = resp.json()
        arts = []
        for item in data.get("data", []):
            title = item.get("title", "").strip()
            link = item.get("url", "").strip()
            text = scrape_article_text(link)
            if title and text:
                arts.append({
                    "source": "Mediastack",
                    "title": title,
                    "text": text,
                    "link": link
                })
            time.sleep(0.05)
        return arts

    def fetch_hn_rssapi():
        url = "https://hnrss.org/frontpage.jsonfeed"
        resp = requests.get(url)
        data = resp.json()
        arts = []
        for item in data.get("items", [])[:7]:
            title = item.get("title", "").strip()
            link = item.get("url", "").strip()
            text = scrape_article_text(link)
            if title and text:
                arts.append({
                    "source": "HackerNews",
                    "title": title,
                    "text": text,
                    "link": link
                })
            time.sleep(0.05)
        return arts

    def fetch_newsapi_articles():
        cats = ['business', 'technology', 'science', 'health', 'general']
        arts = []
        for cat in cats:
            url = f'https://newsapi.org/v2/top-headlines?category={cat}&language=en&pageSize=7&apiKey={newsapi_api_key}'
            resp = requests.get(url)
            data = resp.json()
            for item in data.get("articles", []):
                title = item.get("title", "").strip()
                link = item.get("url", "").strip()
                text = scrape_article_text(link)
                if title and text:
                    arts.append({
                        "source": f"NewsAPI-{cat}",
                        "title": title,
                        "text": text,
                        "link": link
                    })
                time.sleep(0.05)
        return arts

    all_articles = []
    all_articles.extend(fetch_gnews_articles())
    all_articles.extend(fetch_hn_rssapi())
    all_articles.extend(fetch_mediastack_articles())
    all_articles.extend(fetch_newsapi_articles())

    df = pd.DataFrame(all_articles)
    df.to_csv(raw_data.path, index=False)

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    s3.upload_file(raw_data.path, 'mlops', 'data/news_api_raw.csv')
