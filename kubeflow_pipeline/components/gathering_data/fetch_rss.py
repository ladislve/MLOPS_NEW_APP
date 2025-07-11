from kfp import dsl
from kfp.dsl import Output, Dataset
from typing import Annotated

@dsl.component(
    base_image='python:3.10-slim',
    packages_to_install=[
        'pandas',
        'feedparser',
        'newspaper3k',
        'lxml_html_clean',
        'boto3',
    ]
)
def fetch_rss_op(
    raw_data: Annotated[Output[Dataset], "raw_data"],
    endpoint: str,
    aws_access_key_id: str = 'minioadmin',
    aws_secret_access_key: str = 'minioadmin123'
):
    import feedparser
    from newspaper import Article
    import pandas as pd
    import time
    import boto3

    rss_list = [
        'https://feeds.bbci.co.uk/news/world/rss.xml',
        'https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en',
        'https://www.yahoo.com/news/rss',
        'https://www.theguardian.com/world/rss',
        'https://www.reddit.com/r/science/.rss',
        'https://www.theguardian.com/science/space/rss',
        'https://feeds.bbci.co.uk/sport/rss.xml',
        'https://hnrss.org/frontpage',
        'https://feeds.megaphone.fm/vergecast',
        'https://feeds.feedburner.com/speedhunters',
        'https://medium.com/feed/hackernoon',
        'https://www.reddit.com/r/programming/.rss'
    ]

    records = []

    for url in rss_list:
        feed = feedparser.parse(url)
        if feed.bozo:
            print(f"Failed to parse: {url}")
            continue

        for entry in feed.entries[:7]:
            link  = entry.get('link', '').strip()
            title = entry.get('title', '').strip()
            try:
                article = Article(link)
                article.download()
                article.parse()
                text_snippet = article.text[:200]
                if title and text_snippet:
                    records.append({
                        'title': title,
                        'text': text_snippet,
                        'link': link
                    })
                time.sleep(0.05)
            except Exception as e:
                print(f"Failed to scrape {link}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(raw_data.path, index=False)

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    s3.upload_file(raw_data.path, 'mlops', 'data/rss_raw.csv')
