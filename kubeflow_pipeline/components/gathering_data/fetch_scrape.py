from kfp import dsl
from kfp.dsl import Output, Dataset
from typing import Annotated

@dsl.component(
    base_image='python:3.10-slim',
    packages_to_install=[
        'pandas',
        'requests',
        'beautifulsoup4',
        'boto3'
    ]
)
def fetch_scrape_op(
    raw_data: Annotated[Output[Dataset], "raw_data"],
    endpoint: str,
    aws_access_key_id: str = 'minioadmin',
    aws_secret_access_key: str = 'minioadmin123',
    max_workers: int = 5
):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import time
    from urllib.parse import urljoin
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import boto3

    class FastNewsScraper:
        def __init__(self, max_workers=5):
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            self.max_workers = max_workers

        def get_page(self, url, timeout=5):
            try:
                resp = self.session.get(url, timeout=timeout)
                return BeautifulSoup(resp.content, 'html.parser')
            except:
                return None

        def extract_text_fast(self, soup):
            if not soup:
                return ""

            selectors = [
                'article p', '.article-body p', '.entry-content p',
                '.post-content p', '.content p', 'main p'
            ]
            for sel in selectors:
                elems = soup.select(sel)
                if elems:
                    txt = ' '.join(p.get_text().strip() for p in elems[:3])
                    if len(txt) > 50:
                        return txt[:200]
            ps = soup.find_all('p')
            if ps:
                txt = ' '.join(p.get_text().strip() for p in ps[:3])
                return txt[:200]
            return soup.get_text()[:200]

        def extract_article(self, url):
            soup = self.get_page(url)
            if not soup:
                return None, None
            title = None
            for sel in ['h1', 'title', '.headline', '.entry-title']:
                el = soup.select_one(sel)
                if el:
                    title = el.get_text().strip()
                    if len(title) > 10:
                        break
            text = self.extract_text_fast(soup)
            return title, text

        def scrape_site(self, cfg):
            name, base, patterns = cfg
            soup = self.get_page(base)
            if not soup:
                return []
            links = set()
            for a in soup.find_all('a', href=True):
                href = a['href']
                for pat in patterns:
                    if pat in href:
                        full = urljoin(base, href) if href.startswith('/') else href
                        if full.startswith('http'):
                            links.add(full)
                if len(links) >= 7:
                    break
            arts = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as exec:
                futs = {exec.submit(self.extract_article, u): u for u in list(links)[:7]}
                for fut in as_completed(futs):
                    u = futs[fut]
                    try:
                        t, txt = fut.result()
                        if t and txt:
                            arts.append({'site': name, 'title': t, 'text': txt, 'link': u})
                    except:
                        pass
            return arts

        def scrape_hackernews(self):
            soup = self.get_page('https://news.ycombinator.com/')
            if not soup:
                return []
            hn = []
            for a in soup.select('a.storylink')[:7]:
                title = a.get_text().strip()
                href = a.get('href')
                url = urljoin('https://news.ycombinator.com/', href) if href.startswith('item?') else href
                hn.append({'site': 'Hacker News', 'title': title, 'text': f'Hacker News story: {title}'[:500], 'link': url})
            return hn

        def scrape_all(self):
            cfgs = [
                ('BBC News', 'https://www.bbc.com/news', ['/news/', '/sport/']),
                ('NPR News','https://www.npr.org/sections/news/',['/2024/','/2025/']),
                ('Reuters','https://www.reuters.com/news/archive/worldNews',['/world/','/business/','/technology/']),
                ('TechCrunch','https://techcrunch.com/',['/2024/','/2025/']),
                ('The Verge','https://www.theverge.com/tech',['/2024/','/2025/']),
                ('Wired','https://www.wired.com/most-recent/',['/story/','/article/']),
                ('Nature','https://www.nature.com/news',['/articles/','/news/']),
                ('Al Jazeera','https://www.aljazeera.com/news/',['/news/','/2024/','/2025/']),
                ('Speedhunters','https://www.speedhunters.com/',['/2024/','/2025/'])
            ]
            all_art = []
            with ThreadPoolExecutor(max_workers=3) as exec:
                futs = [exec.submit(self.scrape_site, c) for c in cfgs]
                futs.append(exec.submit(self.scrape_hackernews))
                for fut in as_completed(futs):
                    try:
                        all_art.extend(fut.result())
                    except:
                        pass
            return all_art

    scraper = FastNewsScraper(max_workers)
    articles = scraper.scrape_all()
    df = pd.DataFrame(articles)
    df.to_csv(raw_data.path, index=False)

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    s3.upload_file(raw_data.path, 'mlops', 'data/news_fast.csv')
