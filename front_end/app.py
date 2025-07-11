import streamlit as st
import feedparser
import random
import pandas as pd
import requests
from readability import Document
from bs4 import BeautifulSoup
import re

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "https://www.yahoo.com/news/rss"
]

BENTOML_URL = "http://bentoml-news.kubeflow.svc.cluster.local:3000/predict"

UNWANTED_PHRASES = [
    "Manage your account",
    "Listen to this article on BBC Sounds",
    "Written by",
    "Reviewed by",
    "Follow us on",
    "Sign up for",
    "Subscribe to",
    "Share this article",
    "Read more",
    "Advertisement",
    "Sponsored content",
]

def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    for phrase in UNWANTED_PHRASES:
        text = text.replace(phrase, '')
    return text.strip()

def extract_main_content(entry):
    try:
        response = requests.get(entry.link, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            doc = Document(response.text)
            content_html = doc.summary()
            soup = BeautifulSoup(content_html, 'html.parser')
            paragraphs = [p.get_text() for p in soup.find_all('p')]
            text = ' '.join(paragraphs)
            return clean_text(text[:3000])
    except Exception as e:
        print(f"Error fetching {entry.link}: {e}")
        return "Failed to fetch content"

def fetch_articles():
    all_entries = []
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        all_entries.extend(feed.entries)
    selected_entries = random.sample(all_entries, min(10, len(all_entries)))

    articles = []
    for entry in selected_entries:
        text = extract_main_content(entry)
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "text": text
        })
    return pd.DataFrame(articles)

def get_category_prediction(text_snippet):
    try:
        response = requests.post(BENTOML_URL, json={"text": text_snippet})
        if response.ok:
            return response.text.strip()
        else:
            return "Prediction failed"
    except Exception as e:
        return f"Error: {e}"

st.set_page_config(layout="centered")
st.title("Random News Articles with Category Prediction")

df = fetch_articles()

predictions = []
for _, row in df.iterrows():
    snippet = row['text'][:300]
    prediction = get_category_prediction(snippet)
    predictions.append(prediction)

df['category'] = predictions

for idx, row in df.iterrows():
    with st.expander(label=f"{row['title']}"):
        st.markdown(f"### {row['title']}")
        st.write(f"**Category:** {row['category']}")
        st.write(f"[Read original article]({row['link']})")
        st.write(row['text'])
