FROM python:3.11-slim

WORKDIR /app

COPY app.py /app/app.py

RUN pip install streamlit requests readability-lxml feedparser beautifulsoup4

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true", "--server.enableCORS=false"]
