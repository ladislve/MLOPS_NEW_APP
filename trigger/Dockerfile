FROM python:3.11-slim

WORKDIR /app

COPY trigger_pipeline.py /app/trigger_pipeline.py
COPY news_pipeline.yaml /app/news_pipeline.yaml

RUN pip install kfp

CMD ["python", "trigger_pipeline.py"]