FROM python:3.10-slim
COPY train.py /train.py
RUN pip install pandas torch transformers boto3 scikit-learn
ENTRYPOINT ["python", "/train.py"]
