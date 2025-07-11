from kfp import dsl
from kfp.dsl import Input, Output, Dataset
from typing import Annotated

@dsl.component(
    base_image='python:3.10-slim',
    packages_to_install=[
        'pandas',
        'google-generativeai',
        'boto3',
        'scikit-learn'
    ]
)
def preprocess_op(
    merged_data: Annotated[Input[Dataset], "merged_data"],
    endpoint: str,
    api_key: str,
    train_data: Annotated[Output[Dataset], "train_data"],
    val_data: Annotated[Output[Dataset], "val_data"],
    aws_access_key_id: str = 'minioadmin',
    aws_secret_access_key: str = 'minioadmin123',
):
    import os
    import pandas as pd
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    import google.generativeai as genai
    from sklearn.model_selection import train_test_split
    import boto3

    os.environ['GOOGLE_API_KEY'] = api_key
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-lite')

    df = pd.read_csv(merged_data.path)

    def build_prompt(text: str) -> str:
        return (
            "You are an expert at analyzing news articles. Read the following article and assign it to a general category. "
            "Don't use very specific or niche labels. The category should reflect the broad subject matter, such as "
            "'Politics', 'Technology', 'Gaming', 'Health', 'Sports', 'Finance', 'Crime', etc.\n\n"
            "Output ONLY the category name. Do NOT include explanations or anything else.\n\n"
            f"Article:\n{text[:3000]}\n\nCategory:"
        )

    def categorize_sync(prompt: str) -> str:
        try:
            response = model.generate_content(prompt)
            return response.text.strip().split('\n')[0]
        except Exception as e:
            print(f"Error during categorize_sync: {e}")
            return 'Unknown'

    async def categorize_text(text: str, executor: ThreadPoolExecutor):
        loop = asyncio.get_event_loop()
        prompt = build_prompt(text)
        return await loop.run_in_executor(executor, categorize_sync, prompt)

    async def categorize_all(texts):
        MAX_CONCURRENT_TASKS = 5
        executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS)
        sem = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

        async def sem_task(txt):
            async with sem:
                return await categorize_text(txt, executor)

        tasks = [sem_task(txt) for txt in texts]
        return await asyncio.gather(*tasks)

    texts = df['text'].fillna('').tolist()
    categories = asyncio.run(categorize_all(texts))
    df['category'] = categories

    try:
        counts = df['category'].value_counts()
        stratify = df['category'] if counts.min() >= 2 else None
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=stratify
        )
    except Exception as e:
        print(f"Stratified split failed: {e}, falling back to random split.")
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

    train_df.to_csv(train_data.path, index=False)
    val_df.to_csv(val_data.path, index=False)

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    s3.upload_file(train_data.path, 'mlops', 'data/train.csv')
    s3.upload_file(val_data.path, 'mlops', 'data/val.csv')
