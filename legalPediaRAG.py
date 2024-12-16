import os
import asyncio
import aiohttp
import weaviate
import logging
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from weaviate.classes.init import AdditionalConfig, Timeout
import json
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ingestion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Weaviate URL
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
LEGALPEDIA_API_KEY=os.getenv("LEGALPEDIA_API_KEY")
LEGALPEDIA_BASE_URL = os.getenv("LEGALPEDIA_BASE_URL", "https://legalpediaresources.com/api/v1")
LEGALPEDIA_BEARER_TOKEN = os.getenv("LEGALPEDIA_BEARER_TOKEN")

# Connect to the Weaviate client
# Connect to the Weaviate client
try:
    weaviate_client = weaviate.connect_to_local( 
    host="157.245.215.92",  # Use a string to specify the host
    port=8080,
     additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=60, insert=120)  # Timeout values in seconds
        ),
        skip_init_checks=True  # Optional: bypass initial connection checks if needed
    )
    logger.info("Connected to Weaviate successfully.")
except weaviate.exceptions.WeaviateConnectionError as e:
    logger.error(f"Failed to connect to Weaviate: {e}")
    raise SystemExit("Exiting script due to connection failure.")


# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)

# Initialize embeddings using OpenAI's model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Track progress
progress_file = "legalpedia_ingestion_progress.json"

# Successful IDs tracker
successful_ids = defaultdict(list)

def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return json.load(f)
    return {}

def save_progress(progress):
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)

async def fetch_single(session, endpoint, doc_id, retries, method="GET", payload=None):
    url = f"{LEGALPEDIA_BASE_URL}/{endpoint}/{doc_id}" if method == "GET" else f"{LEGALPEDIA_BASE_URL}/{endpoint}"
    headers = {
        'Authorization': f'Bearer {LEGALPEDIA_BEARER_TOKEN}',
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'API-Key': LEGALPEDIA_API_KEY
    }

    for attempt in range(retries + 1):
        try:
            if method == "GET":
                payload = {'api_key': LEGALPEDIA_API_KEY}  # Add API-Key for GET if needed
                async with session.get(url, headers=headers, timeout=15, params=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    elif response.status == 404:
                        logger.warning(f"No data for ID {doc_id} in endpoint {endpoint}")
                        return None
                    else:
                        logger.error(f"Failed to fetch ID {doc_id} from {endpoint}: {response.status}")
            elif method == "POST":
                payload['api_key'] = LEGALPEDIA_API_KEY
                async with session.post(url, headers=headers, json=payload, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        logger.error(f"POST request failed for endpoint {endpoint} with status {response.status}")
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching data from {endpoint} (Attempt {attempt + 1}): {e}")
        await asyncio.sleep(2)
    return None

async def fetch_paginated_data(session, endpoint, title, payload, retries=2):
    documents = []
    page = 1
    payload['api_key'] = LEGALPEDIA_API_KEY  # Add API-Key to payload


    while True:
        logger.info(f"Fetching page {page} for {endpoint}")
        payload["page"] = page
        data = await fetch_single(session, endpoint, None, retries, method="POST", payload=payload)

        if data and "results" in data and data["results"]:
            for item in data["results"]:
                content = json.dumps(item)
                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    documents.append({
                        "title": title,
                        "content": chunk
                    })
            page += 1
        else:
            logger.info(f"No more data for {endpoint} after page {page}")
            break

    return documents

async def fetch_data(endpoint, start_id, max_id, concurrency_limit=50, retries=2):
    progress = load_progress()
    last_processed_id = progress.get(endpoint, start_id)
    documents = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for doc_id in range(last_processed_id, max_id + 1):
            tasks.append(fetch_single(session, endpoint, doc_id, retries))

            # Concurrency limit
            if len(tasks) >= concurrency_limit:
                results = await asyncio.gather(*tasks)
                for idx, data in enumerate(results):
                    doc_id = last_processed_id + idx
                    if data:
                        content = json.dumps(data)
                        chunks = text_splitter.split_text(content)

                        for chunk in chunks:
                            documents.append({
                                "title": endpoint,
                                "content": chunk
                            })

                        successful_ids[endpoint].append(doc_id)
                        progress[endpoint] = doc_id
                        save_progress(progress)
                        logger.info(f"Successfully processed document ID {doc_id} from {endpoint}")
                tasks = []

        # Process any remaining tasks
        if tasks:
            results = await asyncio.gather(*tasks)
            for idx, data in enumerate(results):
                doc_id = last_processed_id + idx
                if data:
                    content = json.dumps(data)
                    chunks = text_splitter.split_text(content)

                    for chunk in chunks:
                        documents.append({
                            "title": endpoint,
                            "content": chunk
                        })

                    successful_ids[endpoint].append(doc_id)
                    progress[endpoint] = doc_id
                    save_progress(progress)
                    logger.info(f"Successfully processed document ID {doc_id} from {endpoint}")

    return documents

def ingest_to_weaviate(documents):
    try:
        WeaviateVectorStore.from_documents(documents, embeddings, client=weaviate_client)
        logger.info(f"Successfully ingested {len(documents)} documents into Weaviate.")
    except Exception as e:
        logger.error(f"Error during ingestion to Weaviate: {e}")


def close_weaviate_client():
    """Close the Weaviate client connection."""
    try:
        weaviate_client.close()
        logger.info("Weaviate client closed successfully.")
    except Exception as e:
        logger.error(f"Error closing Weaviate client: {e}")

def ingest_legalpedia_data():
    endpoints = [
        {"name": "judgements", "max_id": 12000},
        {"name": "laws_of_federation", "max_id": 2000},
        {"name": "rules_of_court", "max_id": 7000},
        {"name": "forms_and_precedents", "max_id": 500},
        {"name": "articles_and_journals", "max_id": 500},
        {"name": "law_dictionary", "max_id": 2000},
        {"name": "legal_maxims", "max_id": 3000}
    ]

    for endpoint in endpoints:
        logger.info(f"Starting ingestion for endpoint: {endpoint['name']}")
        documents = asyncio.run(fetch_data(endpoint['name'], start_id=0, max_id=endpoint['max_id']))
        logger.info(f"Fetched {len(documents)} documents from {endpoint['name']}")
        ingest_to_weaviate(documents)
        logger.info(f"Completed ingestion for endpoint: {endpoint['name']}")

    # Handle State Rules of Court (POST request and paginated)
    state_rules_payload = {"page": 1}  # Initial payload for pagination
    logger.info("Starting ingestion for endpoint: state_rules")
    documents = asyncio.run(fetch_paginated_data(
        aiohttp.ClientSession(), "rules_of_court/state_rules", "State Rules of Court", state_rules_payload
    ))
    logger.info(f"Fetched {len(documents)} documents from state_rules")
    ingest_to_weaviate(documents)
    logger.info("Completed ingestion for endpoint: state_rules")

if __name__ == "__main__":
    ingest_legalpedia_data()





