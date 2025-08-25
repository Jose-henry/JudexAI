import os
import asyncio
import logging
import json
from collections import defaultdict
from bs4 import BeautifulSoup
from weaviate import Client as WeaviateClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from aiohttp import ClientSession

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

# Constants
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
LEGALPEDIA_USERNAME = os.getenv("LEGALPEDIA_USERNAME")
LEGALPEDIA_PASSWORD = os.getenv("LEGALPEDIA_PASSWORD")

# Connect to Weaviate client
try:
    weaviate_client = WeaviateClient(
        url=WEAVIATE_URL,
        additional_headers={"Content-Type": "application/json"}
    )
    logger.info("Connected to Weaviate successfully.")
except Exception as e:
    logger.error(f"Failed to connect to Weaviate: {e}")
    raise SystemExit("Exiting script due to connection failure.")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)

# Initialize embeddings using OpenAI's model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Caching mechanism
CACHE_DIR = "scrape_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(url):
    sanitized_url = url.replace("/", "_").replace(".", "_")
    return os.path.join(CACHE_DIR, f"{sanitized_url}.json")

def is_cache_valid(cache_path):
    if not os.path.exists(cache_path):
        return False
    stats = os.stat(cache_path)
    one_week_ago = (os.path.getmtime(cache_path) > (os.path.getmtime(cache_path) - 7 * 24 * 60 * 60))
    return stats.st_mtime > one_week_ago

def read_cache(url):
    cache_path = get_cache_path(url)
    if is_cache_valid(cache_path):
        with open(cache_path, "r") as cache_file:
            return json.load(cache_file)
    return None

def write_cache(url, content):
    cache_path = get_cache_path(url)
    with open(cache_path, "w") as cache_file:
        json.dump(content, cache_file)

async def login_to_legalpedia(session):
    login_url = "https://legalpediaresources.com/login"
    try:
        async with session.post(login_url, data={
            'email': LEGALPEDIA_USERNAME,
            'password': LEGALPEDIA_PASSWORD
        }) as response:
            if response.status == 200:
                logger.info("Logged in successfully.")
                return await response.text()
            else:
                logger.error(f"Login failed with status: {response.status}")
                return None
    except Exception as e:
        logger.error(f"Error logging into Legalpedia: {e}")
        return None

async def scrape_legalpedia_page(session, url):
    cache_content = read_cache(url)
    if cache_content:
        logger.info(f"Using cached content for {url}")
        return cache_content

    try:
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Extract data (example: extracting judgment titles and content)
                judgment_cards = soup.select(".judgment-list .card")
                content = [
                    {
                        "title": card.select_one(".card-title").get_text(strip=True),
                        "content": card.select_one(".card-content").get_text(strip=True)
                    }
                    for card in judgment_cards
                ]

                write_cache(url, content)
                return content
            else:
                logger.error(f"Failed to scrape {url} with status: {response.status}")
                return []
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return []

async def scrape_data_concurrently(urls):
    async with ClientSession() as session:
        tasks = [scrape_legalpedia_page(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

def ingest_to_weaviate(documents):
    try:
        WeaviateVectorStore.from_documents(documents, embeddings, client=weaviate_client)
        logger.info(f"Successfully ingested {len(documents)} documents into Weaviate.")
    except Exception as e:
        logger.error(f"Error during ingestion to Weaviate: {e}")

def main():
    urls = [
        "https://legalpediaresources.com/admin/judgements",
        "https://legalpediaresources.com/admin/articles",
        # Add more URLs as needed
    ]

    logger.info("Starting data scraping...")
    documents = asyncio.run(scrape_data_concurrently(urls))

    logger.info(f"Scraped {len(documents)} items. Starting ingestion...")
    chunks = []
    for doc in documents:
        text_chunks = text_splitter.split_text(doc["content"])
        chunks.extend([
            {
                "title": doc["title"],
                "content": chunk
            }
            for chunk in text_chunks
        ])

    ingest_to_weaviate(chunks)

    logger.info("Data ingestion complete.")

if __name__ == "__main__":
    main()
