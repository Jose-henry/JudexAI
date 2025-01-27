import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings
import weaviate
import openai
from langchain_weaviate.vectorstores import WeaviateVectorStore


# Load environment variables
WEAVIATE_SCHEME = os.getenv("WEAVIATE_SCHEME", "http")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "157.245.215.92:8080")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Weaviate Python client.
client = weaviate.connect_to_local(
    host="157.245.215.92",  # Use a string to specify the host
    port=8080,
    grpc_port=50051,
)


print("Connected to Weaviate?", client.is_ready())




# Initialize embeddings using OpenAI's model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")




vector_store = WeaviateVectorStore(text_key="text", embedding=embeddings, client=client, index_name="Article")



query = "ABUBAKAR USMAN V HABILA MATHIAS"
docs = vector_store.similarity_search(query, k=3)

print(docs)
client.close()
