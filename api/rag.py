import weaviate
from langchain_core.tools import tool
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
from weaviate.classes.init import AdditionalConfig, Timeout

@tool
def rag_tool(query: str):
    """
    Retrieves context from a Weaviate vector database based on the query provided.
    
    Args:
        query (str): The query to search for relevant documents in the Weaviate vector store.

    Returns:
        dict: A dictionary containing the most relevant documents and their scores.
    """
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
        print("Connected to Weaviate successfully.")
    except weaviate.exceptions.WeaviateConnectionError as e:
        print(f"Failed to connect to Weaviate: {e}")
        raise SystemExit("Exiting script due to connection failure.")

    # Initialize embeddings using OpenAI's model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Use WeaviateVectorStore for similarity search
    try:
        vector_store = WeaviateVectorStore(text_key="content", embedding=embeddings, client=weaviate_client, index_name="Article")
        results = vector_store.similarity_search_with_score(query, k=5)

        # Format results
        formatted_results = [
            {
                "content": result[0].page_content,
                "metadata": result[0].metadata,
                "score": result[1]
            }
            for result in results
        ]
        return {"query": query, "results": formatted_results}
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return {"error": "Failed to retrieve documents/no results", "details": str(e)}
