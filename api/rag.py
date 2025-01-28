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
        dict: A dictionary containing either the relevant documents and their scores,
              or an error message indicating whether to use web search or show database unavailability.
    """
    # Check if query explicitly requests Legalpedia
    legalpedia_explicitly_requested = any(term.lower() in query.lower() 
                                        for term in ["legalpedia", "rag tool", "database"])

    # Connect to the Weaviate client
    try:
        weaviate_client = weaviate.connect_to_local(
            host="157.245.215.92",
            port=8080,
            grpc_port=50051,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30, query=60, insert=120)
            ),
        )
        
        # Initialize embeddings using OpenAI's model
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Use WeaviateVectorStore for similarity search
        vector_store = WeaviateVectorStore(
            text_key="text",
            embedding=embeddings,
            client=weaviate_client,
            index_name="Article"
        )
        results = vector_store.similarity_search_with_score(query, k=5)

        # Format results
        formatted_results = [
            {
                "content": result[0].page_content,
                "metadata": result[0].metadata or {},
                "score": result[1]
            }
            for result in results
        ]
        return {"query": query, "results": formatted_results}

    except weaviate.exceptions.WeaviateConnectionError as e:
        if legalpedia_explicitly_requested:
            return {
                "error": "database_unavailable",
                "message": "I apologize for the inconvenience, but I cannot access the Legalpedia database at the moment."
            }
        else:
            return {
                "error": "use_web_search",
                "message": "The Legalpedia database is currently unavailable. I'll search the web for relevant information instead."
            }
            
    except Exception as e:
        if legalpedia_explicitly_requested:
            return {
                "error": "database_unavailable",
                "message": "I apologize for the inconvenience, but I cannot access the Legalpedia database at the moment."
            }
        else:
            return {
                "error": "use_web_search",
                "message": "Unable to retrieve information from Legalpedia. I'll search the web for relevant information instead."
            }
            
    finally:
        try:
            if 'weaviate_client' in locals():
                weaviate_client.close()
        except:
            pass