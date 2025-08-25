import os
import requests
import tempfile
import hashlib
from dotenv import load_dotenv
from typing import Dict, List
from langchain_core.tools import Tool
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from llama_cloud_services import LlamaParse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import nest_asyncio

# Load environment variables from a .env files
load_dotenv()

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

# Global instances (embeddings and vectorstore removed)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=[
        "\n\n## ",  # Legal section headers
        "\n\n### ",  # Subsections
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ". ",    # Sentences
        " ",     # Words
        ""       # Characters
    ]
)

# Document cache to avoid reprocessing
document_cache: Dict[str, Dict] = {}

# Initialize LlamaParse conditionally
llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
llama_parser = None

if llama_api_key:
    llama_parser = LlamaParse(
        api_key=llama_api_key,
        num_workers=4,
        verbose=True,
        language="en",
        result_type="markdown"
    )

def download_file(url: str) -> bytes:
    """Download file from URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        raise Exception(f"Failed to download file: {str(e)}")

def process_pdf_with_llamaparse(url: str) -> tuple[str, int, List[Document]]:
    """
    Process PDF with LlamaParse and return documents for vector storage.
    Returns: (summary, number_of_chunks, document_objects)
    """
    if not llama_parser:
        return "LlamaParse not available - LLAMA_CLOUD_API_KEY not set", 0, []
    
    # Generate document ID from URL
    doc_id = hashlib.md5(url.encode()).hexdigest()
    
    # Check cache first - now returns documents for external storage
    if doc_id in document_cache:
        cached_info = document_cache[doc_id]
        # Retrieve documents from cache if available
        if 'documents' in cached_info:
            summary = f"Document already processed and cached. Contains {cached_info['chunks']} chunks of legal content."
            return summary, cached_info['chunks'], cached_info['documents']
        return f"Document already processed and cached with {cached_info['chunks']} chunks.", cached_info['chunks'], []
    
    try:
        # Download PDF to temporary file
        pdf_bytes = download_file(url)
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name
        
        try:
            # Parse with LlamaParse - handle this synchronously to avoid async issues
            result = llama_parser.parse(temp_path)
            
            # Get markdown documents
            if hasattr(result, 'get_markdown_documents'):
                markdown_docs = result.get_markdown_documents(split_by_page=True)
            else:
                # Fallback for different LlamaParse versions
                markdown_docs = result
            
            # Convert to LangChain documents and split
            documents = []
            full_text = ""
            
            for i, doc in enumerate(markdown_docs):
                if hasattr(doc, 'text'):
                    page_text = doc.text
                    page_metadata = getattr(doc, 'metadata', {})
                else:
                    # Handle different document structures
                    page_text = str(doc)
                    page_metadata = {'page': i}
                    
                full_text += f"\n\n--- Page {page_metadata.get('page', i)} ---\n\n"
                full_text += page_text
            
            # Create intelligent chunks
            chunks = text_splitter.split_text(full_text)
            
            for i, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": url,
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_{i}",
                        "chunk_index": i,
                        "document_type": "legal_pdf",
                        "total_chunks": len(chunks)
                    }
                ))
            
            # Cache the document metadata with documents
            import time
            document_cache[doc_id] = {
                'url': url,
                'chunks': len(documents),
                'documents': documents,  # Store documents for external access
                'processed_at': time.time()
            }
            
            # Create summary
            summary = f"Successfully processed PDF: {url}\n"
            summary += f"- Document contains {len(documents)} chunks of legal content\n"
            summary += f"- Content includes legal text, cases, provisions, and analysis\n"
            summary += f"- Document is now available for vector storage\n"
            
            return summary, len(documents), documents
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        return f"Failed to process PDF: {str(e)}", 0, []

def pdf_rag_function(input_text: str) -> str:
    """
    Main function that handles PDF processing only.
    Now only processes PDFs, searching is handled by JudgeAgent.
    """
    try:
        input_text = input_text.strip()
        
        # Only process URLs, searching is handled by JudgeAgent
        if input_text.startswith(('http://', 'https://', 'www.')):
            summary, chunks, documents = process_pdf_with_llamaparse(input_text)
            return summary
        else:
            return "Search functionality has been moved to the JudgeAgent. Please use the agent's built-in search capabilities."
    except Exception as e:
        return f"Error in PDF processing function: {str(e)}"

# Function to get processed documents for migration
def get_processed_documents() -> List[Document]:
    """Get all processed documents for vector store migration."""
    all_documents = []
    for cache_info in document_cache.values():
        if 'documents' in cache_info:
            all_documents.extend(cache_info['documents'])
    return all_documents

# Function to get document cache info
def get_document_cache() -> Dict[str, Dict]:
    """Get the document cache for migration."""
    return document_cache.copy()

# Create the tool
pdf_rag_tool = Tool(
    name="llama_pdf_document_processor",
    description="""Process PDF documents from URLs for legal analysis. 
    
    Use this tool ONLY to process PDF documents by passing a URL (starting with http:// or https://).
    After processing, the content becomes available for search through the agent's internal vector store.
    
    Searching processed documents is now handled by the agent itself, not through this tool.
    
    Example: "https://example.com/legal-document.pdf"
    """,
    func=pdf_rag_function
)