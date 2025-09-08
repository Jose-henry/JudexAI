import os
import hashlib
from typing import Dict, List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

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

# Initialize embeddings and in-memory vector store for PDF knowledge

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
pdf_vectorstore = InMemoryVectorStore(embedding=embeddings)

# Track which document IDs have been stored to avoid duplicate inserts
_stored_doc_ids = set()

def _ensure_cached_and_indexed(url: str, documents: List[Document]) -> str:
    """Store processed documents into cache and vector store, avoiding duplicates."""
    doc_id = hashlib.md5(url.encode()).hexdigest()
    import time
    # Cache
    document_cache[doc_id] = {
        'url': url,
        'chunks': len(documents),
        'documents': documents,
        'processed_at': time.time()
    }
    # Vector store (dedupe per doc_id)
    try:
        if doc_id not in _stored_doc_ids and documents:
            pdf_vectorstore.add_documents(documents)
            _stored_doc_ids.add(doc_id)
    except Exception:
        pass
    # Summary
    summary = f"Successfully processed PDF: {url}\n"
    summary += f"- Document contains {len(documents)} chunks of legal content\n"
    summary += f"- Document has been embedded and stored for internal search\n"
    try:
        total_indexed = len(pdf_vectorstore.store)
        summary += f"- Total indexed documents: {total_indexed}\n"
    except Exception:
        pass
    return summary

def store_processed_pdf(url: str, documents: List[Document]) -> str:
    """Public API: store externally processed PDF chunks into cache and vector store."""
    return _ensure_cached_and_indexed(url, documents)

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

# PDF knowledge search helpers (used by JudgeAgent)
def has_pdf_knowledge() -> bool:
    try:
        return len(pdf_vectorstore.store) > 0
    except Exception:
        return False

def search_pdf_knowledge(query: str, k: int = 5) -> str:
    """Search through processed PDF knowledge stored in the in-memory vector store."""
    try:
        if not has_pdf_knowledge():
            return "No documents have been processed yet. Please provide a PDF URL first."
        results = pdf_vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return "No relevant content found in the processed documents."
        response = f"Found {len(results)} relevant sections in the processed documents:\n\n"
        for i, (doc, score) in enumerate(results, 1):
            content = doc.page_content[:500].strip()
            source = doc.metadata.get('source', 'Unknown source')
            chunk_info = f"Chunk {doc.metadata.get('chunk_index', '?')} of {doc.metadata.get('total_chunks', '?')}"
            response += f"{i}. **{chunk_info}** (Relevance: {1-score:.2f})\n"
            response += f"Source: {source}\n"
            response += f"Content: {content}...\n\n"
        return response
    except Exception as e:
        return f"Search failed: {str(e)}"

# Return top matching documents for synthesis/streaming answers
def retrieve_pdf_knowledge(query: str, k: int = 5) -> List[Document]:
    try:
        if not has_pdf_knowledge():
            return []
        results = pdf_vectorstore.similarity_search_with_score(query, k=k)
        return [doc for doc, _ in results]
    except Exception:
        return []

# Note: PDF processing tool has been removed from this module.