# api/docling_pdf_rag_tool.py#
# best with locally hosted model

import os
import requests
import tempfile
import hashlib
from dotenv import load_dotenv
from typing import Dict, List
from langchain_core.tools import Tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import PdfFormatOption
from io import BytesIO
import time

# Load environment variables from a .env files
load_dotenv()

# Global instances
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)

vectorstore = InMemoryVectorStore(embedding=embeddings)

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

# Initialize Docling converter with optimized settings
def get_docling_converter():
    """Get a configured Docling converter instance."""
    # Configure pipeline options for better legal document processing
    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,  # Enable table structure recognition
        do_ocr=True,  # Enable OCR for scanned documents
    )
    
    # Use accurate TableFormer mode for better table extraction
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.table_structure_options.do_cell_matching = True  # Map to PDF cells
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return converter

def download_file(url: str) -> bytes:
    """Download file from URL."""
    try:
        response = requests.get(url, timeout=60)  # Increased timeout for large files
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        raise Exception(f"Failed to download file: {str(e)}")

def process_pdf_with_docling(url: str) -> tuple[str, int, List[str]]:
    """
    Process PDF with Docling and add to InMemoryVectorStore.
    Returns: (summary, number_of_chunks, relevant_snippets)
    """
    # Generate document ID from URL
    doc_id = hashlib.md5(url.encode()).hexdigest()
    
    # Check cache first
    if doc_id in document_cache:
        cached_info = document_cache[doc_id]
        # Retrieve some content from vector store for summary
        try:
            results = vectorstore.similarity_search("main content summary", k=3)
            snippets = [doc.page_content[:300] + "..." for doc in results if doc.metadata.get("doc_id") == doc_id]
            return f"Document already processed and cached. Contains {cached_info['chunks']} chunks of legal content.", cached_info['chunks'], snippets
        except:
            return f"Document already processed and cached with {cached_info['chunks']} chunks.", cached_info['chunks'], []
    
    try:
        # Download PDF
        pdf_bytes = download_file(url)
        
        # Create DocumentStream from bytes
        pdf_stream = BytesIO(pdf_bytes)
        source = DocumentStream(name="document.pdf", stream=pdf_stream)
        
        # Get Docling converter
        converter = get_docling_converter()
        
        # Convert the document
        result = converter.convert(source, max_num_pages=500, max_file_size=50*1024*1024)  # 50MB limit
        
        # Get the Docling document
        docling_doc = result.document
        
        # Export to markdown for better text processing
        full_text = docling_doc.export_to_markdown()
        
        # If markdown is empty, try to get text content directly
        if not full_text.strip():
            # Fallback to text content from document elements
            full_text = ""
            for element in docling_doc.texts:
                if hasattr(element, 'text'):
                    full_text += element.text + "\n"
        
        # Add document metadata as header
        metadata_header = f"# Document Analysis\n\n"
        metadata_header += f"**Source URL:** {url}\n"
        metadata_header += f"**Processing Tool:** Docling\n"
        metadata_header += f"**Document Type:** PDF\n"
        metadata_header += f"**Processed At:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        metadata_header += "---\n\n"
        
        full_text = metadata_header + full_text
        
        # Create intelligent chunks
        chunks = text_splitter.split_text(full_text)
        
        # Create LangChain documents
        documents = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only add non-empty chunks
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": url,
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_{i}",
                        "chunk_index": i,
                        "document_type": "legal_pdf",
                        "total_chunks": len(chunks),
                        "processing_tool": "docling",
                        "processed_at": time.time()
                    }
                ))
        
        # Add to InMemory vector store
        if documents:
            document_ids = vectorstore.add_documents(documents)
            
            # Cache the document metadata
            document_cache[doc_id] = {
                'url': url,
                'chunks': len(documents),
                'document_ids': document_ids,
                'processed_at': time.time()
            }
            
            # Create summary and get relevant snippets
            summary = f"✅ Successfully processed PDF with Docling: {url}\n"
            summary += f"📄 Document contains {len(documents)} chunks of content\n"
            summary += f"🔍 Content includes text, tables, and structured elements\n"
            summary += f"💾 Document is now searchable in the knowledge base\n"
            
            # Get first few chunks as preview
            preview_chunks = documents[:3]
            snippets = []
            for doc in preview_chunks:
                snippet = doc.page_content[:300].strip()
                if snippet and not snippet.startswith("# Document Analysis"):
                    snippets.append(snippet + "...")
            
            if snippets:
                summary += f"👁️ Preview: {snippets[0][:200]}..."
            
            return summary, len(documents), snippets
        else:
            return "❌ PDF processed but no content could be extracted", 0, []
            
    except Exception as e:
        return f"❌ Failed to process PDF with Docling: {str(e)}", 0, []

def search_processed_documents(query: str, k: int = 5) -> str:
    """Search through processed documents."""
    try:
        if len(vectorstore.store) == 0:
            return "⚠️ No documents have been processed yet. Please provide a PDF URL first."
            
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        if not results:
            return "🔍 No relevant content found in the processed documents for your query."
        
        response = f"🎯 Found {len(results)} relevant sections in the processed documents:\n\n"
        
        for i, (doc, score) in enumerate(results, 1):
            content = doc.page_content[:500].strip()
            source = doc.metadata.get('source', 'Unknown source')
            chunk_info = f"Chunk {doc.metadata.get('chunk_index', '?')} of {doc.metadata.get('total_chunks', '?')}"
            processing_tool = doc.metadata.get('processing_tool', 'unknown')
            
            response += f"## {i}. **{chunk_info}** (Relevance: {1-score:.2f})\n"
            response += f"**Source:** {source}\n"
            response += f"**Processed with:** {processing_tool}\n"
            response += f"**Content:**\n{content}...\n\n"
            response += "---\n\n"
        
        return response
        
    except Exception as e:
        return f"❌ Search failed: {str(e)}"

def pdf_rag_function(input_text: str) -> str:
    """
    Main function that handles PDF processing and RAG operations with Docling.
    Input can be either a URL (for processing) or a query (for searching).
    """
    try:
        input_text = input_text.strip()
        
        # Check if input is a URL
        if input_text.startswith(('http://', 'https://', 'www.')):
            # Process PDF with Docling
            summary, chunks, snippets = process_pdf_with_docling(input_text)
            return summary
        else:
            # Search processed documents
            return search_processed_documents(input_text)
    except Exception as e:
        return f"❌ Error in PDF RAG function: {str(e)}"

# Create the tool with updated description
docling_pdf_rag_tool = Tool(
    name="pdf_document_processor",
    description="""Process PDF documents from URLs and search through their content using Docling. 
    
    Use this tool in two ways:
    1. **To process a PDF**: Pass a URL (starting with http:// or https://) to download, parse, and index the PDF content using Docling's advanced document understanding
    2. **To search processed PDFs**: Pass a search query to find relevant information from previously processed documents
    
    Docling provides superior PDF processing with:
    - Advanced table structure recognition and extraction
    - OCR capabilities for scanned documents  
    - Better handling of complex layouts and formatting
    - Structured content extraction (headers, paragraphs, lists, tables)
    - High-quality markdown conversion
    
    This tool can handle legal documents, contracts, case files, research papers, and other PDF content. 
    After processing, the content becomes searchable and can provide detailed analysis of the document's contents.
    
    Examples:
    - Process: "https://example.com/legal-document.pdf"  
    - Search: "What are the key provisions about contract termination?"
    - Search: "Extract all table data related to financial terms"
    """,
    func=pdf_rag_function
)

def get_vectorstore_status() -> dict:
    """Get current status of the vector store."""
    return {
        "total_documents": len(vectorstore.store),
        "cached_pdfs": len(document_cache),
        "cached_urls": [info['url'] for info in document_cache.values()],
        "processing_tool": "docling"
    }

def clear_vectorstore():
    """Clear all processed documents."""
    global document_cache
    try:
        vectorstore.delete_collection()
        document_cache.clear()
        return "🧹 All processed documents cleared successfully"
    except:
        # Fallback if delete_collection not available
        vectorstore.store.clear()
        document_cache.clear()
        return "🧹 Vector store cleared successfully"