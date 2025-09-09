from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
import base64
import io
import os
from PIL import Image
import requests
import hashlib
import json
from api.JudgeAgent2 import langgraph_agent_executor, stream_agent_response_async
from api.llama_pdf_rag_tool import store_processed_pdf
from llama_cloud_services import LlamaParse
import tempfile
import requests
import hashlib
import asyncio
import uuid
import logging

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("preprocess")

# Enhanced request body schema to handle URLs
class MessageContent(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    messages: List[MessageContent]
    upload_ids: Optional[List[str]] = None  # Preprocessed uploads to include
    session_id: Optional[str] = None  # Session that groups preprocessed uploads

class QueryResponse(BaseModel):
    response: str
    cached: bool = False

# In-memory cache for responses
response_cache: Dict[str, str] = {}

# In-memory store for preprocessed uploads (grouped by session_id and upload_id)
# Structure: { session_id: { upload_id: { 'type': 'image'|'document', 'url': str, 'data': Any, 'metadata': Dict[str, Any] } } }
preprocessed_store: Dict[str, Dict[str, Dict[str, Any]]] = {}

def generate_cache_key(messages: List[MessageContent], upload_ids: Optional[List[str]] = None) -> str:
    """
    Generate a unique cache key for the request messages.
    
    Args:
        messages (List[MessageContent]): List of message contents
        
    Returns:
        str: Cache key
    """
    # Convert messages to a consistent string representation
    message_str = json.dumps([{
        'role': msg.role,
        'content': msg.content,
    } for msg in messages], sort_keys=True)
    if upload_ids:
        message_str += json.dumps(sorted(upload_ids))
    
    # Generate hash
    return hashlib.sha256(message_str.encode()).hexdigest()

async def download_file(url: str) -> bytes:
    """
    Download file from URL.
    
    Args:
        url (str): URL of the file to download
        
    Returns:
        bytes: Content of the downloaded file
    """
    logger.info(f"Downloading file: {url}")
    try:
        response = await asyncio.to_thread(requests.get, url,  timeout=10)
        response.raise_for_status()
        content = response.content
        logger.info(f"Downloaded {len(content)} bytes from: {url}")
        return content
    except requests.RequestException as e:
        logger.error(f"Download failed for {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")

async def process_image_url(url: str) -> str:
    """
    Process an image from URL and convert to base64.
    
    Args:
        url (str): URL of the image file
        
    Returns:
        str: Base64-encoded image string
    """
    logger.info(f"Image processing started: {url}")
    image_bytes = await download_file(url)
    try:
        img = Image.open(io.BytesIO(image_bytes))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        logger.info(f"Image processing succeeded: {url} (base64 length={len(b64)})")
        return b64
    except Exception as e:
        logger.error(f"Image processing failed for {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

async def process_document_url(url: str) -> List[Any]:
    """
    Process a document (pdf/docx/xlsx/pptx/txt/md/html etc.) via LlamaParse and return
    a list of LangChain Document chunks with metadata.
    """
    llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not llama_api_key:
        raise HTTPException(status_code=500, detail="LLAMA_CLOUD_API_KEY not set")
    parser = LlamaParse(
        api_key=llama_api_key,
        num_workers=4,
        verbose=True,
        language="en",
        result_type="markdown"
    )
    logger.info(f"Document processing started: {url}")
    try:
        file_bytes = await download_file(url)
    except HTTPException:
        raise

    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(str(url).split('?')[0])[1] or '.bin', delete=False) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        logger.info(f"LlamaParse parsing started: {url}")
        result = await asyncio.to_thread(parser.parse, temp_path)
        logger.info(f"LlamaParse parsing completed: {url}")
        if hasattr(result, 'get_markdown_documents'):
            markdown_docs = result.get_markdown_documents(split_by_page=True)
        else:
            markdown_docs = result

        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n## ", "\n\n### ", "\n\n", "\n", ". ", " ", ""]
        )
        full_text = ""
        for i, doc in enumerate(markdown_docs):
            if hasattr(doc, 'text'):
                page_text = doc.text
                page_metadata = getattr(doc, 'metadata', {})
            else:
                page_text = str(doc)
                page_metadata = {'page': i}
            full_text += f"\n\n--- Page {page_metadata.get('page', i)} ---\n\n"
            full_text += page_text

        chunks = text_splitter.split_text(full_text)
        doc_id = hashlib.md5(url.encode()).hexdigest()
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": url,
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_{i}",
                    "chunk_index": i,
                    "document_type": "document",
                    "total_chunks": len(chunks)
                }
            ))
        logger.info(f"Document chunking completed: {url} (chunks={len(documents)})")
        return documents
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass

## Removed legacy process_pdf_url; unified into process_document_url

# -------------------------
# Preupload URL processing
# -------------------------

class PreprocessFileItem(BaseModel):
    url: HttpUrl
    kind: Optional[str] = "auto"  # 'auto' | 'image' | 'document'

class PreprocessRequest(BaseModel):
    files: List[PreprocessFileItem]
    session_id: Optional[str] = None

class PreprocessItemResult(BaseModel):
    upload_id: str
    url: HttpUrl
    type: str
    status: str
    preview: Optional[str] = None

class PreprocessResponse(BaseModel):
    session_id: str
    items: List[PreprocessItemResult]

def _detect_type(url: str, kind: Optional[str]) -> str:
    if kind and kind in ["image", "document"]:
        return kind
    ext = str(url).split('?')[0].split('.')[-1].lower()
    if ext in ["png", "jpg", "jpeg", "gif", "webp", "bmp"]:
        return "image"
    return "document"

def _get_or_create_session(session_id: Optional[str]) -> str:
    sid = session_id or str(uuid.uuid4())
    if sid not in preprocessed_store:
        preprocessed_store[sid] = {}
    return sid

@app.post("/files/preprocess-urls", response_model=PreprocessResponse)
async def preprocess_urls(req: PreprocessRequest):
    """Accepts file URLs, processes images/documents concurrently, and stores results under session/upload IDs."""
    session_id = _get_or_create_session(req.session_id)
    results: List[PreprocessItemResult] = []

    # Limit concurrency to 5 (matches UI limit)
    semaphore = asyncio.Semaphore(5)

    async def handle_one(index: int, f: PreprocessFileItem) -> PreprocessItemResult:
        async with semaphore:
            file_type = _detect_type(str(f.url), f.kind)
            upload_id = str(uuid.uuid4())
            try:
                if file_type == "image":
                    logger.info(f"[session={session_id}] [upload_id={upload_id}] Image preprocessing queued: {f.url}")
                    image_base64 = await process_image_url(str(f.url))
                    preprocessed_store[session_id][upload_id] = {
                        'type': 'image',
                        'url': str(f.url),
                        'data': image_base64,
                        'metadata': {
                            'file_index': index
                        }
                    }
                    logger.info(f"[session={session_id}] [upload_id={upload_id}] Image preprocessing success: {f.url}")
                    return PreprocessItemResult(
                        upload_id=upload_id,
                        url=f.url,
                        type='image',
                        status='processed',
                        preview='image/png;base64,' + image_base64[:64] + '...'
                    )
                else:
                    logger.info(f"[session={session_id}] [upload_id={upload_id}] Document preprocessing queued: {f.url}")
                    documents = await process_document_url(str(f.url))
                    preprocessed_store[session_id][upload_id] = {
                        'type': 'document',
                        'url': str(f.url),
                        'data': documents,
                        'metadata': {
                            'file_index': index,
                            'total_chunks': len(documents)
                        }
                    }
                    logger.info(f"[session={session_id}] [upload_id={upload_id}] Document preprocessing success: {f.url} (chunks={len(documents)})")
                    preview_text = documents[0].page_content[:200] + ('...' if len(documents[0].page_content) > 200 else '') if documents else ''
                    return PreprocessItemResult(
                        upload_id=upload_id,
                        url=f.url,
                        type='document',
                        status='processed',
                        preview=preview_text
                    )
            except Exception as e:
                logger.error(f"[session={session_id}] [upload_id={upload_id}] Preprocessing failed: {f.url} error={e}")
                return PreprocessItemResult(
                    upload_id=upload_id,
                    url=f.url,
                    type=file_type,
                    status=f"failed: {str(e)}",
                    preview=None
                )

    tasks = [handle_one(i, f) for i, f in enumerate(req.files)]
    results = await asyncio.gather(*tasks)
    return PreprocessResponse(session_id=session_id, items=list(results))

# Chat history to maintain conversation context (moved to global scope for persistence)
global_chat_history = []

@app.post("/judge", response_model=QueryResponse)
async def query_judge_agent(request: QueryRequest):
    """
    Endpoint to interact with the Lextech AI Judge Assistant using file URLs.
    
    Args:
        request (QueryRequest): Request containing messages and optional file URLs
        
    Returns:
        QueryResponse: AI's response to the content
    """
    global global_chat_history
    
    try:
        # Generate cache key for the request
        cache_key = generate_cache_key(request.messages, request.upload_ids)

        # Check if response is in cache
        if cache_key in response_cache:
            return QueryResponse(
                response=response_cache[cache_key],
                cached=True
            )

        # Process messages and files
        formatted_messages = []
        # Collect hints for processed document sources
        processed_doc_hints: List[str] = []
        # If upload_ids provided, index documents and collect images
        if request.upload_ids and request.session_id:
            session_bucket = preprocessed_store.get(request.session_id, {})
            for up_id in request.upload_ids:
                item = session_bucket.get(up_id)
                if not item:
                    logger.warning(f"[session={request.session_id}] Missing upload_id={up_id} in store")
                    continue
                if item.get('type') == 'document':
                    # Index now into RAG
                    try:
                        docs = item.get('data') or []
                        if docs:
                            meta = item.setdefault('metadata', {})
                            if not meta.get('indexed'):
                                logger.info(f"[session={request.session_id}] Indexing document upload_id={up_id} url={item.get('url','')} chunks={len(docs)}")
                                summary = store_processed_pdf(item.get('url', ''), docs)
                                meta['indexed'] = True
                                logger.info(f"[session={request.session_id}] Indexed document upload_id={up_id}")
                            processed_doc_hints.append(item.get('url', ''))
                    except Exception:
                        logger.exception(f"[session={request.session_id}] Indexing failed for upload_id={up_id}")
                        pass
                elif item.get('type') == 'image':
                    # Attach image block to messages later
                    img_b64 = item.get('data')
                    meta = item.setdefault('metadata', {})
                    if img_b64 and not meta.get('attached'):
                        logger.info(f"[session={request.session_id}] Attaching image upload_id={up_id}")
                        formatted_messages.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        })
                        meta['attached'] = True
        
        for msg in request.messages:
            # Start with the original message content
            enhanced_content = msg.content
            
            # Add text content
            formatted_messages.append({
                "type": "text",
                "text": enhanced_content
            })
            
            # Add hints for any preprocessed documents only if the query seems document-related
            if processed_doc_hints:
                # Check if the user's query seems to be about documents
                query_lower = msg.content.lower()
                document_keywords = [
                    'document', 'pdf', 'contract', 'agreement', 'clause', 'section', 
                    'article', 'paragraph', 'provision', 'uploaded', 'file', 'text',
                    'content', 'summarize', 'explain', 'what does', 'what is in',
                    'from the', 'in the document', 'according to'
                ]
                
                is_document_query = any(keyword in query_lower for keyword in document_keywords)
                
                if is_document_query:
                    hint = "\n\n(Use processed document context from: " + ", ".join(processed_doc_hints) + ")"
                    enhanced_content += hint
                    formatted_messages[-1]["text"] = enhanced_content

            # Legacy msg.file_url support removed; rely on preupload upload_ids

        # Format final message structure for LangChain (like original working code)
        chat_messages = [("human", formatted_messages)]
        
        # Include chat history for continuity
        recent_history = global_chat_history[-10:] if len(global_chat_history) > 10 else global_chat_history
        messages_with_history = recent_history + chat_messages
        
        config = {"configurable": {"thread_id": "main-conversation"}}

        # Collect agent response (exactly like original working code)
        full_response = ""
        for chunk in langgraph_agent_executor.stream({"messages": messages_with_history}, config):
            if 'agent' in chunk and chunk['agent'].get('messages'):
                for msg in chunk['agent']['messages']:
                    if hasattr(msg, 'content') and msg.content:
                        full_response += msg.content

        # Cache the response
        response_cache[cache_key] = full_response

        # Update global chat history
        last_msg = request.messages[-1] if request.messages else None
        if last_msg:
            content_desc = last_msg.content
            global_chat_history.extend([
                ("human", content_desc),
                ("ai", full_response)
            ])
            
            # Trim history to prevent memory bloat
            if len(global_chat_history) > 20:
                global_chat_history = global_chat_history[-20:]

        return QueryResponse(response=full_response, cached=False)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/judge/stream")
async def query_judge_agent_stream(request: QueryRequest):
    """
    Enhanced streaming endpoint with tool call notifications and loading indicators.
    """
    async def generate_stream():
        global global_chat_history
        try:
            # Process messages and files (including preprocessed uploads)
            formatted_messages = []
            processed_doc_hints: List[str] = []
            if request.upload_ids and request.session_id:
                session_bucket = preprocessed_store.get(request.session_id, {})
                for up_id in request.upload_ids:
                    item = session_bucket.get(up_id)
                    if not item:
                        logger.warning(f"[session={request.session_id}] Missing upload_id={up_id} in store")
                        continue
                    if item.get('type') == 'document':
                        try:
                            docs = item.get('data') or []
                            if docs:
                                meta = item.setdefault('metadata', {})
                                if not meta.get('indexed'):
                                    logger.info(f"[session={request.session_id}] Indexing document upload_id={up_id} url={item.get('url','')} chunks={len(docs)}")
                                    summary = store_processed_pdf(item.get('url', ''), docs)
                                    meta['indexed'] = True
                                processed_doc_hints.append(item.get('url', ''))
                                logger.info(f"[session={request.session_id}] Indexed document upload_id={up_id}")
                        except Exception:
                            logger.exception(f"[session={request.session_id}] Indexing failed for upload_id={up_id}")
                            pass
                    elif item.get('type') == 'image':
                        img_b64 = item.get('data')
                        meta = item.setdefault('metadata', {})
                        if img_b64 and not meta.get('attached'):
                            logger.info(f"[session={request.session_id}] Attaching image upload_id={up_id}")
                            formatted_messages.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                            })
                            meta['attached'] = True
            
            for msg in request.messages:
                enhanced_content = msg.content
                
                formatted_messages.append({
                    "type": "text",
                    "text": enhanced_content
                })
                
                # Add hints for any preprocessed documents only if the query seems document-related
                if processed_doc_hints:
                    # Check if the user's query seems to be about documents
                    query_lower = msg.content.lower()
                    document_keywords = [
                        'document', 'pdf', 'contract', 'agreement', 'clause', 'section', 
                        'article', 'paragraph', 'provision', 'uploaded', 'file', 'text',
                        'content', 'summarize', 'explain', 'what does', 'what is in',
                        'from the', 'in the document', 'according to'
                    ]
                    
                    is_document_query = any(keyword in query_lower for keyword in document_keywords)
                    
                    if is_document_query:
                        hint = "\n\n(Use processed document context from: " + ", ".join(processed_doc_hints) + ")"
                        formatted_messages[-1]["text"] = enhanced_content + hint

                # Legacy msg.file_url support removed; rely on preupload upload_ids

            chat_messages = [("human", formatted_messages)]
            recent_history = global_chat_history[-10:] if len(global_chat_history) > 10 else global_chat_history
            messages_with_history = recent_history + chat_messages
            config = {"configurable": {"thread_id": "main-conversation"}}

            # Enhanced streaming with tool notifications
            full_response = ""
            async for chunk in stream_agent_response_async({"messages": messages_with_history}, config):
                if chunk:
                    full_response += chunk
                    
                    # Detect if this is a tool notification
                    is_tool_notification = any(indicator in chunk for indicator in ['📄', '🔍', '🔎', '⏳', '✅'])
                    
                    # Send chunk with appropriate type
                    chunk_type = 'tool_notification' if is_tool_notification else 'chunk'
                    yield f"data: {json.dumps({'content': chunk, 'type': chunk_type})}\n\n"
                    
                    # Add small delay to make loading indicators visible
                    if is_tool_notification:
                        await asyncio.sleep(0.5)
                    else:
                        await asyncio.sleep(0.01)
            
            # Update chat history
            last_msg = request.messages[-1] if request.messages else None
            if last_msg:
                content_desc = last_msg.content
                global_chat_history.extend([
                    ("human", content_desc),
                    ("ai", full_response)
                ])
                
                if len(global_chat_history) > 20:
                    global_chat_history = global_chat_history[-20:]
            
            yield f"data: {json.dumps({'content': '', 'type': 'end'})}\n\n"
            
        except Exception as e:
            error_data = json.dumps({'error': str(e), 'type': 'error'})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        }
    )

@app.post("/clear-cache")
async def clear_cache():
    """Clear the in-memory response cache and chat history"""
    global global_chat_history
    response_cache.clear()
    global_chat_history.clear()
    # Also clear preprocessed store
    preprocessed_store.clear()
    return {"message": "Cache, preprocessed store, and chat history cleared successfully"}

@app.get("/cache-status")
async def get_cache_status():
    """Get current cache status"""
    return {
        "cache_size": len(response_cache),
        "chat_history_size": len(global_chat_history),
        "cached_keys": list(response_cache.keys())[:5]  # Only show first 5 keys
    }

@app.get("/chat-history")
async def get_chat_history():
    """Get current chat history (last 10 messages)"""
    return {
        "chat_history": global_chat_history[-10:] if len(global_chat_history) > 10 else global_chat_history,
        "total_messages": len(global_chat_history)
    }

@app.post("/reset-conversation")
async def reset_conversation():
    """Reset the conversation thread"""
    global global_chat_history
    global_chat_history.clear()
    return {"message": "Conversation reset successfully"}

@app.get("/test-stream")
async def test_stream():
    """Test endpoint to verify streaming functionality"""
    async def generate_test_stream():
        for i in range(10):
            newline = '\n'
            yield f"data: {json.dumps({'content': f'Test chunk {i+1}{newline}', 'type': 'chunk'})}\n\n"
            await asyncio.sleep(0.5)
        yield f"data: {json.dumps({'content': '', 'type': 'end'})}\n\n"
    
    return StreamingResponse(
        generate_test_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.get("/", response_class=HTMLResponse)
async def get_test_page():
    """Simple HTML page to test streaming functionality"""
    html_content = """
    <!DOCTYPE html>
<html>
<head>
    <title>Judge Agent Streaming Test - Fixed</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .input-group { margin: 20px 0; }
        input[type="text"], textarea { width: 100%; padding: 12px; margin: 5px 0; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
        button { padding: 12px 24px; background: #007bff; color: white; border: none; cursor: pointer; border-radius: 5px; margin: 5px; font-size: 14px; }
        button:hover { background: #0056b3; }
        button:disabled { background: #6c757d; cursor: not-allowed; }
        .btn-streaming { background: #28a745; }
        .btn-streaming:hover { background: #218838; }
        .btn-streaming:disabled { background: #6c757d; }
        .response { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; min-height: 100px; background: #f8f9fa; }
        .status { margin: 10px 0; padding: 10px; border-radius: 5px; font-size: 12px; }
        .status.streaming { background: #d1ecf1; color: #0c5460; }
        .status.complete { background: #d4edda; color: #155724; }
        .status.error { background: #f8d7da; color: #721c24; }
        
        /* Tool notification styles */
        .tool-notification {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 8px 12px;
            margin: 8px 0;
            border-radius: 4px;
            font-style: italic;
            color: #856404;
        }
        
        /* Loading animation */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading-spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #007bff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 8px;
        }
        
        .tool-working {
            background: #e8f4fd;
            border-left: 4px solid #007bff;
            color: #0c5460;
        }
        
        .tool-complete {
            background: #d4edda;
            border-left: 4px solid #28a745;
            color: #155724;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏛️ Lextech AI Judge Assistant - Fixed</h1>
        <p>Enhanced streaming with tool call notifications and loading indicators (Fixed duplicate tool notifications)</p>
        
        <div style="background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <h3>🔧 Fix Applied:</h3>
            <ul style="margin: 10px 0;">
                <li><strong>✅ No More Duplicates:</strong> Tool notifications now properly track active tools</li>
                <li><strong>🎯 Smart Detection:</strong> Distinguishes between processing and searching the same PDF</li>
                <li><strong>⏳ Clean UI:</strong> Only shows relevant tool activity without spam</li>
            </ul>
        </div>
        
        <div class="input-group">
            <label for="message"><strong>Legal Question:</strong></label>
            <textarea id="message" rows="4" placeholder="Type your question...">Summarize key issues in the uploaded documents.</textarea>
        </div>

        <div class="input-group">
            <label><strong>Upload (URLs) — up to 5:</strong></label>
            <div id="url-list"></div>
            <div style="display:flex; gap:8px; margin-top:8px;">
                <input type="text" id="url-input" placeholder="https://..." style="flex:1;" />
                <button onclick="addUrl()">➕ Add URL</button>
                <button onclick="startPreprocess()">⬆️ Preprocess</button>
            </div>
            <small>URLs are sent to the server for preprocessing; you'll see per-URL status below.</small>
            <div id="preprocess-status" style="margin-top:10px;"></div>
        </div>
        
        <div class="input-group">
            <button onclick="sendMessage(false)">📤 Send Message (No Streaming)</button>
            <button class="btn-streaming" onclick="sendMessage(true)">🚀 Send Message (Fixed Enhanced Streaming)</button>
            <button onclick="clearResponse()">🗑️ Clear Response</button>
            <button onclick="resetConversation()">🔄 Reset Chat</button>
        </div>
        
        <div class="status" id="status">Ready to send message...</div>
        
        <div class="response" id="response">
            <p><em>Response will appear here...</em></p>
        </div>
    </div>

    <script>
        let isStreaming = false;
        let sessionId = null;
        let uploadItems = []; // { upload_id, url, type, status }
        function addUrl() {
            const input = document.getElementById('url-input');
            const url = input.value.trim();
            if (!url) return;
            const list = document.getElementById('url-list');
            const current = Array.from(list.querySelectorAll('[data-url]')).map(n => n.getAttribute('data-url'));
            if (current.length >= 5) { alert('Maximum 5 URLs'); return; }
            if (current.includes(url)) { alert('Already added'); return; }
            const row = document.createElement('div');
            row.setAttribute('data-url', url);
            row.setAttribute('data-key', encodeURI(url));
            row.style.display = 'flex';
            row.style.alignItems = 'center';
            row.style.gap = '8px';
            row.style.marginTop = '6px';
            row.innerHTML = `<code style="flex:1; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(url)}</code><button onclick="this.parentElement.remove()">✖</button>`;
            list.appendChild(row);
            input.value = '';
        }

        async function startPreprocess() {
            const list = document.getElementById('url-list');
            const urls = Array.from(list.querySelectorAll('[data-url]')).map(n => n.getAttribute('data-url'));
            const statusDiv = document.getElementById('preprocess-status');
            if (urls.length === 0) {
                alert('Add 1-5 URLs first');
                return;
            }
            statusDiv.innerHTML = '';
            urls.forEach(u => {
                const item = document.createElement('div');
                item.setAttribute('data-status-key', encodeURI(u));
                item.className = 'tool-notification tool-working';
                item.innerHTML = `<span class="loading-spinner"></span>Preprocessing: ${escapeHtml(u)}`;
                statusDiv.appendChild(item);
            });
            try {
                const res = await fetch('/files/preprocess-urls', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId, files: urls.map(u => ({ url: u })) })
                });
                if (!res.ok) throw new Error('Preprocess failed');
                const data = await res.json();
                sessionId = data.session_id;
                uploadItems = data.items || [];
                // Update statuses
                uploadItems.forEach(it => {
                    const node = statusDiv.querySelector(`[data-status-key="${CSS.escape(encodeURI(it.url))}"]`);
                    if (!node) return;
                    node.className = it.status.startsWith('failed') ? 'tool-notification error' : 'tool-notification tool-complete';
                    node.innerHTML = (it.status.startsWith('failed') ? '❌ ' : '✅ ') + `${escapeHtml(it.type)}: ${escapeHtml(it.url)} — ${escapeHtml(it.status)}`;
                });
            } catch (e) {
                alert('Preprocessing error: ' + e.message);
            }
        }
        
        async function sendMessage(useStreaming = false) {
            const message = document.getElementById('message').value;
            const responseDiv = document.getElementById('response');
            const statusDiv = document.getElementById('status');
            const buttons = document.querySelectorAll('button');
            
            if (!message.trim()) {
                alert('Please enter a message');
                return;
            }
            
            if (isStreaming) {
                alert('Already processing a request. Please wait...');
                return;
            }
            
            isStreaming = true;
            buttons.forEach(btn => btn.disabled = true);
            
            if (useStreaming) {
                await handleEnhancedStreamingResponse(message, responseDiv, statusDiv);
            } else {
                await handleNormalResponse(message, responseDiv, statusDiv);
            }
            
            isStreaming = false;
            buttons.forEach(btn => btn.disabled = false);
        }
        
        async function handleNormalResponse(message, responseDiv, statusDiv) {
            statusDiv.className = 'status streaming';
            statusDiv.textContent = '⏳ Processing request (No Streaming)...';
            responseDiv.innerHTML = '<p><em>🔄 Processing your request...</em></p>';
            
            try {
                const response = await fetch('/judge', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        messages: [{
                            role: 'user',
                            content: message
                        }],
                        session_id: sessionId,
                        upload_ids: uploadItems.filter(i => i.status === 'processed').map(i => i.upload_id)
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                
                statusDiv.className = 'status complete';
                statusDiv.textContent = '✅ Response completed successfully!' + (data.cached ? ' (From Cache)' : '');
                responseDiv.innerHTML = '<div style="white-space: pre-wrap;"><strong>Complete Response:</strong>' + String.fromCharCode(10) + String.fromCharCode(10) + escapeHtml(data.response) + '</div>';
                
            } catch (error) {
                statusDiv.className = 'status error';
                statusDiv.textContent = '❌ Error occurred: ' + error.message;
                responseDiv.innerHTML = '<p style="color: red;">Error: ' + escapeHtml(error.message) + '</p>';
            }
        }
        
        async function handleEnhancedStreamingResponse(message, responseDiv, statusDiv) {
            statusDiv.className = 'status streaming';
            statusDiv.textContent = '🚀 Enhanced streaming with tool notifications...';
            responseDiv.innerHTML = '<p><em>🔄 Initializing response...</em></p>';
            
            // Tool state tracking to prevent duplicates
            let activeTool = null;
            let toolMessageCount = 0;
            const seenToolNotifications = new Set();
            
            try {
                const response = await fetch('/judge/stream', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        messages: [{
                            role: 'user',
                            content: message
                        }],
                        session_id: sessionId,
                        upload_ids: uploadItems.filter(i => i.status === 'processed').map(i => i.upload_id)
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullResponse = '';
                
                while (true) {
                    const { done, value } = await reader.read();
                    
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split(String.fromCharCode(10));
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                
                                if (data.type === 'tool_notification') {
                                    const content = data.content;
                                    
                                    // Create a unique key for this tool notification
                                    const notificationKey = content.replace(/[⏳✅📄🔍🔎]/g, '').trim();
                                    
                                    // Only show if it's a new type of tool notification
                                    if (!seenToolNotifications.has(notificationKey) || 
                                        (content.includes('✅') && activeTool)) {
                                        
                                        seenToolNotifications.add(notificationKey);
                                        
                                        const toolDiv = document.createElement('div');
                                        
                                        if (content.includes('📄 Let me process') || 
                                            content.includes('🔍 Let me search') || 
                                            content.includes('🔎 Let me search')) {
                                            // Starting a new tool
                                            activeTool = content.includes('📄') ? 'pdf' : 'search';
                                            toolMessageCount++;
                                            
                                            // Only show the first instance or if it's a different tool type
                                            if (toolMessageCount === 1 || 
                                                (content.includes('🔍') && activeTool !== 'search') ||
                                                (content.includes('🔎') && activeTool !== 'search')) {
                                                toolDiv.className = 'tool-notification';
                                                toolDiv.textContent = content;
                                                responseDiv.appendChild(toolDiv);
                                            }
                                        } else if (content.includes('⏳')) {
                                            // Tool working - only show if no duplicate and store reference for spinner removal
                                            toolDiv.className = 'tool-notification tool-working';
                                            toolDiv.innerHTML = '<span class="loading-spinner"></span>' + escapeHtml(content);
                                            toolDiv.setAttribute('data-spinner-active', 'true');
                                            responseDiv.appendChild(toolDiv);
                                        } else if (content.includes('✅') && activeTool) {
                                            // Tool completed - stop any active spinners and show completion
                                            
                                            // Find and remove/update any active spinner notifications for this tool
                                            const activeSpinners = responseDiv.querySelectorAll('[data-spinner-active="true"]');
                                            activeSpinners.forEach(spinnerDiv => {
                                                // Remove the spinner but keep the text
                                                const spinnerElement = spinnerDiv.querySelector('.loading-spinner');
                                                if (spinnerElement) {
                                                    spinnerElement.remove();
                                                }
                                                spinnerDiv.removeAttribute('data-spinner-active');
                                                // Change the styling to show it's completed
                                                spinnerDiv.className = 'tool-notification tool-complete';
                                            });
                                            
                                            // Add the completion message
                                            toolDiv.className = 'tool-notification tool-complete';
                                            toolDiv.textContent = content;
                                            responseDiv.appendChild(toolDiv);
                                            activeTool = null; // Reset active tool
                                        }
                                        
                                        responseDiv.scrollTop = responseDiv.scrollHeight;
                                    }
                                    
                                } else if (data.type === 'chunk' && data.content) {
                                    fullResponse += data.content;
                                    
                                    // Update or create response container
                                    let existingContainer = responseDiv.querySelector('.response-content');
                                    if (!existingContainer) {
                                        existingContainer = document.createElement('div');
                                        existingContainer.className = 'response-content';
                                        existingContainer.style.whiteSpace = 'pre-wrap';
                                        existingContainer.style.marginTop = '10px';
                                        responseDiv.appendChild(existingContainer);
                                    }
                                    
                                    existingContainer.textContent = fullResponse;
                                    responseDiv.scrollTop = responseDiv.scrollHeight;
                                    
                                } else if (data.type === 'end') {
                                    statusDiv.className = 'status complete';
                                    statusDiv.textContent = '✅ Enhanced streaming completed successfully!';
                                    
                                    // Add final formatting
                                    const finalDiv = document.createElement('div');
                                    finalDiv.innerHTML = '<strong style="color: #28a745;">Complete Response:</strong>';
                                    finalDiv.style.marginTop = '15px';
                                    finalDiv.style.paddingTop = '15px';
                                    finalDiv.style.borderTop = '2px solid #28a745';
                                    
                                    responseDiv.insertBefore(finalDiv, responseDiv.firstChild);
                                    
                                } else if (data.type === 'error') {
                                    statusDiv.className = 'status error';
                                    statusDiv.textContent = '❌ Error occurred: ' + data.error;
                                    const errorDiv = document.createElement('p');
                                    errorDiv.style.color = 'red';
                                    errorDiv.textContent = 'Error: ' + data.error;
                                    responseDiv.appendChild(errorDiv);
                                }
                            } catch (e) {
                                console.log('Failed to parse chunk:', line);
                            }
                        }
                    }
                }
                
            } catch (error) {
                statusDiv.className = 'status error';
                statusDiv.textContent = '❌ Connection error: ' + error.message;
                responseDiv.innerHTML = '<p style="color: red;">Connection Error: ' + escapeHtml(error.message) + '</p>';
            }
        }
        
        function clearResponse() {
            document.getElementById('response').innerHTML = '<p><em>Response will appear here...</em></p>';
            document.getElementById('status').className = 'status';
            document.getElementById('status').textContent = 'Ready to send message...';
        }
        
        async function resetConversation() {
            try {
                const response = await fetch('/reset-conversation', { method: 'POST' });
                if (response.ok) {
                    document.getElementById('status').className = 'status complete';
                    document.getElementById('status').textContent = '✅ Conversation reset successfully!';
                    clearResponse();
                }
            } catch (error) {
                console.error('Failed to reset conversation:', error);
            }
        }
        
        function escapeHtml(unsafe) {
            return unsafe
                 .replace(/&/g, "&amp;")
                 .replace(/</g, "&lt;")
                 .replace(/>/g, "&gt;")
                 .replace(/"/g, "&quot;")
                 .replace(/'/g, "&#039;");
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)