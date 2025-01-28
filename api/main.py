from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
from pydantic import BaseModel, HttpUrl
import base64
import io
import os
from PIL import Image
import fitz  # PyMuPDF for PDF processing
import requests
import hashlib
import json
from api.JudgeAgent2 import langgraph_agent_executor

# Initialize FastAPI app
app = FastAPI()

# Enhanced request body schema to handle URLs
class MessageContent(BaseModel):
    role: str
    content: str
    file_url: Optional[List[HttpUrl]] = None

class QueryRequest(BaseModel):
    messages: List[MessageContent]

class QueryResponse(BaseModel):
    response: str
    cached: bool = False

# In-memory cache for responses
response_cache: Dict[str, str] = {}



def generate_cache_key(messages: List[MessageContent]) -> str:
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
        'file_url': [str(url) for url in (msg.file_url or [])]
    } for msg in messages], sort_keys=True)
    
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
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")

async def process_pdf_url(url: str) -> List[str]:
    """
    Process a PDF from URL and convert to base64 images.
    
    Args:
        url (str): URL of the PDF file
        
    Returns:
        List[str]: List of base64-encoded images, one per page
    """
    pdf_bytes = await download_file(url)
    images_base64 = []
    
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            images_base64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
            
        return images_base64
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

async def process_image_url(url: str) -> str:
    """
    Process an image from URL and convert to base64.
    
    Args:
        url (str): URL of the image file
        
    Returns:
        str: Base64-encoded image string
    """
    image_bytes = await download_file(url)
    try:
        img = Image.open(io.BytesIO(image_bytes))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

# Chat history to maintain conversation context
chat_history = []

@app.post("/judge", response_model=QueryResponse)
async def query_judge_agent(request: QueryRequest):
    """
    Endpoint to interact with the Lextech AI Judge Assistant using file URLs.
    
    Args:
        request (QueryRequest): Request containing messages and optional file URLs
        
    Returns:
        QueryResponse: AI's response to the content
    """
    try:
        # Generate cache key for the request
        cache_key = generate_cache_key(request.messages)

        # Check if response is in cache
        if cache_key in response_cache:
            return QueryResponse(
                response=response_cache[cache_key],
                cached=True
            )


        # Format messages for multimodal input
        formatted_messages = []
        
        for msg in request.messages:
            # Add text content
            formatted_messages.append({
                "type": "text",
                "text": msg.content
            })
            
            # Process file URLs if present
            if msg.file_url:
                for url in msg.file_url:
                    file_ext = str(url).split('.')[-1].lower()
                    
                    if file_ext == 'pdf':
                        # Process PDF and add each page as an image
                        images_base64 = await process_pdf_url(str(url))
                        for image_base64 in images_base64:
                            formatted_messages.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            })
                    elif file_ext in ['png', 'jpg', 'jpeg']:
                        # Process single image
                        image_base64 = await process_image_url(str(url))
                        formatted_messages.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        })
                    else:
                        raise HTTPException(status_code=400, detail="Unsupported file format")

        # Format final message structure for LangChain
        chat_messages = [("human", formatted_messages)]
        config = {"configurable": {"thread_id": "main-conversation"}}

        # Collect agent response
        full_response = ""
        for chunk in langgraph_agent_executor.stream({"messages": chat_messages}, config):
            if 'agent' in chunk and chunk['agent'].get('messages'):
                for msg in chunk['agent']['messages']:
                    if hasattr(msg, 'content'):
                        full_response += msg.content

        # Cache the response
        response_cache[cache_key] = full_response

        # Update chat history with the last message and response
        last_msg = request.messages[-1] if request.messages else None
        if last_msg:
            file_urls = list(last_msg.file_url) if last_msg.file_url else []
            chat_history.extend([
                ("user", f"{last_msg.content} (with files: {', '.join(map(str, file_urls))})" if file_urls else last_msg.content),
                ("ai", full_response)
            ])

        return QueryResponse(response=full_response, cached=False)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    


# Optional: Add cache management endpoints
@app.post("/clear-cache")
async def clear_cache():
    """Clear the in-memory response cache"""
    response_cache.clear()
    return {"message": "Cache cleared successfully"}

@app.get("/cache-status")
async def get_cache_status():
    """Get current cache status"""
    return {
        "cache_size": len(response_cache),
        "cached_keys": list(response_cache.keys())
    }