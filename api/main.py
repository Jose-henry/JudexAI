from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
import base64
import io
import os
from PIL import Image
import fitz  # PyMuPDF for PDF processing
from api.JudgeAgent2 import langgraph_agent_executor  # Import the LangChain agent

# Initialize FastAPI app
app = FastAPI()

# Define the request body schema
class QueryRequest(BaseModel):
    messages: List[dict]  # [{"role": "human/ai", "content": "query text"}]

# Define the response model
class QueryResponse(BaseModel):
    response: str

# Directory to store PDF files
NOTES_DIR = "notes"
os.makedirs(NOTES_DIR, exist_ok=True)

# Helper functions for processing files
def pdf_to_images(pdf_bytes: bytes) -> List[str]:
    """
    Convert a PDF to a list of base64-encoded image strings.

    Args:
        pdf_bytes (bytes): Byte content of the PDF file.

    Returns:
        List[str]: A list of base64-encoded image strings, one for each page.
    """
    images_base64 = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        images_base64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

    return images_base64

def image_to_base64(image_bytes: bytes) -> str:
    """
    Convert an image to a base64-encoded string.

    Args:
        image_bytes (bytes): Byte content of the image file.

    Returns:
        str: A base64-encoded image string.
    """
    img = Image.open(io.BytesIO(image_bytes))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Chat history to maintain conversation context
chat_history = []

@app.post("/judge", response_model=QueryResponse)
async def query_judge_agent(
    request: Request,
    file: UploadFile = None
):
    """
    Endpoint to interact with the Lextech AI Judge Assistant, allowing for text and file uploads.

    Args:
        request (Request): Raw request object to handle multipart or JSON input.
        file (UploadFile, optional): An optional file upload (PDF or image).

    Returns:
        QueryResponse: AI's response to the uploaded content and text input.
    """
    try:
        content_type = request.headers.get('content-type', '')
        
        if 'multipart/form-data' in content_type:
            form = await request.form()
            request_str = form.get('request')
            if not request_str:
                raise HTTPException(status_code=400, detail="Form field 'request' is required")
            request_data = QueryRequest.model_validate_json(request_str)
            file = form.get('file')
        else:
            body = await request.json()
            request_data = QueryRequest.model_validate(body)

        # Format messages for multimodal input
        formatted_messages = []
        for msg in request_data.messages:
            formatted_msg = {
                "type": "text",
                "text": msg["content"]
            }
            formatted_messages.append(formatted_msg)

        # Process file if present
        if file:
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension == "pdf":
                pdf_bytes = await file.read()
                images_base64 = pdf_to_images(pdf_bytes)
                # Add each page as a separate image message
                for image_base64 in images_base64:
                    formatted_messages.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    })
            elif file_extension in ["png", "jpg", "jpeg"]:
                image_bytes = await file.read()
                image_base64 = image_to_base64(image_bytes)
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

        # Update chat history
        question = request_data.messages[-1]["content"] if request_data.messages else ""
        if file:
            question += f" (with uploaded document: {file.filename})"
        chat_history.extend([
            ("user", question),
            ("ai", full_response)
        ])

        return QueryResponse(response=full_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")