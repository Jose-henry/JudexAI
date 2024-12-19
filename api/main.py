from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from JudgeAgent2 import langgraph_agent_executor  # Adjusted import path
import os
from fastapi.responses import FileResponse

# Initialize FastAPI app
app = FastAPI()

# Define the request body schema
class QueryRequest(BaseModel):
    messages: List[dict]  # [{"role": "human/ai", "content": "query text"}]

# Define the response model
class QueryResponse(BaseModel):
    response: str
    download_link: str = None

# Directory to store PDF filess
NOTES_DIR = "notes"
os.makedirs(NOTES_DIR, exist_ok=True)

@app.post("/judge", response_model=QueryResponse)
async def query_judge_agent(request: QueryRequest):
    """
    Endpoint to interact with the Lextech AI Judge Assistant.
    Accepts a list of messages and returns the AI's response.
    Optionally generates a PDF and provides a download link.
    """
    try:
        # Reformat messages for the agent
        chat_messages = [(msg["role"], msg["content"]) for msg in request.messages]
        config = {"configurable": {"thread_id": "main-conversation"}}

        # Collect agent response
        full_response = ""
        for chunk in langgraph_agent_executor.stream({"messages": chat_messages}, config):
            if 'agent' in chunk and chunk['agent'].get('messages'):
                for msg in chunk['agent']['messages']:
                    if hasattr(msg, 'content'):
                        full_response += msg.content

        # Generate a PDF file
        pdf_filename = os.path.join(NOTES_DIR, "response_note.pdf")
        with open(pdf_filename, "w") as f:
            f.write(full_response)

        # Generate a download link
        download_link = f"/download/response_note.pdf"

        return {"response": full_response, "download_link": download_link}
    except Exception as e:
        return {"response": f"Error occurred: {str(e)}", "download_link": None}

@app.get("/download/{file_name}")
async def download_file(file_name: str):
    """
    Endpoint to serve PDF files for download.
    """
    file_path = os.path.join(NOTES_DIR, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf", filename=file_name)
    return {"error": "File not found."}
