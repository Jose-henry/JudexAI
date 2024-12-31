from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from api.JudgeAgent2 import langgraph_agent_executor  # Adjusted import path
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

# Directory to store PDF files
NOTES_DIR = "notes"
os.makedirs(NOTES_DIR, exist_ok=True)

# Chat history to maintain conversation context
chat_history = []

@app.post("/judge", response_model=QueryResponse)
async def query_judge_agent(request: QueryRequest):
    """
    Endpoint to interact with the Lextech AI Judge Assistant.
    Accepts a list of messages and returns the AI's response.
    """
    try:
        # Reformat messages for the agent
        chat_messages = chat_history + [(msg["role"], msg["content"]) for msg in request.messages]
        config = {"configurable": {"thread_id": "main-conversation"}}

        # Collect agent response
        full_response = ""
        for chunk in langgraph_agent_executor.stream({"messages": chat_messages}, config):
            if 'agent' in chunk and chunk['agent'].get('messages'):
                for msg in chunk['agent']['messages']:
                    if hasattr(msg, 'content'):
                        full_response += msg.content

        # Extract the user's question from the request messages
        question = request.messages[-1]["content"]  # Assuming the last message is the user's question
        chat_history.extend([
            ("user", question),
            ("ai", full_response)
        ])

        return {"response": full_response}
    except Exception as e:
        return {"response": f"Error occurred: {str(e)}"}

# @app.get("/current-conversation")
# async def get_current_conversation():
#     """
#     Endpoint to retrieve the current conversation context (user and AI responses).
#     """
#     formatted_history = [
#         {"role": role, "content": content} for role, content in chat_history
#     ]
#     return {"conversation": formatted_history}

@app.get("/")
async def get_home():
    """
    Endpoint to display some text on the browser.
    """
    return {"Welcome to the Lextech AI Judge API."}
