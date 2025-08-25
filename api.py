from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from api.JudgeAgent2 import langgraph_agent_executor  # Import the agent

app = FastAPI()

# Define the request body schema
class QueryRequest(BaseModel):
    messages: List[dict]  # Chat history with [{"role": "human/ai", "content": "text"}]

# Define the response model
class QueryResponse(BaseModel):
    response: str

@app.post("/judge", response_model=QueryResponse)
async def query_judge_agent(request: QueryRequest):
    """
    Endpoint to interact with the Lextech AI Judge Assistant.
    Accepts chat history and returns the AI's response.
    """
    try:
        # Format messages for the agent
        chat_messages = [(msg["role"], msg["content"]) for msg in request.messages]

        # Prepare config for agent execution
        config = {"configurable": {"thread_id": "main-conversation"}}

        # Collect agent response
        full_response = ""
        for chunk in langgraph_agent_executor.stream(
            {"messages": chat_messages},
            config,
        ):
            if 'agent' in chunk and chunk['agent'].get('messages'):
                for msg in chunk['agent']['messages']:
                    if hasattr(msg, 'content'):
                        content = msg.content
                        if content:
                            full_response += content

        return {"response": full_response}

    except Exception as e:
        return {"response": f"Error occurred: {str(e)}"}

# Run the server: uvicorn api:app --reload
