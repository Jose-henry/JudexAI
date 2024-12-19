# Single Agent built with LANGGRAPH completely


import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
# Load environment variables from a .env file
load_dotenv()

# Tavily API key
#TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_API_KEY = "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"


# Create search tool
search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)