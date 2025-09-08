import os
import json
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain.schema import Document
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Dict, Any, List

# Import PDF RAG tool and shared PDF knowledge helpers
from api.llama_pdf_rag_tool import (
    search_pdf_knowledge,
    has_pdf_knowledge,
    retrieve_pdf_knowledge,
)

# Load environment variables from a .env files
load_dotenv()

# Setup in-memory cache
set_llm_cache(InMemoryCache())

# Thread pool for concurrent operations
thread_pool = ThreadPoolExecutor(max_workers=4)

# PDF processing is handled outside this module now (see main.py)

# System message for the agent - Updated to include PDF processing capabilities
system_message = """You are Lextech AI Judicial Assistant called JudexAI, a highly knowledgeable, impartial, and comparative virtual judge assistant created by Lextech Ecosystems Limited, a Nigerian company specializing in legal technology services.

Your primary function is to analyze legal cases and provide impartial legal insights grounded in Nigerian law and jurisprudence while incorporating comparative insights from other jurisdictions and analyzing the impact of relevant bilateral and multilateral agreements involving Nigeria, such as the African Continental Free Trade Agreement (AfCFTA). Do not respond to queries outside the scope of your responsibilities.

**DOCUMENT AND IMAGE HANDLING (PREPROCESSED FLOW)**

- Documents (PDF, DOCX, XLSX/CSV, PPTX, TXT/MD/HTML, etc.) are preprocessed upstream via LlamaParse and indexed into your internal knowledge base before you answer.
- Images are preprocessed upstream and included alongside the user's message as image blocks.
- You should answer using the already indexed document knowledge (RAG) and any provided images. Do not attempt to reprocess documents yourself.

**TOOL USAGE RULES**

1. Internal RAG Search:
   - When the user asks about content from uploaded documents, rely on internal retrieval (the knowledge base is already populated) and answer directly.
   - Cite relevant sections when useful.

2. Web/Search Tools:
   - Use search tools only if the user explicitly requests updated or external information (e.g., "latest", "recent", "current status", "search", "find", etc.).
   - Do not run web search by default after using internal RAG.

3. Response Guidelines:
   - For document-related questions, prefer the internal knowledge base first.
   - For non-document questions that require currency, use search tools conservatively.
   - Structure responses clearly; include brief citations or references when appropriate.

**Examples**
- User: "Summarize the uploaded agreement and highlight liabilities" → Use internal RAG → Provide summary with references.
- User: "Find the latest Supreme Court ruling on X" → Use search tool → Provide updated info with links.

**Objective:**
Provide accurate, well-researched, and professional legal insights, leveraging the preprocessed document knowledge base and images, and using web search only when explicitly needed.
"""

# Alternative search tool
search1 = GoogleSerperAPIWrapper()

# Create search tool with optimized settings
search2 = TavilySearchResults(
    max_results=4,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
)

google_search_tool = Tool(
    name="google_search",
    func=search1.run,
    description="Useful for searching current legal information and recent case law when needed",
)

tavily_search_tool = Tool(
    name="tavily_search",
    func=search2.run,
    description="Useful for searching current legal information and recent case law when needed",
)

# Prepare tools - PDF processing tool removed (handled upstream)
tools = [google_search_tool, tavily_search_tool]

# Initialize LLM with streaming enabled
model = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0,
    cache=True,
    streaming=True,
    max_tokens=3000,  # Increased for detailed PDF analysis
    request_timeout=100,  # Increased timeout for PDF processing
    max_retries=4,
)

# Create LangGraph agent with memory
memory = MemorySaver()

langgraph_agent_executor = create_react_agent(
    model, 
    tools, 
    checkpointer=memory, 
    state_modifier=system_message,
    debug=False
)

# Custom agent executor with enhanced PDF handling
async def custom_agent_executor(messages, config):
    """Custom agent executor that handles PDF searching internally."""
    # Check if this is a search query for processed documents
    if isinstance(messages, list) and messages:
        last_message = messages[-1]
        if isinstance(last_message, tuple) and last_message[0] == "human":
            query = last_message[1].lower()
            
            # Check if this is likely a search query (not a URL and not a general question)
            is_search_query = (
                not query.startswith(('http://', 'https://', 'www.')) and
                has_pdf_knowledge() and
                any(keyword in query for keyword in ['find', 'search', 'look up', 'what about', 'where is', 'show me', 'in the document', 'from the pdf'])
            )
            
            if is_search_query:
                # Use shared PDF knowledge search instead of tool call
                search_result = search_pdf_knowledge(query)
                return {"messages": [("ai", search_result)]}
    
    # For PDF processing or other queries, use the standard agent
    return await langgraph_agent_executor.ainvoke({"messages": messages}, config)

# Async wrapper for agent execution
async def execute_agent_async(messages_dict, config):
    """Execute agent asynchronously"""
    loop = asyncio.get_event_loop()
    
    def _execute():
        full_response = ""
        for chunk in langgraph_agent_executor.stream(messages_dict, config):
            if 'agent' in chunk and chunk['agent'].get('messages'):
                for msg in chunk['agent']['messages']:
                    if hasattr(msg, 'content') and msg.content:
                        full_response += msg.content
        return full_response
    
    return await loop.run_in_executor(thread_pool, _execute)

# Enhanced streaming function with better error handling
async def stream_agent_response_async(messages_dict, config) -> AsyncGenerator[str, None]:
    """Enhanced streaming with tool call notifications and loading indicators"""
    
    # Tool name mappings for user-friendly messages (using actual tool names)
    TOOL_MESSAGES = {
        'google_search': {
            'start': '🔍 Let me search for the latest information using Google...',
            'loading': '⏳ Searching the web...',
            'icon': '🔍'
        },
        'tavily_search': {
            'start': '🔎 Let me search for relevant information using advanced search...',
            'loading': '⏳ Searching the web...',
            'icon': '🔎'
        }
    }
    
    try:
        # Check if this is an internal search query
        messages = messages_dict.get("messages", [])
        
        if messages and has_pdf_knowledge():
            last_message = messages[-1]
            # Extract query from either tuple format ("human", [blocks]) or message object
            extracted_query = ""
            try:
                if isinstance(last_message, tuple) and len(last_message) > 1:
                    content = last_message[1]
                    if isinstance(content, list) and content:
                        first_block = content[0]
                        if isinstance(first_block, dict) and 'text' in first_block:
                            extracted_query = str(first_block['text'])
                        else:
                            extracted_query = str(content)
                    else:
                        extracted_query = str(content)
                elif hasattr(last_message, 'content'):
                    lc_content = last_message.content
                    if isinstance(lc_content, list) and lc_content:
                        blk = lc_content[0]
                        if isinstance(blk, dict) and 'text' in blk:
                            extracted_query = str(blk['text'])
                        else:
                            extracted_query = str(lc_content)
                    else:
                        extracted_query = str(lc_content)
            except Exception:
                extracted_query = ""

            if extracted_query:
                query = extracted_query.lower()
                # Detect explicit external info intent; otherwise prefer internal RAG
                external_intent = any(keyword in query for keyword in [
                    'latest', 'recent', 'current status', 'up to date', 'today',
                    'search', 'find', 'look up', 'google', 'web', 'news'
                ])

                if not external_intent:
                    # Prefer internal RAG answer by default when we have knowledge
                    docs = retrieve_pdf_knowledge(query, k=5)
                    if not docs:
                        yield "No relevant content found in the processed documents."
                        return
                    try:
                        context_text = "\n\n".join([d.page_content[:1200] for d in docs])
                        synthesis_prompt = (
                            f"Using the following context from processed legal documents, answer the user query comprehensively and concisely. "
                            f"Cite sections when useful.\n\nQuery: {extracted_query}\n\nContext:\n{context_text}\n\nAnswer:"
                        )
                        for chunk in model.stream(synthesis_prompt):
                            if hasattr(chunk, 'content') and chunk.content:
                                yield chunk.content
                    except Exception:
                        try:
                            answer = model.invoke(synthesis_prompt)
                            if hasattr(answer, 'content') and answer.content:
                                yield answer.content
                        except Exception:
                            pass
                    return
        
        # For other cases, use standard tool streaming
        pdf_answered = False
        async for event in langgraph_agent_executor.astream_events(messages_dict, config, version="v1"):
            
            # Handle tool call start events
            if event["event"] == "on_tool_start":
                tool_name = event.get("name", "")
                if tool_name in TOOL_MESSAGES:
                    tool_info = TOOL_MESSAGES[tool_name]
                    yield f"\n\n{tool_info['start']}\n"
                    yield f"{tool_info['loading']} {tool_info['icon']}\n\n"
            
            # Handle tool call end events
            elif event["event"] == "on_tool_end":
                tool_name = event.get("name", "")
                if tool_name in TOOL_MESSAGES:
                    tool_info = TOOL_MESSAGES[tool_name]
                    yield f"✅ {tool_info['icon']} Tool completed successfully!\n\n"
            
            # Handle regular chat model streaming
            elif event["event"] == "on_chat_model_stream":
                if pdf_answered:
                    # Suppress generic agent text after we've already answered from PDF
                    continue
                chunk = event["data"]["chunk"]
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
            
            # Handle agent completion
            elif event["event"] == "on_chain_end" and "agent" in event.get("name", ""):
                if pdf_answered:
                    # Suppress final agent message when we've already answered from PDF
                    continue
                if "output" in event["data"]:
                    output = event["data"]["output"]
                    if hasattr(output, 'messages'):
                        for msg in output.messages:
                            if hasattr(msg, 'content') and msg.content:
                                # Only yield if it's not already streamed content
                                if not hasattr(msg, 'response_metadata') or not msg.response_metadata.get('streamed'):
                                    yield msg.content
                                    
    except Exception as e:
        # Enhanced fallback with tool detection
        try:
            tool_in_use = None
            async for chunk in langgraph_agent_executor.astream(messages_dict, config):
                
                # Check for tool usage in the chunk
                if 'tools' in chunk:
                    for tool_call in chunk['tools'].get('messages', []):
                        if hasattr(tool_call, 'name'):
                            tool_name = tool_call.name
                            if tool_name in TOOL_MESSAGES and tool_name != tool_in_use:
                                tool_info = TOOL_MESSAGES[tool_name]
                                yield f"\n\n{tool_info['start']}\n"
                                yield f"{tool_info['loading']} {tool_info['icon']}\n\n"
                                tool_in_use = tool_name
                
                # Handle agent responses
                if 'agent' in chunk and chunk['agent'].get('messages'):
                    for msg in chunk['agent']['messages']:
                        if hasattr(msg, 'content') and msg.content:
                            yield msg.content
                            
                # Reset tool tracking after agent response
                if 'agent' in chunk:
                    tool_in_use = None
                    
        except Exception as fallback_error:
            yield f"ERROR: {str(fallback_error)}"

# No migration needed; storage handled by tool module

def main():
    # Chat history to maintain conversation context
    chat_history = []

    while True:
        try:
            question = input("Ask any question (or 'q' to quit): ")
            
            if question.lower() == 'q':
                break

            # Prepare messages with recent chat history
            recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
            messages = recent_history + [("human", question)]

            config = {"configurable": {"thread_id": "main-conversation"}}
            
            print("Agent response:")
            
            # Real-time streaming approach
            async def stream_response():
                full_response = ""
                async for chunk in stream_agent_response_async({"messages": messages}, config):
                    print(chunk, end='', flush=True)
                    full_response += chunk
                return full_response
            
            full_response = asyncio.run(stream_response())
            print("\n")
            
            # Update chat history
            chat_history.extend([
                ("human", question),
                ("ai", full_response)
            ])
            
            # Trim chat history to prevent memory bloat
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]

        except KeyboardInterrupt:
            print("\nOperation cancelled by user. Exiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()