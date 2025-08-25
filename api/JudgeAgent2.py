import os
import json
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.schema import Document
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Dict, Any, List

# Import the modified PDF RAG tool
from api.llama_pdf_rag_tool import pdf_rag_tool, get_processed_documents, get_document_cache

# Load environment variables from a .env files
load_dotenv()

# Setup in-memory cache
set_llm_cache(InMemoryCache())

# Thread pool for concurrent operations
thread_pool = ThreadPoolExecutor(max_workers=4)

# Vector store and embeddings for JudgeAgent
judge_embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)

judge_vectorstore = InMemoryVectorStore(embedding=judge_embeddings)

# Document cache for JudgeAgent
judge_document_cache: Dict[str, Dict] = {}

# Migrate existing processed documents from tool
def migrate_processed_documents():
    """Migrate processed documents from tool to JudgeAgent vector store."""
    try:
        # Get documents from tool
        documents = get_processed_documents()
        if documents:
            judge_vectorstore.add_documents(documents)
            # Migrate cache info
            tool_cache = get_document_cache()
            judge_document_cache.update(tool_cache)
            print(f"Migrated {len(documents)} documents to JudgeAgent vector store")
    except Exception as e:
        print(f"Migration failed: {str(e)}")

# Search function for JudgeAgent
def judge_search_documents(query: str, k: int = 5) -> str:
    """Search through processed documents in JudgeAgent's vector store."""
    try:
        if len(judge_vectorstore.store) == 0:
            return "No documents have been processed yet. Please provide a PDF URL first."
            
        results = judge_vectorstore.similarity_search_with_score(query, k=k)
        
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

# Enhanced PDF processing that stores in JudgeAgent's vector store
def enhanced_pdf_processing(url: str) -> str:
    """Process PDF and store in JudgeAgent's vector store."""
    try:
        # Use the tool to process PDF
        result = pdf_rag_tool.func(url)
        
        # Check if processing was successful
        if "Successfully processed" in result or "already processed" in result:
            # Migrate the newly processed documents
            migrate_processed_documents()
            return result
        else:
            return result
            
    except Exception as e:
        return f"PDF processing failed: {str(e)}"

# System message for the agent - Updated to include PDF processing capabilities
system_message = """You are Lextech AI Judicial Assistant called JudexAI, a highly knowledgeable, impartial, and comparative virtual judge assistant created by Lextech Ecosystems Limited, a Nigerian company specializing in legal technology services.

Your primary function is to analyze legal cases and provide impartial legal insights grounded in Nigerian law and jurisprudence while incorporating comparative insights from other jurisdictions and analyzing the impact of relevant bilateral and multilateral agreements involving Nigeria, such as the African Continental Free Trade Agreement (AfCFTA), do not respond to queries outside the scope of your responsibilities.

**ENHANCED PDF PROCESSING CAPABILITIES**

You have an advanced PDF processing system:

- When a user provides a PDF URL, use the `llama_pdf_document_processor` tool to process it. The document is then stored in your internal knowledge base for future reference.
- For subsequent questions about processed PDFs, use your internal search functions to answer, without reprocessing or calling external tools.
- Processed documents remain available for future queries, enabling efficient and fast responses.

**TOOL USAGE RULES:**

1. **PDF Processing:**
   - Use the `llama_pdf_document_processor` tool only when a new PDF URL is provided.

2. **Search Tools:**
   - Use search tools only if the user explicitly requests updated or external information (e.g., "latest info", "recent updates", "current status", "search for", "find information about").
   - Never use search tools automatically after processing a PDF.
   - If unsure, ask the user for clarification.

3. **Response Guidelines:**
   - For questions about PDF content, rely solely on the processed document.
   - Use search tools only when explicitly requested.
   - Structure responses clearly and provide citations where appropriate.

**Example:**
- User: "Tell me about this PDF [url]" → Process PDF → Answer using PDF content.
- User: "Get the latest information about this case" → Use search tool → Provide updated information.

If asked to perform tasks outside this scope, politely decline and refer the user to Lextech Ecosystems Limited.

**Objective:**  
Provide accurate, well-researched, and professional legal insights, leveraging all available tools, especially your PDF processing and internal search capabilities.

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

# Prepare tools - Now includes PDF RAG tool
tools = [google_search_tool, pdf_rag_tool, tavily_search_tool]

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
    debug=True
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
                len(judge_vectorstore.store) > 0 and
                any(keyword in query for keyword in ['find', 'search', 'look up', 'what about', 'where is', 'show me', 'in the document', 'from the pdf'])
            )
            
            if is_search_query:
                # Use internal search instead of tool call
                search_result = judge_search_documents(query)
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
        'llama_pdf_document_processor': {
            'start': '📄 Let me process the document...',
            'loading': '⏳ Processing PDF content...',
            'icon': '📄'
        },
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
        
        if messages and len(judge_vectorstore.store) > 0:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                query = last_message.content.lower()
                is_search_query = (
                    not query.startswith(('http://', 'https://', 'www.')) and
                    any(keyword in query for keyword in ['find', 'search', 'look up', 'what about', 'where is', 'show me', 'in the document', 'from the pdf'])
                )
                
                if is_search_query:
                    # Perform internal search and yield results directly
                    search_result = judge_search_documents(query)
                    yield search_result
                    return
        
        # For other cases, use standard tool streaming
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
                chunk = event["data"]["chunk"]
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
            
            # Handle agent completion
            elif event["event"] == "on_chain_end" and "agent" in event.get("name", ""):
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

# Migrate existing documents on startup
migrate_processed_documents()

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