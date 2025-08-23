# from pydantic import BaseModel
# from fastapi import FastAPI
# from typing import List
# from ai_agent import get_response_from_ai_agent
# class RequestState(BaseModel):
#     model_name:str
#     model_provider:str
#     system_prompt:str
#     messages:List[str]
#     allow_search:bool
# ALLOWED_MODEL_NAMES=[ "openai/gpt-oss-120b", "llama-3.3-70b-versatile", "command-r", "gemini-2.5-flash-lite"]
# app=FastAPI(title="LangGraph AI Agent")

# @app.post("/chat")
# def chat_endpoint(request: RequestState): 
#     """
#     API Endpoint to interact with the Chatbot using LangGraph and search tools.
#     It dynamically selects the model specified in the request
#     """
#     if request.model_name not in ALLOWED_MODEL_NAMES:
#         return {"error": "Invalid model name. Kindly select a valid AI model"}
    
#     llm_id = request.model_name
#     query = request.messages
#     allow_search = request.allow_search
#     system_prompt = request.system_prompt
#     provider = request.model_provider

#     # Create AI Agent and get response from it! 
#     response=get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider)
#     return response

# #Step3: Run app & Explore Swagger UI Docs
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=9999)



from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, TypedDict, Annotated

import uuid
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# ---------- your existing import ----------
from ai_agent import get_response_from_ai_agent

# ------------------- Config -------------------
ALLOWED_MODEL_NAMES = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "command-r",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",        
]

app = FastAPI(title="LangGraph AI Agent with Memory")

# ----------------- SQLite Checkpointer -----------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ------------------- LangGraph State -------------------
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    model_name: str
    model_provider: str
    system_prompt: str
    allow_search: bool

def chat_node(state: ChatState):
    """
    Single node that calls your ai_agent with full history + controls,
    and returns an AIMessage. Memory is handled by LangGraph + SqliteSaver.
    """
    model_name = state["model_name"]
    provider = state["model_provider"]
    system_prompt = state["system_prompt"]
    allow_search = state["allow_search"]

    history_texts = [m.content for m in state["messages"]]

    ai_text = get_response_from_ai_agent(
        model_name,              
        history_texts,           
        allow_search,            
        system_prompt,           
        provider                
    )

    return {"messages": [AIMessage(content=ai_text)]}

# Build the graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

# ------------------- API Schemas -------------------
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]      
    allow_search: bool
    thread_id: Optional[str] = None

# ------------------- Helpers -------------------
def new_thread_id() -> str:
    return str(uuid.uuid4())

def get_history_messages(thread_id: str) -> List[BaseMessage]:
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    if state and "messages" in state.values:
        return state.values["messages"]
    return []

@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    Continues a conversation (by thread_id) or starts a new one.
    Uses LangGraph + SqliteSaver to persist memory.
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model"}

    thread_id = request.thread_id or new_thread_id()

    latest_user_text = request.messages[-1] if request.messages else ""

    input_state: ChatState = {
        "messages": [HumanMessage(content=latest_user_text)],
        "model_name": request.model_name,
        "model_provider": request.model_provider,
        "system_prompt": request.system_prompt,
        "allow_search": request.allow_search,
    }


    result = chatbot.invoke(
        input_state,
        config={"configurable": {"thread_id": thread_id}},
    )

    ai_msg = result["messages"][-1].content if result.get("messages") else ""

    return {"thread_id": thread_id, "response": ai_msg}

@app.get("/threads")
def list_threads():
    """
    Returns all known thread IDs from the Sqlite checkpointer.
    """
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return {"threads": list(all_threads)}

@app.get("/history/{thread_id}")
def get_history(thread_id: str):
    """
    Returns the role/content history for a given thread_id.
    """
    messages = get_history_messages(thread_id)
    history = []
    for m in messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        history.append({"role": role, "content": m.content})
    return {"thread_id": thread_id, "history": history}

# ------------------- Dev server -------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
