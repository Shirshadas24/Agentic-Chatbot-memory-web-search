from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage
from langchain_groq import ChatGroq
import os
load_dotenv()
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
COHERE_API_KEY=os.environ.get("COHERE_API_KEY")
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")


cohere_llm=ChatCohere(model="command-r")
gemini_llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"))
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")
search_tool=TavilySearch(max_results=2)

#Step3: Setup AI Agent with Search tool functionality

system_prompt="Act as an AI chatbot who is smart and friendly"

# agent=create_react_agent(
#     model=cohere_llm,
#     tools=[search_tool]
# )
# query="tell about a new movie"
# state={"messages":query}
# response=agent.invoke(state)
# print(response)

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider=="Cohere":
        llm=ChatCohere(model=llm_id)
    elif provider=="Gemini":
        llm=ChatGoogleGenerativeAI(model=llm_id,google_api_key=os.getenv("GEMINI_API_KEY"))
    elif provider=="Groq":
        llm=ChatGroq(model=llm_id)
    tools=[TavilySearch(max_results=2)] if allow_search else []
    agent=create_react_agent(
        model=llm,
        tools=tools,
        # state_modifier=system_prompt
    )
    state={"messages": query}
    response=agent.invoke(state)
    messages=response.get("messages")
    ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]
