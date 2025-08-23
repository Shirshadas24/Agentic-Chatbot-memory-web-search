# import streamlit as st

# st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
# st.title("Agentic Bot")
# st.write("Create and Interact with the AI Agents!")

# system_prompt=st.text_area("Define your AI Agent: ", height=70, placeholder="Type your system prompt here...")

# MODEL_NAMES_GROQ = [ "openai/gpt-oss-120b", "llama-3.3-70b-versatile"]
# MODEL_NAMES_COHERE = ["command-r"]
# MODEL_NAMES_GEMINI=["gemini-2.5-flash"]

# provider=st.radio("Select Provider:", ("Groq", "Cohere","Gemini"))

# if provider == "Groq":
#     selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
# elif provider == "Cohere":
#     selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_COHERE)
# elif provider== "Gemini":
#     selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_GEMINI)
# allow_web_search=st.checkbox("Allow Web Search")

# user_query=st.text_area("Enter your query: ", height=150, placeholder="Ask Anything!")

# API_URL="http://127.0.0.1:9999/chat"

# if st.button("Ask Agent!"):
#     if user_query.strip():
#         #Step2: Connect with backend via URL
#         import requests

#         payload={
#             "model_name": selected_model,
#             "model_provider": provider,
#             "system_prompt": system_prompt,
#             "messages": [user_query],
#             "allow_search": allow_web_search
#         }

#         response=requests.post(API_URL, json=payload)
#         if response.status_code == 200:
#             response_data = response.json()
#             if "error" in response_data:
#                 st.error(response_data["error"])
#             else:
#                 st.subheader("Agent Response")
#                 st.markdown(f"**Final Response:** {response_data}")
import streamlit as st
import requests
import uuid

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("Agentic Bot with Memory")

API_URL = "http://127.0.0.1:9999"

# ---------- Session Setup ----------
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = None
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# ---------- Sidebar ----------
st.sidebar.title("Conversations")

if st.sidebar.button("New Chat"):
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["message_history"] = []

threads = requests.get(f"{API_URL}/threads").json()["threads"]
for t_id in threads[::-1]:
    if st.sidebar.button(f"Thread {t_id[:8]}..."):
        st.session_state["thread_id"] = t_id
        history = requests.get(f"{API_URL}/history/{t_id}").json()["history"]
        st.session_state["message_history"] = history

# ---------- Model / Settings ----------
system_prompt = st.text_area("Define your AI Agent:", height=70, placeholder="Type system prompt...")

MODEL_NAMES_GROQ = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"]
MODEL_NAMES_COHERE = ["command-r"]
MODEL_NAMES_GEMINI = ["gemini-2.5-flash-lite"]

provider = st.radio("Select Provider:", ("Groq", "Cohere", "Gemini"))

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "Cohere":
    selected_model = st.selectbox("Select Cohere Model:", MODEL_NAMES_COHERE)
elif provider == "Gemini":
    selected_model = st.selectbox("Select Gemini Model:", MODEL_NAMES_GEMINI)

allow_web_search = st.checkbox("Allow Web Search")

# ---------- Chat UI ----------
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_query = st.chat_input("Ask anything...")

if user_query:
    # Display user msg
    st.session_state["message_history"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.text(user_query)

    payload = {
        "model_name": selected_model,
        "model_provider": provider,
        "system_prompt": system_prompt,
        "messages": [user_query],
        "allow_search": allow_web_search,
        "thread_id": st.session_state["thread_id"],
    }

    response = requests.post(f"{API_URL}/chat", json=payload).json()
    st.session_state["thread_id"] = response["thread_id"]

    # Show AI response
    ai_response = response["response"]
    st.session_state["message_history"].append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.text(ai_response)
