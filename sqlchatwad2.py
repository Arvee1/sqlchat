import streamlit as st 
import sqlite3 as sql
import pandas as pd 
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from typing_extensions import Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display 
from langchain_core.messages import HumanMessage

# Inject CSS to hide the top right icon bar
hide_github_icon = """
            <style>
            /* Hide the GitHub icon */
            .viewerBadge_link__1S137 {visibility: hidden;}
            </style>
            """
st.markdown(hide_github_icon, unsafe_allow_html=True)

# Set API keys from session state
openai_api_key = st.secrets["api_key"]
llm = init_chat_model("gpt-4o-mini", model_provider="openai", openai_api_key=openai_api_key)
# Now that we're using persistence, we need to specify a thread ID
# so that we can continue the run after review.
config = {"configurable": {"thread_id": "1"}}

db = SQLDatabase.from_uri("sqlite:///wad2024.db")

class State(TypedDict):
  question: str
  query: str
  result: str
  answer: str

llm = init_chat_model("gpt-4o-mini", model_provider="openai", openai_api_key=openai_api_key)

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
assert len(prompt_template.messages) == 1
system_message = prompt_template.format(dialect="SQLite", top_k=5)

assert len(query_prompt_template.messages) == 1

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

agent_executor = create_react_agent(llm, tools, prompt=system_message)

st.title("👨‍💻 Wazzup!!!! What do you want to know about Australian Workplace Agreements?")
st.write("The Workplace Agreements Database represents all workplace agreements in Australia. The data in this instance is from the 2024 Full WAD Dataset.")

# Display a special note at the beginning of your app
st.warning("This app is experimental and data should not be treated as correct.")

prompt = st.text_area("Please enter what you want to know about info in the WAD.")

if st.button("Submit to AI", type="primary"):
    # Stream the response and display each step
    result_container = st.empty()  # Create a placeholder for the result
  
    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        stream_mode="values",
    ):
        message = step["messages"][-1]
        content = getattr(message, 'content', 'No content available')
        
        # Display each step's content on the app
        result_container.write(content)
