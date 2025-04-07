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

# Set API keys from session state
openai_api_key = st.secrets["api_key"]
llm = init_chat_model("gpt-4o-mini", model_provider="openai", openai_api_key=openai_api_key)
# Now that we're using persistence, we need to specify a thread ID
# so that we can continue the run after review.
config = {"configurable": {"thread_id": "1"}}

#db = sql.connect('wad2024.db')
#cursor = db.cursor() #cursor object
#with open('wad2024.sql', 'r') as f: #Not sure if the 'r' is necessary, but recommended.
#     cursor.executescript(f.read())

# st.write(cursor.execute("SELECT * FROM General LIMIT 10;"))

db = SQLDatabase.from_uri("sqlite:///wad2024.db")
st.write(db.run("SELECT * FROM General LIMIT 10;"))

# print(db.dialect)
# print(db.get_usable_table_names())
# db.run("SELECT * FROM Artist LIMIT 10;")

class State(TypedDict):
  question: str
  query: str
  result: str
  answer: str

llm = init_chat_model("gpt-4o-mini", model_provider="openai", openai_api_key=openai_api_key)

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
assert len(prompt_template.messages) == 1
# prompt_template.messages[0].pretty_print()
system_message = prompt_template.format(dialect="SQLite", top_k=5)

assert len(query_prompt_template.messages) == 1
# query_prompt_template.messages[0].pretty_print()

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
# print(tools)

agent_executor = create_react_agent(llm, tools, prompt=system_message)

# question = "Which country's customers spent the most?"
# question = "Describe the playlisttrack table"
prompt = st.text_area("Please enter what you want to know about info in the WAD.")

if st.button("Submit to AI", type="primary"):
    # user_input = input("User: ")
    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
