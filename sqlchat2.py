import streamlit as st 
import sqlite3 as sql
import pandas as pd
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent

# Set API keys from session state
openai_api_key = st.secrets["api_key"]
llm = init_chat_model("gpt-4o-mini", model_provider="openai", openai_api_key=openai_api_key)
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
assert len(prompt_template.messages) == 1
system_message = prompt_template.format(dialect="SQLite", top_k=5)
assert len(query_prompt_template.messages) == 1


st.write("Create Database")

db = sql.connect('Chinook.db')
cursor = db.cursor() #cursor object
with open('Chinook_Sqlite.sql', 'r') as f: #Not sure if the 'r' is necessary, but recommended.
     cursor.executescript(f.read())

st.write(cursor.execute("SELECT * FROM Artist LIMIT 10;"))

# db1 = SQLDatabase.from_uri("sqlite:///Chinook.db")
# db1 = SQLDatabase.from_uri("Chinook.db")

class State(TypedDict):
  question: str
  query: str
  result: str
  answer: str

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

agent_executor = create_react_agent(llm, tools, prompt=system_message)

st.write("after create database")

user_input = input("Get me the list of all the artists.")
for step in agent_executor.stream(
     {"messages": [{"role": "user", "content": user_input}]},
     stream_mode="values",
):
     # step["messages"][-1].pretty_print()
     st.write(step["messages"][-1].pretty_print())

st.write("After Stream")

# df = pd.read_sql('SELECT * FROM Artist LIMIT 10', db)
# st.write(df.to_string())
