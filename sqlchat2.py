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

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
db.run("SELECT * FROM Artist LIMIT 10;")
# db1 = SQLDatabase.from_uri("sqlite:///Chinook.db")
# db1 = SQLDatabase.from_uri("Chinook.db")

class State(TypedDict):
  question: str
  query: str
  result: str
  answer: str

llm = init_chat_model("gpt-4o-mini", model_provider="openai", openai_api_key=openai_api_key)

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

assert len(query_prompt_template.messages) == 1
query_prompt_template.messages[0].pretty_print()

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
# graph = graph_builder.compile()

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["execute_query"])

# Now that we're using persistence, we need to specify a thread ID
# so that we can continue the run after review.
config = {"configurable": {"thread_id": "1"}}
# display(Image(graph.get_graph().draw_mermaid_png()))

for step in graph.stream(
    {"question": "Which artist made the most money?"},
    config,
    stream_mode="updates",
):
    st.write(step)

# If approved, continue the graph execution
for step in graph.stream(None, config, stream_mode="updates"):
     st.write(step)
