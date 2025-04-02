import streamlit as st
import sqlite3 as sql
import pandas as pd
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent

st.write("Hello World")

connection = sql.connect('Chinook.db')
cursor = connection.cursor() #cursor object
with open('Chinook_Sqlite.sql', 'r') as f: #Not sure if the 'r' is necessary, but recommended.
     cursor.executescript(f.read())

st.write("after create database")

df = pd.read_sql('SELECT * FROM Artist LIMIT 10', connection)
st.write(df.to_string())
