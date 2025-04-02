import streamlit as st
import sqlite3 as sql
import pandas as pd

st.write("Hello World")

connection = sqlite3.connect('Chinook.db')
cursor = connection.cursor() #cursor object
with open('Chinook_Sqlite.sql', 'r') as f: #Not sure if the 'r' is necessary, but recommended.
     cursor.executescript(f.read())

st.write("after create database")

df = pd.read_sql('SELECT * FROM table LIMIT 10', connection)
st.write(df)
