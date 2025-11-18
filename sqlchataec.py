import streamlit as st 
import sqlite3 as sql
import pandas as pd 
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from typing_extensions import Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set API keys from session state 
openai_api_key = st.secrets["api_key"] 

# Initialize LLM
llm = init_chat_model("gpt-4o-mini", model_provider="openai", openai_api_key=openai_api_key)

# Thread configuration
config = {"configurable": {"thread_id": "1"}}

# Database connection
db = SQLDatabase.from_uri("sqlite:///my_aec.db")

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

# Get table names for system message
try:
    table_names = db.get_usable_table_names()
except AttributeError:
    table_names = db.get_table_names()

# Create system message directly (no hub.pull needed)
system_message = f"""You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

Here are the available tables: {', '.join(table_names)}

The SQL dialect is: SQLite

When querying about AEC Annual Returns data:
- Use relevant table names from the list above
- Look for columns related to donors, recipients, amounts, dates, and financial years
- Provide clear, formatted results with proper column names
- If searching for a specific entity, use LIKE with wildcards for flexible matching
"""

# Create toolkit and tools
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# Create agent with proper parameters for newer LangGraph versions
agent_executor = None
try:
    # Try newest version (messages_modifier)
    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        messages_modifier=system_message
    )
    logger.info("Agent created with messages_modifier")
except TypeError as e1:
    logger.info(f"messages_modifier not supported: {e1}")
    try:
        # Try older version (state_modifier)
        agent_executor = create_react_agent(
            model=llm,
            tools=tools,
            state_modifier=system_message
        )
        logger.info("Agent created with state_modifier")
    except TypeError as e2:
        logger.info(f"state_modifier not supported: {e2}")
        # Create without system message as fallback
        agent_executor = create_react_agent(
            model=llm,
            tools=tools
        )
        logger.warning("Agent created without system message")

if not agent_executor:
    st.error("Failed to create database agent. Please check your configuration.")
    st.stop()

# Streamlit UI
st.title("👨‍💻 Wazzup!!!! What do you want to know about Annual Returns collected by the AEC?")
st.write("This is a test on annual returns Dataset.")

# Display a special note at the beginning of your app
st.warning("⚠️ This app is experimental and data should not be treated as correct.")

# Show available tables
with st.expander("📊 Available Database Tables"):
    st.write(f"Found {len(table_names)} tables:")
    for table in table_names:
        st.write(f"- {table}")

# Input area
prompt = st.text_area(
    "Please enter what you want to know about info on Annual Returns:",
    placeholder="e.g., What are the largest donations in 2023? Who donated to the Liberal Party?",
    height=100
)

# Example queries
with st.expander("💡 Example Queries"):
    st.markdown("""
    Try asking questions like:
    - Who donated the most money in 2023?
    - Show me all donations to the Labor Party
    - What are the top 5 donors by total amount?
    - List all donations over $100,000
    - Who are the most frequent donors?
    """)

if st.button("Submit to AI", type="primary", disabled=not prompt.strip()):
    if prompt.strip():
        # Create containers for organized output
        with st.spinner("🤖 AI is analyzing your question..."):
            try:
                # Container for the final answer
                answer_container = st.container()
                
                # Container for intermediate steps (optional, collapsible)
                with st.expander("🔍 View Processing Steps", expanded=False):
                    steps_container = st.empty()
                    all_steps = []
                    
                    # Stream the response - FIXED INDENTATION HERE
                    for step in agent_executor.stream(
                        {"messages": [{"role": "user", "content": prompt}]},
                        stream_mode="values",
                    ):
                        message = step["messages"][-1]
                        content = getattr(message, 'content', 'No content available')
                        
                        # Collect steps
                        all_steps.append(content)
                        
                        # Display intermediate steps
                        steps_container.text("\n\n".join(all_steps))
                
                # Display final answer in main container
                if all_steps:
                    final_answer = all_steps[-1]
                    with answer_container:
                        st.markdown("### 📊 Answer:")
                        st.markdown(final_answer)
                        
                        # Check if the answer seems to contain structured data
                        if "|" in final_answer or "\n" in final_answer:
                            st.info("💡 Tip: The answer above may contain tabular data. Consider copying it to a spreadsheet for better viewing.")
                else:
                    st.warning("No response received from the AI.")
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                logger.error(f"Query execution error: {e}", exc_info=True)
                
                with st.expander("🔧 Troubleshooting"):
                    st.markdown("""
                    **Common Issues:**
                    1. **Database Error**: Check if the database file exists and is accessible
                    2. **Query Too Complex**: Try simplifying your question
                    3. **Table/Column Not Found**: Check the available tables above
                    4. **API Key Issue**: Ensure your OpenAI API key is valid
                    """)
    else:
        st.warning("Please enter a question before submitting.")

# Footer with helpful info
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
    💾 Connected to AEC Annual Returns Database | 
    Powered by GPT-4o-mini & LangChain | 
    ⚠️ Experimental - Verify all results
    </small>
</div>
""", unsafe_allow_html=True)
