import streamlit as st
import sqlite3 as sql
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# LangChain imports organized by category
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATABASE_PATH = "my_aec.db"
DEFAULT_THREAD_ID = "1"
MODEL_NAME = "gpt-4o-mini"
MODEL_PROVIDER = "openai"

class State(TypedDict):
    """State structure for the SQL agent workflow."""
    question: str
    query: str
    result: str
    answer: str

class SQLAgentApp:
    """Encapsulates the SQL Agent application logic."""
    
    def __init__(self):
        self.llm = None
        self.db = None
        self.agent_executor = None
        self._initialize_components()
    
    def _get_api_key(self) -> str:
        """Retrieve API key from Streamlit secrets with error handling."""
        try:
            return st.secrets["api_key"]
        except KeyError:
            st.error("❌ API key not found in secrets. Please check your configuration.")
            st.stop()
    
    def _initialize_llm(self) -> None:
        """Initialize the language model with error handling."""
        try:
            api_key = self._get_api_key()
            self.llm = init_chat_model(
                MODEL_NAME, 
                model_provider=MODEL_PROVIDER, 
                openai_api_key=api_key
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            st.error(f"❌ Failed to initialize language model: {str(e)}")
            st.stop()
    
    def _initialize_database(self) -> None:
        """Initialize database connection with validation."""
        try:
            # Check if database file exists
            if not Path(DATABASE_PATH).exists():
                st.error(f"❌ Database file '{DATABASE_PATH}' not found.")
                st.stop()
            
            self.db = SQLDatabase.from_uri(f"sqlite:///{DATABASE_PATH}")
            
            # Test database connection
            self.db.run("SELECT 1")
            logger.info("Database connection established successfully")
            
        except Exception as e:
            st.error(f"❌ Failed to connect to database: {str(e)}")
            st.stop()
    
    def _initialize_agent(self) -> None:
        """Initialize the SQL agent with proper configuration."""
        try:
            # Pull prompts from hub with error handling
            query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
            prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
            
            # Validate prompt templates
            if not prompt_template.messages:
                raise ValueError("Empty prompt template received from hub")
            
            system_message = prompt_template.format(dialect="SQLite", top_k=5)
            
            # Create toolkit and tools
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            tools = toolkit.get_tools()
            
            # Create agent executor
            self.agent_executor = create_react_agent(
                self.llm, 
                tools, 
                prompt=system_message
            )
            
            logger.info("SQL agent initialized successfully")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize SQL agent: {str(e)}")
            st.stop()
    
    def _initialize_components(self) -> None:
        """Initialize all components in the correct order."""
        self._initialize_llm()
        self._initialize_database()
        self._initialize_agent()
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database structure."""
        try:
            tables = self.db.get_table_names()
            table_info = {}
            
            for table in tables[:5]:  # Limit to first 5 tables for performance
                try:
                    sample_rows = self.db.run(f"SELECT * FROM {table} LIMIT 3")
                    table_info[table] = sample_rows
                except Exception as e:
                    logger.warning(f"Could not get info for table {table}: {e}")
                    table_info[table] = "Unable to fetch sample data"
            
            return {
                "tables": tables,
                "table_info": table_info,
                "total_tables": len(tables)
            }
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {"error": str(e)}
    
    def process_query(self, user_query: str) -> None:
        """Process user query through the SQL agent with streaming."""
        if not user_query.strip():
            st.warning("⚠️ Please enter a question about the Annual Returns dataset.")
            return
        
        try:
            with st.spinner("🤖 Processing your query..."):
                # Create containers for different types of output
                result_container = st.container()
                
                # Stream the response
                for step in self.agent_executor.stream(
                    {"messages": [{"role": "user", "content": user_query}]},
                    stream_mode="values",
                ):
                    message = step["messages"][-1]
                    content = getattr(message, 'content', 'No content available')
                    
                    # Display the content with proper formatting
                    with result_container:
                        st.markdown("### 🔍 Agent Response:")
                        st.write(content)
                        
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            logger.error(f"Query processing error: {e}")

def setup_page_config() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="AEC Annual Returns Query Tool",
        page_icon="👨‍💻",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_sidebar(app: SQLAgentApp) -> None:
    """Render the sidebar with database information."""
    with st.sidebar:
        st.header("📊 Database Information")
        
        if st.button("🔄 Refresh Database Info"):
            with st.spinner("Loading database information..."):
                db_info = app.get_database_info()
                
                if "error" not in db_info:
                    st.success(f"✅ Connected to database with {db_info['total_tables']} tables")
                    
                    with st.expander("📋 Available Tables"):
                        for table in db_info["tables"]:
                            st.write(f"• {table}")
                    
                    with st.expander("🔍 Sample Data"):
                        for table, data in db_info["table_info"].items():
                            st.subheader(table)
                            st.text(str(data)[:200] + "..." if len(str(data)) > 200 else str(data))
                else:
                    st.error(f"Database error: {db_info['error']}")

def main():
    """Main application entry point."""
    setup_page_config()
    
    # Initialize the application
    try:
        app = SQLAgentApp()
    except Exception:
        return  # Error handling is done in the class
    
    # Render the main interface
    st.title("👨‍💻 AEC Annual Returns Query Tool")
    st.markdown("### 🔍 Ask questions about Annual Returns collected by the AEC")
    
    # Display warning
    st.warning("⚠️ **Experimental Tool**: This app is experimental and data should not be treated as authoritative.")
    
    # Render sidebar
    render_sidebar(app)
    
    # Main query interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_query = st.text_area(
            "💬 Enter your question about Annual Returns data:",
            placeholder="e.g., How many annual returns were submitted in 2023?",
            height=100
        )
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button("🚀 Submit Query", type="primary", use_container_width=True):
            app.process_query(user_query)
        
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()
    
    # Add usage tips
    with st.expander("💡 Usage Tips"):
        st.markdown("""
        - Be specific in your questions
        - Ask about trends, counts, or specific data points
        - The AI will generate SQL queries to answer your questions
        - Example questions:
          - "What are the top 5 entities by donation amount?"
          - "How many returns were filed each year?"
          - "Show me returns from a specific state"
        """)

if __name__ == "__main__":
    main()
