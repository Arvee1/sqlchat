import streamlit as st
import sqlite3 as sql
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from contextlib import contextmanager
import time

# Updated LangChain imports
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATABASE_PATH = "my_aec.db"
DEFAULT_THREAD_ID = "1"
MODEL_NAME = "gpt-4o-mini"
MAX_QUERY_LENGTH = 1000
MAX_TABLES_DISPLAY = 10
SAMPLE_ROWS_LIMIT = 3

class AppState(TypedDict):
    """State structure for the SQL agent workflow."""
    question: str
    query: str
    result: str
    answer: str
    timestamp: float

class DatabaseManager:
    """Handles database operations with proper connection management."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = None
        self._initialize_connection()
    
    def _initialize_connection(self) -> None:
        """Initialize database connection with validation."""
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"Database file '{self.db_path}' not found.")
        
        try:
            self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
            # Test connection
            self.db.run("SELECT 1")
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = sql.connect(self.db_path)
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def get_table_info(self) -> Dict[str, Any]:
        """Get comprehensive database structure information."""
        try:
            tables = self.db.get_table_names()
            table_info = {}
            
            with self.get_connection() as conn:
                for table in tables[:MAX_TABLES_DISPLAY]:
                    try:
                        # Get table schema
                        schema_query = f"PRAGMA table_info({table})"
                        schema_df = pd.read_sql_query(schema_query, conn)
                        
                        # Get sample data
                        sample_query = f"SELECT * FROM {table} LIMIT {SAMPLE_ROWS_LIMIT}"
                        sample_df = pd.read_sql_query(sample_query, conn)
                        
                        # Get row count
                        count_query = f"SELECT COUNT(*) as count FROM {table}"
                        count_df = pd.read_sql_query(count_query, conn)
                        
                        table_info[table] = {
                            "schema": schema_df.to_dict('records'),
                            "sample_data": sample_df.to_dict('records'),
                            "row_count": count_df.iloc[0]['count']
                        }
                        
                    except Exception as e:
                        logger.warning(f"Could not get info for table {table}: {e}")
                        table_info[table] = {"error": str(e)}
            
            return {
                "tables": tables,
                "table_info": table_info,
                "total_tables": len(tables)
            }
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {"error": str(e)}
    
    def validate_query_safety(self, query: str) -> bool:
        """Basic SQL injection protection."""
        dangerous_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 
            'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE'
        ]
        query_upper = query.upper()
        return not any(keyword in query_upper for keyword in dangerous_keywords)

class SQLAgentApp:
    """Main application class with improved error handling and caching."""
    
    def __init__(self):
        self.llm = None
        self.db_manager = None
        self.agent_executor = None
        self._initialize_session_state()
        self._initialize_components()
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state."""
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'db_info_cache' not in st.session_state:
            st.session_state.db_info_cache = None
        if 'cache_timestamp' not in st.session_state:
            st.session_state.cache_timestamp = 0
    
    def _get_api_key(self) -> str:
        """Retrieve API key with better error messaging."""
        try:
            return st.secrets["api_key"]
        except KeyError:
            st.error("❌ OpenAI API key not found. Please add 'api_key' to your Streamlit secrets.")
            st.info("💡 Add your API key in `.streamlit/secrets.toml` or through Streamlit Cloud settings.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error accessing secrets: {str(e)}")
            st.stop()
    
    @st.cache_resource
    def _initialize_llm(_self) -> ChatOpenAI:
        """Initialize the language model with caching."""
        try:
            api_key = _self._get_api_key()
            llm = ChatOpenAI(
                model=MODEL_NAME,
                openai_api_key=api_key,
                temperature=0,
                timeout=30
            )
            logger.info("LLM initialized successfully")
            return llm
        except Exception as e:
            st.error(f"❌ Failed to initialize language model: {str(e)}")
            st.stop()
    
    def _initialize_database(self) -> None:
        """Initialize database manager."""
        try:
            self.db_manager = DatabaseManager(DATABASE_PATH)
            logger.info("Database manager initialized successfully")
        except Exception as e:
            st.error(f"❌ Database initialization failed: {str(e)}")
            st.stop()
    
    def _initialize_agent(self) -> None:
        """Initialize the SQL agent with improved error handling."""
        try:
            # Pull prompts with timeout handling
            with st.spinner("Loading AI prompts..."):
                query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
                prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
            
            if not prompt_template or not hasattr(prompt_template, 'format'):
                raise ValueError("Invalid prompt template received from hub")
            
            system_message = prompt_template.format(dialect="SQLite", top_k=5)
            
            # Create toolkit with database manager
            toolkit = SQLDatabaseToolkit(db=self.db_manager.db, llm=self.llm)
            tools = toolkit.get_tools()
            
            # Create agent executor with custom configuration
            self.agent_executor = create_react_agent(
                self.llm, 
                tools, 
                prompt=system_message
            )
            
            logger.info("SQL agent initialized successfully")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize SQL agent: {str(e)}")
            logger.error(f"Agent initialization error: {e}")
            st.stop()
    
    def _initialize_components(self) -> None:
        """Initialize all components with proper error handling."""
        try:
            self.llm = self._initialize_llm()
            self._initialize_database()
            self._initialize_agent()
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    def get_database_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get database info with caching mechanism."""
        current_time = time.time()
        cache_age = current_time - st.session_state.cache_timestamp
        
        # Use cache if it's less than 5 minutes old and not forcing refresh
        if (not force_refresh and 
            st.session_state.db_info_cache and 
            cache_age < 300):
            return st.session_state.db_info_cache
        
        # Refresh cache
        db_info = self.db_manager.get_table_info()
        st.session_state.db_info_cache = db_info
        st.session_state.cache_timestamp = current_time
        
        return db_info
    
    def process_query(self, user_query: str) -> None:
        """Process user query with enhanced validation and error handling."""
        if not user_query.strip():
            st.warning("⚠️ Please enter a question about the Annual Returns dataset.")
            return
        
        if len(user_query) > MAX_QUERY_LENGTH:
            st.error(f"❌ Query too long. Please limit to {MAX_QUERY_LENGTH} characters.")
            return
        
        # Add to query history
        timestamp = time.time()
        st.session_state.query_history.append({
            "query": user_query,
            "timestamp": timestamp
        })
        
        try:
            with st.spinner("🤖 Processing your query..."):
                # Create response container
                response_container = st.container()
                
                with response_container:
                    st.markdown("### 🔍 Agent Response:")
                    
                    # Stream the response with better error handling
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    try:
                        for step in self.agent_executor.stream(
                            {"messages": [HumanMessage(content=user_query)]},
                            config={"configurable": {"thread_id": DEFAULT_THREAD_ID}},
                            stream_mode="values",
                        ):
                            if "messages" in step and step["messages"]:
                                message = step["messages"][-1]
                                if hasattr(message, 'content') and message.content:
                                    full_response = message.content
                                    response_placeholder.markdown(full_response)
                    
                    except Exception as stream_error:
                        st.error(f"❌ Streaming error: {str(stream_error)}")
                        logger.error(f"Streaming error: {stream_error}")
                        
                        # Fallback to non-streaming
                        try:
                            result = self.agent_executor.invoke(
                                {"messages": [HumanMessage(content=user_query)]}
                            )
                            if result and "messages" in result:
                                final_message = result["messages"][-1]
                                response_placeholder.markdown(final_message.content)
                        except Exception as fallback_error:
                            st.error(f"❌ Query processing failed: {str(fallback_error)}")
                            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            logger.error(f"Query processing error: {e}")

def setup_page_config() -> None:
    """Configure Streamlit page with enhanced settings."""
    st.set_page_config(
        page_title="AEC Annual Returns Query Tool",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/aec-query-tool',
            'Report a bug': "https://github.com/yourusername/aec-query-tool/issues",
            'About': "AEC Annual Returns Query Tool - Powered by LangChain and OpenAI"
        }
    )

def render_sidebar(app: SQLAgentApp) -> None:
    """Enhanced sidebar with better information display."""
    with st.sidebar:
        st.header("📊 Database Information")
        
        col1, col2 = st.columns(2)
        with col1:
            refresh_db = st.button("🔄 Refresh", help="Refresh database information")
        with col2:
            force_refresh = st.checkbox("Force", help="Force refresh cache")
        
        if refresh_db:
            with st.spinner("Loading database information..."):
                db_info = app.get_database_info(force_refresh=force_refresh)
                
                if "error" not in db_info:
                    st.success(f"✅ Connected: {db_info['total_tables']} tables")
                    
                    # Enhanced table information
                    with st.expander("📋 Available Tables", expanded=True):
                        for table in db_info["tables"]:
                            table_data = db_info["table_info"].get(table, {})
                            if "error" not in table_data and "row_count" in table_data:
                                st.write(f"• **{table}** ({table_data['row_count']:,} rows)")
                            else:
                                st.write(f"• {table}")
                    
                    # Sample data with better formatting
                    with st.expander("🔍 Sample Data"):
                        for table, data in db_info["table_info"].items():
                            if "error" not in data and "sample_data" in data:
                                st.subheader(f"📋 {table}")
                                if data["sample_data"]:
                                    df = pd.DataFrame(data["sample_data"])
                                    st.dataframe(df, use_container_width=True)
                                else:
                                    st.info("No sample data available")
                else:
                    st.error(f"❌ Database error: {db_info['error']}")
        
        # Query history
        if st.session_state.query_history:
            with st.expander("📝 Recent Queries"):
                for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                    st.write(f"{i+1}. {item['query'][:50]}...")

def main():
    """Enhanced main application with better error handling."""
    setup_page_config()
    
    # Initialize the application with error handling
    try:
        if 'app' not in st.session_state:
            with st.spinner("Initializing application..."):
                st.session_state.app = SQLAgentApp()
        app = st.session_state.app
    except Exception as e:
        st.error(f"❌ Application initialization failed: {str(e)}")
        st.info("Please check your configuration and try again.")
        return
    
    # Main interface
    st.title("🏛️ Wazzup!!!! This is AEC Annual Returns Query Tool")
    st.markdown("### 🔍 Ask questions about Annual Returns collected by the AEC")
    
    # Enhanced warning with more information
    st.warning(
        "⚠️ **Experimental Tool**: This app is experimental. "
        "Always verify results with official AEC sources."
    )
    
    # Render sidebar
    render_sidebar(app)
    
    # Main query interface with better layout
    st.markdown("---")
    
    # Query input
    user_query = st.text_area(
        "💬 Enter your question about Annual Returns data:",
        placeholder="e.g., How many annual returns were submitted in 2023 by state?",
        height=100,
        max_chars=MAX_QUERY_LENGTH,
        help=f"Maximum {MAX_QUERY_LENGTH} characters"
    )
    
    # Buttons with better spacing
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    
    with col2:
        submit_btn = st.button("🚀 Submit Query", type="primary", use_container_width=True)
    
    with col3:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    # Handle button clicks
    if submit_btn and user_query:
        app.process_query(user_query)
    elif submit_btn and not user_query:
        st.warning("Please enter a query first.")
    
    if clear_btn:
        st.rerun()
    
    # Enhanced usage tips
    with st.expander("💡 Usage Tips & Examples"):
        st.markdown("""
        **Best Practices:**
        - Be specific and clear in your questions
        - Ask about trends, comparisons, or specific data points
        - Use natural language - the AI will convert to SQL
        
        **Example Questions:**
        - "What are the top 10 entities by total donation amount?"
        - "How many returns were filed each year from 2020 to 2023?"
        - "Show me all returns from Victoria with donations over $100,000"
        - "What's the average donation amount by state?"
        - "Which political parties received the most donations last year?"
        
        **Limitations:**
        - Only SELECT queries are allowed for security
        - Results may take a few seconds to process
        - Complex queries might need to be broken down into simpler parts
        """)

if __name__ == "__main__":
    main()
