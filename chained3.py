"""
Integrated Australian Political Donation Analysis System
======================================================

Combines three analysis tools:
1. Web Search - Find articles about political donors
2. Legislation Search - Query reporting requirements and regulations
3. Database Search - Query declared donations from AEC database

This integrated system allows cross-referencing and comprehensive analysis
of political donation information from multiple sources.
"""

import streamlit as st
import sys

# sqlite3 workaround
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import requests
import openai
import replicate
import re
import time
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from functools import wraps
from contextlib import contextmanager
import pandas as pd
import sqlite3 as sql

# ChromaDB and vector search imports
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain SQL agent imports
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== CONFIGURATION ==========
class Config:
    """Unified configuration for all components."""
    # Web Search Config
    MAX_SEARCH_RESULTS = 3
    REQUEST_TIMEOUT = 10
    RATE_LIMIT_CALLS_PER_MINUTE = 10
    MIN_NAME_LENGTH = 2
    
    # RAG Config
    CHROMA_DATA_PATH = "chroma_data/"
    EMBED_MODEL = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "political_docs"
    DOC_FILE = "24146b01_Electoral_Reform.txt"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    DEFAULT_RESULTS = 20
    CONTEXT_RESULTS = 8
    MAX_TOKENS = 800
    TEMPERATURE = 0.4
    TOP_P = 0.8
    
    # Database Config
    DATABASE_PATH = "my_aec.db"
    MODEL_NAME = "gpt-4o-mini"
    MAX_QUERY_LENGTH = 1000
    MAX_TABLES_DISPLAY = 10
    SAMPLE_ROWS_LIMIT = 3
    
    # Logging
    QUERY_LOG_FILE = "integrated_queries.json"

# ========== SHARED UTILITIES ==========
class IntegratedLogger:
    """Centralized logging for all components."""
    
    def __init__(self, config: Config):
        self.config = config
        self.log_file = Path(config.QUERY_LOG_FILE)
        self._ensure_log_file_exists()
    
    def _ensure_log_file_exists(self):
        if not self.log_file.exists():
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
    
    def log_query(self, username: str, component: str, query: str, response: str, metadata: dict = None):
        """Log query across all components."""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "username": username,
                "component": component,
                "query": query,
                "response": response,
                "metadata": metadata or {},
                "session_id": st.session_state.get('session_id', 'unknown')
            }
            
            logs.append(log_entry)
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Logged {component} query for user: {username}")
        except Exception as e:
            logger.error(f"Failed to log query: {e}")

def validate_input(text: str) -> bool:
    """Shared input validation."""
    if not text or not isinstance(text, str):
        return False
    
    if len(text.strip()) == 0 or len(text) > 2000:
        return False
    
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    return True

def rate_limit(calls_per_minute: int = 10):
    """Rate limiting decorator."""
    def decorator(func):
        func.last_called = 0
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            min_interval = 60 / calls_per_minute
            
            if now - func.last_called < min_interval:
                wait_time = min_interval - (now - func.last_called)
                st.warning(f"Please wait {wait_time:.1f} seconds before making another request.")
                return None
            
            func.last_called = now
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ========== COMPONENT 1: WEB SEARCH ==========
class WebSearchComponent:
    """Web search functionality for political donation articles."""
    
    def __init__(self, config: Config, logger: IntegratedLogger):
        self.config = config
        self.logger = logger
    
    @rate_limit(calls_per_minute=10)
    def search_articles(self, query: str) -> Tuple[List[Dict], str]:
        """Search for articles using Tavily API."""
        if not validate_input(query):
            return [], "Invalid search query."
        
        url = "https://api.tavily.com/search"
        
        try:
            if 'tavily_key' not in st.secrets:
                st.error("Tavily API key not configured.")
                return [], "Search service not configured."
            
            headers = {
                "Authorization": f"Bearer {st.secrets['tavily_key']}",
                "Content-Type": "application/json"
            }
            
            data = {"query": f"Australian political donations {query}"}
            
            response = requests.post(url, json=data, headers=headers, timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            results = response.json()
            
            if not results or 'results' not in results:
                return [], "No search results found."
            
            sources = []
            formatted = []
            
            for i, item in enumerate(results.get("results", [])[:self.config.MAX_SEARCH_RESULTS], 1):
                snippet = item.get("content", "No content available")[:500]
                url_link = item.get("url", "")
                title = item.get("title", f"Source {i}")[:100]
                
                source_info = {
                    "id": i,
                    "title": title,
                    "url": url_link,
                    "snippet": snippet
                }
                sources.append(source_info)
                
                if url_link:
                    formatted.append(f'> {snippet}\nSource [{i}]: {title} - {url_link}')
                else:
                    formatted.append(f'> {snippet}\nSource [{i}]: {title}')
            
            formatted_text = "\n\n".join(formatted) if formatted else "No web results found."
            return sources, formatted_text
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return [], f"Search failed: {str(e)}"
    
    def generate_response(self, query: str, sources: List[Dict], search_results: str) -> str:
        """Generate AI response using OpenAI."""
        try:
            if 'api_key' not in st.secrets:
                return "OpenAI API key not configured."
            
            client = openai.Client(api_key=st.secrets["api_key"])
            
            system_prompt = """You are an expert analyst of Australian political donations. 
            Focus on factual information about donation patterns, compliance issues, and transparency.
            When citing sources, use the format 'Source [1]' etc."""
            
            user_message = f"""
            User question: {query}
            
            Available web search results:
            {search_results}
            
            Please analyze this information in the context of Australian political donation transparency 
            and compliance. Cite sources using 'Source [X]' format.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return f"Error generating response: {str(e)}"

# ========== COMPONENT 2: LEGISLATION SEARCH ==========
class LegislationSearchComponent:
    """RAG system for political donation legislation and requirements."""
    
    def __init__(self, config: Config, logger: IntegratedLogger):
        self.config = config
        self.logger = logger
        self.collection = None
        self.chunks = None
        self.use_replicate = False
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB and load documents."""
        try:
            # Check if Replicate token is available
            if 'REPLICATE_API_TOKEN' in st.secrets:
                import os
                os.environ['REPLICATE_API_TOKEN'] = st.secrets['REPLICATE_API_TOKEN']
                self.use_replicate = True
                logger.info("Replicate API configured")
            else:
                logger.warning("Replicate API token not found, will use OpenAI for legislation responses")
            
            client = chromadb.PersistentClient(path=self.config.CHROMA_DATA_PATH)
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.EMBED_MODEL
            )
            self.collection = client.get_or_create_collection(
                name=self.config.COLLECTION_NAME,
                embedding_function=embedding_func
            )
            
            # Load and chunk document
            doc_path = Path(self.config.DOC_FILE)
            if doc_path.exists():
                with open(doc_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.CHUNK_SIZE,
                    chunk_overlap=self.config.CHUNK_OVERLAP
                )
                
                self.chunks = splitter.split_text(content)
                
                # Populate collection if empty
                if self.collection.count() == 0:
                    ids = [f"doc_chunk_{i}" for i in range(len(self.chunks))]
                    self.collection.add(documents=self.chunks, ids=ids)
                    logger.info(f"Added {len(self.chunks)} chunks to vector database")
                else:
                    logger.info(f"Using existing collection with {self.collection.count()} documents")
                    
        except Exception as e:
            logger.error(f"Legislation search initialization error: {e}", exc_info=True)
    
    def search_legislation(self, query: str) -> List[str]:
        """Search legislation documents."""
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return []
            
            results = self.collection.query(
                query_texts=[query],
                include=["documents"],
                n_results=self.config.CONTEXT_RESULTS
            )
            
            docs = results.get("documents", [[]])[0]
            logger.info(f"Found {len(docs)} relevant documents for query: {query[:50]}")
            return docs
            
        except Exception as e:
            logger.error(f"Legislation search error: {e}", exc_info=True)
            return []
    
    def generate_response(self, query: str, context_docs: List[str]) -> str:
        """Generate response using Replicate LLaMA or OpenAI as fallback."""
        try:
            if not context_docs:
                return "No relevant legislation found."
            
            context = '\n\n---\n\n'.join(context_docs)
            
            system_prompt = """You are an expert on Australian political donation laws and requirements.
            Explain legislation and compliance requirements in simple, clear language."""
            
            full_prompt = f"""Based on the following legislation:

CONTEXT:
{context}

USER QUESTION: {query}

Please explain the relevant legal requirements and compliance obligations in simple terms."""

            # Try Replicate first if available
            if self.use_replicate:
                try:
                    logger.info("Using Replicate LLaMA for legislation response")
                    result = ""
                    for event in replicate.stream(
                        "meta/meta-llama-3-70b-instruct",
                        input={
                            "top_p": self.config.TOP_P,
                            "prompt": full_prompt,
                            "max_tokens": self.config.MAX_TOKENS,
                            "temperature": self.config.TEMPERATURE,
                            "prompt_template": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{{prompt}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                        }
                    ):
                        result += str(event)
                    
                    if result.strip():
                        return result.strip()
                    else:
                        logger.warning("Replicate returned empty response, falling back to OpenAI")
                        
                except Exception as replicate_error:
                    logger.error(f"Replicate error: {replicate_error}, falling back to OpenAI")
            
            # Fallback to OpenAI
            logger.info("Using OpenAI for legislation response")
            if 'api_key' not in st.secrets:
                return "AI service not configured. Please add API keys."
            
            client = openai.Client(api_key=st.secrets["api_key"])
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Legislation generation error: {e}", exc_info=True)
            return f"Error generating response: {str(e)}"

# ========== COMPONENT 3: DATABASE SEARCH ==========
class DatabaseSearchComponent:
    """SQL database search for declared donations."""
    
    def __init__(self, config: Config, logger: IntegratedLogger):
        self.config = config
        self.logger = logger
        self.db = None
        self.agent_executor = None
        self._initialize()
    
    def _initialize(self):
        """Initialize database connection and SQL agent."""
        try:
            if not Path(self.config.DATABASE_PATH).exists():
                logger.warning(f"Database file '{self.config.DATABASE_PATH}' not found.")
                return
            
            self.db = SQLDatabase.from_uri(f"sqlite:///{self.config.DATABASE_PATH}")
            logger.info(f"Connected to database: {self.config.DATABASE_PATH}")
            
            # Initialize LLM
            if 'api_key' in st.secrets:
                llm = ChatOpenAI(
                    model=self.config.MODEL_NAME,
                    openai_api_key=st.secrets["api_key"],
                    temperature=0
                )
                
                # Create SQL agent
                toolkit = SQLDatabaseToolkit(db=self.db, llm=llm)
                tools = toolkit.get_tools()
                
                # Get table names (using the non-deprecated method)
                try:
                    table_names = self.db.get_usable_table_names()
                except AttributeError:
                    # Fallback for older versions
                    table_names = self.db.get_table_names()
                
                # Define SQL agent system message
                sql_system_message = f"""You are an agent designed to interact with a SQL database.
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

The SQL dialect is: SQLite"""
                
                # Create the agent - try different parameter names based on version
                try:
                    # Newer LangGraph versions use 'messages_modifier'
                    self.agent_executor = create_react_agent(
                        model=llm,
                        tools=tools,
                        messages_modifier=sql_system_message
                    )
                    logger.info("SQL agent initialized successfully with messages_modifier")
                except TypeError:
                    try:
                        # Some versions might use 'state_modifier'
                        self.agent_executor = create_react_agent(
                            model=llm,
                            tools=tools,
                            state_modifier=sql_system_message
                        )
                        logger.info("SQL agent initialized successfully with state_modifier")
                    except TypeError:
                        # Fallback: create without system message modifier
                        self.agent_executor = create_react_agent(
                            model=llm,
                            tools=tools
                        )
                        logger.warning("SQL agent initialized without system message (parameter not supported in this version)")
                
        except Exception as e:
            logger.error(f"Database initialization error: {e}", exc_info=True)
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database structure information."""
        if not self.db:
            return {"error": "Database not available"}
        
        try:
            # Use non-deprecated method
            try:
                tables = self.db.get_usable_table_names()
            except AttributeError:
                tables = self.db.get_table_names()
                
            return {
                "tables": tables,
                "total_tables": len(tables),
                "status": "connected"
            }
        except Exception as e:
            logger.error(f"Database info error: {e}")
            return {"error": str(e)}
    
    def query_database(self, query: str) -> str:
        """Query database using natural language."""
        if not self.agent_executor:
            return "Database agent not available. Please check database configuration."
        
        try:
            logger.info(f"Querying database: {query[:100]}")
            
            # Add SQL context to the query
            enhanced_query = f"""Using the AEC political donations database, please answer this question: {query}

Available tables include information about donations received, donations made, party returns, third party returns, and donor returns.

Please write a SQL query to answer the question, execute it, and provide a clear answer."""
            
            # Invoke the agent with the correct format
            result = self.agent_executor.invoke(
                {"messages": [HumanMessage(content=enhanced_query)]}
            )
            
            if result and "messages" in result:
                response = result["messages"][-1].content
                logger.info(f"Database query successful, response length: {len(response)}")
                return response
            
            return "No response generated."
            
        except Exception as e:
            logger.error(f"Database query error: {e}", exc_info=True)
            return f"Error querying database: {str(e)}"

# ========== MAIN INTEGRATED APPLICATION ==========
class IntegratedDonationAnalyzer:
    """Main integrated application."""
    
    def __init__(self):
        self.config = Config()
        self.logger = IntegratedLogger(self.config)
        
        # Initialize components
        self.web_search = WebSearchComponent(self.config, self.logger)
        self.legislation_search = LegislationSearchComponent(self.config, self.logger)
        self.database_search = DatabaseSearchComponent(self.config, self.logger)
        
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state."""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
    
    def render_authentication(self):
        """Render user authentication."""
        st.title("🏛️ Australian Political Donation Analysis System")
        st.markdown("### Integrated Analysis of Political Donations, Legislation & Declared Data")
        
        with st.form("auth_form"):
            username = st.text_input(
                "Enter your name to begin:",
                placeholder="Your full name"
            )
            
            if st.form_submit_button("Start Analysis", type="primary"):
                if username.strip():
                    st.session_state.username = username.strip()
                    st.session_state.authenticated = True
                    st.success(f"Welcome, {username}! 🎉")
                    st.rerun()
                else:
                    st.error("Please enter your name.")
    
    def render_main_interface(self):
        """Render main tabbed interface."""
        st.title("🏛️ Australian Political Donation Analysis System")
        st.markdown(f"Welcome, **{st.session_state.username}**!")
        
        # Create tabs for different analysis types
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Web Search", 
            "📋 Legislation", 
            "💾 Database", 
            "📊 Integrated Analysis"
        ])
        
        with tab1:
            self.render_web_search_tab()
        
        with tab2:
            self.render_legislation_tab()
        
        with tab3:
            self.render_database_tab()
        
        with tab4:
            self.render_integrated_analysis_tab()
    
    def render_web_search_tab(self):
        """Render web search interface."""
        st.header("🔍 Web Search for Political Donation Articles")
        st.markdown("Search for recent news and articles about political donations and donors.")
        
        query = st.text_input(
            "Search for articles about:",
            placeholder="e.g., major political donors 2024, donation transparency issues",
            key="web_search_input"
        )
        
        if st.button("🔍 Search Articles", type="primary", key="web_search_btn"):
            if query:
                with st.spinner("Searching for articles..."):
                    sources, search_results = self.web_search.search_articles(query)
                
                if sources:
                    with st.spinner("Analyzing articles..."):
                        response = self.web_search.generate_response(query, sources, search_results)
                    
                    st.markdown("### 📰 Analysis Results:")
                    st.markdown(response)
                    
                    # Log the query
                    self.logger.log_query(
                        st.session_state.username,
                        "web_search",
                        query,
                        response,
                        {"sources_found": len(sources)}
                    )
                    
                    # Show sources
                    with st.expander("📚 Sources"):
                        for source in sources:
                            if source.get('url'):
                                st.markdown(f"**[{source['title']}]({source['url']})**")
                            st.write(source['snippet'])
                            st.divider()
                else:
                    st.warning("No articles found or search failed.")
    
    def render_legislation_tab(self):
        """Render legislation search interface."""
        st.header("📋 Political Donation Legislation & Requirements")
        st.markdown("Query legislation and compliance requirements for political donations.")
        
        # Check if legislation search is properly initialized
        if not self.legislation_search.collection:
            st.error("❌ Legislation database not initialized. Please check if the document file exists.")
            return
        
        query = st.text_input(
            "Ask about legislation or requirements:",
            placeholder="e.g., What are the reporting requirements for political donations?",
            key="legislation_search_input"
        )
        
        if st.button("📋 Search Legislation", type="primary", key="legislation_search_btn"):
            if query:
                with st.spinner("Searching legislation..."):
                    context_docs = self.legislation_search.search_legislation(query)
                
                if context_docs:
                    st.info(f"Found {len(context_docs)} relevant document sections")
                    
                    with st.spinner("Generating response..."):
                        response = self.legislation_search.generate_response(query, context_docs)
                    
                    st.markdown("### 📜 Legislative Information:")
                    st.markdown(response)
                    
                    # Log the query
                    self.logger.log_query(
                        st.session_state.username,
                        "legislation_search",
                        query,
                        response,
                        {"context_chunks": len(context_docs)}
                    )
                    
                    # Show context
                    with st.expander("📄 Referenced Documents"):
                        for i, doc in enumerate(context_docs[:3], 1):
                            st.markdown(f"**Context {i}:**")
                            st.text(doc[:300] + "..." if len(doc) > 300 else doc)
                            st.divider()
                else:
                    st.warning("No relevant legislation found for this query.")
    
    def render_database_tab(self):
        """Render database search interface."""
        st.header("💾 AEC Declared Donations Database")
        st.markdown("Query the database of declared political donations.")
        
        # Show database info
        db_info = self.database_search.get_database_info()
        if "error" not in db_info:
            st.success(f"✅ Connected to database with {db_info['total_tables']} tables")
            with st.expander("📊 Available Tables"):
                for table in db_info['tables']:
                    st.write(f"- {table}")
        else:
            st.error(f"❌ Database error: {db_info['error']}")
            return
        
        query = st.text_input(
            "Ask about declared donations:",
            placeholder="e.g., Who donated the most to political parties in 2023?",
            key="database_search_input"
        )
        
        if st.button("💾 Query Database", type="primary", key="database_search_btn"):
            if query:
                with st.spinner("Querying database..."):
                    response = self.database_search.query_database(query)
                
                st.markdown("### 📊 Database Results:")
                st.markdown(response)
                
                # Log the query
                self.logger.log_query(
                    st.session_state.username,
                    "database_search",
                    query,
                    response
                )
    
    def render_integrated_analysis_tab(self):
        """Render integrated analysis interface."""
        st.header("📊 Integrated Cross-Reference Analysis")
        st.markdown("Combine insights from web articles, legislation, and declared donations.")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select analysis type:",
            [
                "Donor Compliance Check",
                "News vs Declared Donations",
                "Legislative Compliance Analysis",
                "Custom Cross-Reference"
            ],
            key="analysis_type_select"
        )
        
        if analysis_type == "Donor Compliance Check":
            st.markdown("**Check if a donor appears in news articles and declared donations**")
            donor_name = st.text_input("Enter donor name:", key="donor_name_input")
            
            if st.button("🔍 Analyze Donor", type="primary", key="analyze_donor_btn") and donor_name:
                self.perform_donor_analysis(donor_name)
        
        elif analysis_type == "News vs Declared Donations":
            st.markdown("**Compare news coverage with actual declared donations**")
            topic = st.text_input("Enter topic (e.g., mining industry donations):", key="topic_input")
            
            if st.button("📊 Compare Sources", type="primary", key="compare_sources_btn") and topic:
                self.perform_news_vs_data_analysis(topic)
        
        elif analysis_type == "Legislative Compliance Analysis":
            st.markdown("**Check compliance with donation reporting requirements**")
            entity = st.text_input("Enter entity name:", key="entity_input")
            
            if st.button("📋 Check Compliance", type="primary", key="check_compliance_btn") and entity:
                self.perform_compliance_analysis(entity)
        
        else:  # Custom Cross-Reference
            st.markdown("**Custom analysis across all three sources**")
            custom_query = st.text_area("Describe your analysis request:", key="custom_query_input")
            
            if st.button("🔬 Perform Analysis", type="primary", key="perform_analysis_btn") and custom_query:
                self.perform_custom_analysis(custom_query)
    
    def perform_donor_analysis(self, donor_name: str):
        """Perform comprehensive donor analysis."""
        st.markdown(f"### 🔍 Analysis Results for: {donor_name}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📰 News Coverage:**")
            with st.spinner("Searching news..."):
                sources, search_results = self.web_search.search_articles(f"{donor_name} political donations")
                news_response = self.web_search.generate_response(f"What news coverage exists about {donor_name} and political donations?", sources, search_results)
            st.markdown(news_response[:300] + "..." if len(news_response) > 300 else news_response)
        
        with col2:
            st.markdown("**📋 Legal Requirements:**")
            with st.spinner("Checking legislation..."):
                context_docs = self.legislation_search.search_legislation(f"donation reporting requirements for {donor_name}")
                legal_response = self.legislation_search.generate_response(f"What are the legal reporting requirements that would apply to {donor_name}?", context_docs)
            st.markdown(legal_response[:300] + "..." if len(legal_response) > 300 else legal_response)
        
        with col3:
            st.markdown("**💾 Declared Donations:**")
            with st.spinner("Querying database..."):
                db_response = self.database_search.query_database(f"Show me any donations by {donor_name} or entities related to {donor_name}")
            st.markdown(db_response[:300] + "..." if len(db_response) > 300 else db_response)
        
        # Combined analysis
        st.markdown("---")
        st.markdown("### 🎯 Cross-Reference Summary:")
        
        combined_analysis = f"""
        **Donor Analysis Summary for {donor_name}:**
        
        **News Coverage:** {news_response[:200]}...
        
        **Legal Context:** {legal_response[:200]}...
        
        **Declared Donations:** {db_response[:200]}...
        
        This analysis shows the relationship between public reporting, legal requirements, and actual declared donations.
        """
        
        st.markdown(combined_analysis)
        
        # Log integrated analysis
        self.logger.log_query(
            st.session_state.username,
            "integrated_analysis",
            f"Donor analysis: {donor_name}",
            combined_analysis,
            {"analysis_type": "donor_compliance_check"}
        )
    
    def perform_news_vs_data_analysis(self, topic: str):
        """Compare news coverage with database data."""
        st.markdown(f"### 📊 News vs Data Analysis for: {topic}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📰 News Coverage:**")
            sources, search_results = self.web_search.search_articles(f"{topic} political donations Australia")
            news_response = self.web_search.generate_response(f"What does the news say about {topic} and political donations?", sources, search_results)
            st.markdown(news_response)
        
        with col2:
            st.markdown("**💾 Actual Data:**")
            db_response = self.database_search.query_database(f"Show me donations related to {topic}")
            st.markdown(db_response)
        
        # Comparison analysis
        st.markdown("---")
        st.markdown("### ⚖️ Comparison Analysis:")
        comparison = f"""
        **Key Findings:**
        - News Coverage: {news_response[:150]}...
        - Database Records: {db_response[:150]}...
        
        **Discrepancies/Alignments:** [This would require more sophisticated NLP analysis to automatically identify discrepancies]
        """
        st.markdown(comparison)
    
    def perform_compliance_analysis(self, entity: str):
        """Analyze compliance with reporting requirements."""
        st.markdown(f"### 📋 Compliance Analysis for: {entity}")
        
        # Get legal requirements
        context_docs = self.legislation_search.search_legislation(f"reporting requirements compliance {entity}")
        legal_requirements = self.legislation_search.generate_response(f"What are the specific reporting requirements for {entity}?", context_docs)
        
        # Get actual filings
        db_response = self.database_search.query_database(f"Show me all filings and donations by {entity}")
        
        st.markdown("**📜 Legal Requirements:**")
        st.markdown(legal_requirements)
        
        st.markdown("**💾 Actual Filings:**")
        st.markdown(db_response)
        
        st.markdown("**⚖️ Compliance Assessment:**")
        compliance_analysis = f"""
        Based on the legal requirements and actual filings:
        
        **Requirements:** {legal_requirements[:200]}...
        **Filings:** {db_response[:200]}...
        
        **Compliance Status:** [Analysis would compare requirements vs actual filings]
        """
        st.markdown(compliance_analysis)
    
    def perform_custom_analysis(self, custom_query: str):
        """Perform custom integrated analysis."""
        st.markdown(f"### 🔬 Custom Analysis")
        st.markdown(f"**Query:** {custom_query}")
        
        # Search all three sources
        with st.spinner("Gathering information from all sources..."):
            # Web search
            sources, search_results = self.web_search.search_articles(custom_query)
            web_response = self.web_search.generate_response(custom_query, sources, search_results)
            
            # Legislation search
            context_docs = self.legislation_search.search_legislation(custom_query)
            legal_response = self.legislation_search.generate_response(custom_query, context_docs)
            
            # Database search
            db_response = self.database_search.query_database(custom_query)
        
        # Display results in tabs
        result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
            "📰 Web Results", "📋 Legal Context", "💾 Database Results", "🔬 Integrated Analysis"
        ])
        
        with result_tab1:
            st.markdown(web_response)
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"**[{source['title']}]({source['url']})**")
        
        with result_tab2:
            st.markdown(legal_response)
            if context_docs:
                with st.expander("Document Context"):
                    for i, doc in enumerate(context_docs[:3], 1):
                        st.markdown(f"**Context {i}:**")
                        st.text(doc[:200] + "...")
        
        with result_tab3:
            st.markdown(db_response)
        
        with result_tab4:
            integrated_response = f"""
            ## Integrated Analysis Results
            
            ### 📰 Web/News Perspective:
            {web_response[:300]}...
            
            ### 📋 Legal/Regulatory Context:
            {legal_response[:300]}...
            
            ### 💾 Official Data:
            {db_response[:300]}...
            
            ### 🎯 Key Insights:
            - **Transparency:** How well do news reports align with official data?
            - **Compliance:** Are legal requirements being met based on filed data?
            - **Gaps:** What information is missing or inconsistent across sources?
            
            ### 📊 Recommendations:
            Based on this cross-analysis, consider:
            1. Verifying specific claims against official data
            2. Checking compliance with current reporting requirements
            3. Investigating any discrepancies between public reporting and filed data
            """
            
            st.markdown(integrated_response)
            
            # Log the integrated analysis
            self.logger.log_query(
                st.session_state.username,
                "custom_integrated_analysis",
                custom_query,
                integrated_response,
                {
                    "web_sources": len(sources),
                    "legal_contexts": len(context_docs),
                    "analysis_type": "custom"
                }
            )
    
    def render_sidebar(self):
        """Render sidebar with user info and controls."""
        with st.sidebar:
            st.header(f"👋 {st.session_state.username}")
            
            # System status
            st.markdown("### 🔧 System Status")
            
            # Check component status
            web_status = "✅" if 'tavily_key' in st.secrets else "❌"
            ai_status = "✅" if 'api_key' in st.secrets else "❌"
            db_status = "✅" if Path(self.config.DATABASE_PATH).exists() else "❌"
            doc_status = "✅" if Path(self.config.DOC_FILE).exists() else "❌"
            replicate_status = "✅" if 'REPLICATE_API_TOKEN' in st.secrets else "⚠️ (using OpenAI)"
            
            # Check if database agent is working
            if self.database_search.agent_executor:
                db_agent_status = "✅"
            else:
                db_agent_status = "❌"
            
            st.markdown(f"""
            - **Web Search:** {web_status}
            - **AI Models:** {ai_status}
            - **AEC Database:** {db_status}
            - **Database Agent:** {db_agent_status}
            - **Legal Documents:** {doc_status}
            - **Replicate (Legislation):** {replicate_status}
            """)
            
            st.markdown("### 📊 Usage Statistics")
            
            # Show recent analysis history
            if st.session_state.analysis_history:
                recent_count = len([h for h in st.session_state.analysis_history if 
                                 (datetime.now() - datetime.fromisoformat(h['timestamp'])).days < 1])
                st.metric("Today's Queries", recent_count)
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            
            if st.button("🔄 Refresh Components"):
                # Force refresh of cached components
                st.cache_resource.clear()
                st.success("Components refreshed!")
            
            if st.button("📋 View Query Log"):
                try:
                    with open(self.config.QUERY_LOG_FILE, 'r') as f:
                        logs = json.load(f)
                    
                    user_logs = [log for log in logs if log.get('username') == st.session_state.username]
                    
                    if user_logs:
                        st.markdown("### Recent Queries:")
                        for log in user_logs[-5:]:
                            st.markdown(f"**{log['component']}:** {log['query'][:50]}...")
                    else:
                        st.info("No queries logged yet.")
                except:
                    st.error("Could not load query log.")
            
            if st.button("🚪 Switch User"):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.rerun()
            
            # Help section
            with st.expander("💡 Usage Tips"):
                st.markdown("""
                **Web Search:**
                - Search for recent news about political donors
                - Use specific names or organizations
                - Include terms like "political donations" or "campaign contributions"
                
                **Legislation:**
                - Ask about reporting requirements
                - Query compliance obligations
                - Search for specific legal provisions
                
                **Database:**
                - Use natural language queries
                - Ask about specific donors, amounts, or time periods
                - Compare data across years or parties
                
                **Integrated Analysis:**
                - Cross-reference information across all sources
                - Check for compliance and transparency issues
                - Identify discrepancies between public reports and official data
                """)

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Political Donation Analysis System",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #ff6b6b;
    }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        # Initialize the integrated system
        if 'app' not in st.session_state:
            with st.spinner("Initializing integrated analysis system..."):
                st.session_state.app = IntegratedDonationAnalyzer()
        
        app = st.session_state.app
        
        # Authentication check
        if not st.session_state.authenticated:
            app.render_authentication()
        else:
            # Render main interface
            app.render_sidebar()
            app.render_main_interface()
            
            # Footer
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; color: #666;'>
                <small>
                🏛️ Australian Political Donation Analysis System | 
                Integrating Web Search, Legislation & AEC Database | 
                Built with Streamlit, OpenAI, Replicate & LangChain
                </small>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)
        
        # Show troubleshooting info
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Common Issues:**
            1. **Missing API Keys:** Ensure you have configured in Streamlit secrets:
               - `api_key` (OpenAI)
               - `tavily_key` (Tavily Search)
               - `REPLICATE_API_TOKEN` (optional, for Replicate)
            
            2. **Missing Files:**
               - `my_aec.db` (AEC database)
               - `24146b01_Electoral_Reform.txt` (Legal documents)
            
            3. **Database Issues:**
               - Check if ChromaDB can write to `chroma_data/` directory
               - Ensure SQLite database is not corrupted
            
            4. **Dependencies:**
               - Install required packages: streamlit, openai, replicate, chromadb, langchain
            """)

if __name__ == "__main__":
    main()
