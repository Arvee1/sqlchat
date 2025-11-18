"""
Standalone Cross-Analysis System for Australian Political Donations
=================================================================

This module provides cross-reference analysis capabilities that combine:
1. Web search results (news articles about donors)
2. Legislative/regulatory requirements 
3. Official AEC database records

Run this as a standalone Streamlit app for integrated donation analysis.
"""

import streamlit as st
import requests
import openai
import replicate
import sqlite3 as sql
import pandas as pd
import json
import re
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import sys

# ChromaDB imports
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain imports for SQL
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class CrossAnalysisConfig:
    """Configuration for cross-analysis system."""
    # File paths
    DATABASE_PATH = "my_aec.db"
    LEGISLATION_DOC = "24146b01_Electoral_Reform.txt"
    CHROMA_DATA_PATH = "chroma_data/"
    ANALYSIS_LOG = "cross_analysis_log.json"
    
    # API settings
    REQUEST_TIMEOUT = 15
    MAX_SEARCH_RESULTS = 5
    
    # Document processing
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    CONTEXT_RESULTS = 6
    
    # LLM settings
    TEMPERATURE = 0.3
    MAX_TOKENS = 1000

class CrossAnalysisLogger:
    """Logger for cross-analysis operations."""
    
    def __init__(self, config: CrossAnalysisConfig):
        self.config = config
        self.log_file = Path(config.ANALYSIS_LOG)
        self._ensure_log_exists()
    
    def _ensure_log_exists(self):
        if not self.log_file.exists():
            with open(self.log_file, 'w') as f:
                json.dump([], f)
    
    def log_analysis(self, analysis_type: str, query: str, results: dict):
        """Log cross-analysis results."""
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "analysis_type": analysis_type,
                "query": query,
                "results": results,
                "session_id": st.session_state.get('session_id', 'unknown')
            }
            
            logs.append(log_entry)
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Logging error: {e}")

class WebSearchAnalyzer:
    """Web search component for cross-analysis."""
    
    def __init__(self, config: CrossAnalysisConfig):
        self.config = config
    
    def search_donor_news(self, donor_name: str) -> Tuple[List[Dict], str]:
        """Search for news about a specific donor."""
        if 'tavily_key' not in st.secrets:
            return [], "Tavily API key not configured"
        
        try:
            url = "https://api.tavily.com/search"
            headers = {
                "Authorization": f"Bearer {st.secrets['tavily_key']}",
                "Content-Type": "application/json"
            }
            
            # Enhanced search query for Australian political donations
            search_query = f'"{donor_name}" political donations Australia OR campaign contributions OR electoral funding'
            
            data = {
                "query": search_query,
                "search_depth": "advanced",
                "include_domains": ["abc.net.au", "theguardian.com", "smh.com.au", "theage.com.au", "crikey.com.au"]
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            results = response.json()
            
            sources = []
            summary_text = ""
            
            if results and 'results' in results:
                for i, item in enumerate(results['results'][:self.config.MAX_SEARCH_RESULTS], 1):
                    source = {
                        "id": i,
                        "title": item.get("title", f"Source {i}"),
                        "url": item.get("url", ""),
                        "content": item.get("content", "")[:500],
                        "published_date": item.get("published_date", ""),
                        "relevance_score": item.get("score", 0)
                    }
                    sources.append(source)
                    
                    summary_text += f"\nSource {i}: {source['title']}\n{source['content']}\n---\n"
            
            return sources, summary_text
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return [], f"Search failed: {str(e)}"
    
    def analyze_web_results(self, donor_name: str, sources: List[Dict], summary_text: str) -> str:
        """Generate AI analysis of web search results."""
        if not sources or 'api_key' not in st.secrets:
            return "No web analysis available"
        
        try:
            client = openai.Client(api_key=st.secrets["api_key"])
            
            prompt = f"""
            Analyze the following news coverage about {donor_name} and political donations in Australia.
            
            Focus on:
            1. Donation amounts mentioned
            2. Political parties or candidates involved
            3. Time periods of donations
            4. Any compliance or transparency issues mentioned
            5. Public reaction or controversy
            
            News sources:
            {summary_text}
            
            Provide a concise analysis highlighting key facts and any concerning patterns.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert analyst of Australian political donation transparency and compliance issues."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Web analysis error: {e}")
            return f"Analysis error: {str(e)}"

class LegislationAnalyzer:
    """Legislation/regulatory analysis component."""
    
    def __init__(self, config: CrossAnalysisConfig):
        self.config = config
        self.collection = None
        self.use_replicate = False
        self._initialize_vector_db()
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB for legislation documents."""
        try:
            # Check for Replicate API token
            if 'REPLICATE_API_TOKEN' in st.secrets:
                import os
                os.environ['REPLICATE_API_TOKEN'] = st.secrets['REPLICATE_API_TOKEN']
                self.use_replicate = True
                logger.info("Replicate API configured for legislation analysis")
            
            client = chromadb.PersistentClient(path=self.config.CHROMA_DATA_PATH)
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            self.collection = client.get_or_create_collection(
                name="legislation_analysis",
                embedding_function=embedding_func
            )
            
            # Load and process legislation document if collection is empty
            if self.collection.count() == 0:
                self._load_legislation_document()
                
        except Exception as e:
            logger.error(f"Vector DB initialization error: {e}", exc_info=True)
            self.collection = None
    
    def _load_legislation_document(self):
        """Load and chunk legislation document."""
        doc_path = Path(self.config.LEGISLATION_DOC)
        if not doc_path.exists():
            logger.warning(f"Legislation document not found: {doc_path}")
            return
        
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP
            )
            
            chunks = splitter.split_text(content)
            
            if chunks:
                ids = [f"leg_chunk_{i}" for i in range(len(chunks))]
                self.collection.add(documents=chunks, ids=ids)
                logger.info(f"Loaded {len(chunks)} legislation chunks")
            
        except Exception as e:
            logger.error(f"Document loading error: {e}")
    
    def analyze_compliance_requirements(self, donor_name: str, context: str = "") -> str:
        """Analyze what legal requirements apply to this donor."""
        if not self.collection:
            return "Legislation analysis not available"
        
        try:
            # Search for relevant legal requirements
            query = f"reporting requirements disclosure obligations {donor_name} political donations compliance"
            
            results = self.collection.query(
                query_texts=[query],
                include=["documents"],
                n_results=self.config.CONTEXT_RESULTS
            )
            
            context_docs = results.get("documents", [[]])[0]
            
            if not context_docs:
                return "No relevant legislation found"
            
            # Generate analysis
            combined_context = "\n\n---\n\n".join(context_docs)
            
            prompt = f"""
            Based on Australian political donation legislation, analyze the legal requirements for {donor_name}.
            
            Context from legislation:
            {combined_context}
            
            Additional context: {context}
            
            Provide analysis covering:
            1. Disclosure thresholds that apply
            2. Reporting deadlines and requirements
            3. Penalties for non-compliance
            4. Specific obligations for this type of donor
            """
            
            # Try Replicate first if available, otherwise use OpenAI
            if self.use_replicate:
                try:
                    result = ""
                    for event in replicate.stream(
                        "meta/meta-llama-3-70b-instruct",
                        input={
                            "prompt": prompt,
                            "max_tokens": self.config.MAX_TOKENS,
                            "temperature": self.config.TEMPERATURE,
                            "top_p": 0.8
                        }
                    ):
                        result += str(event)
                    
                    if result.strip():
                        return result.strip()
                except Exception as e:
                    logger.warning(f"Replicate failed, falling back to OpenAI: {e}")
            
            # Fallback to OpenAI
            if 'api_key' in st.secrets:
                client = openai.Client(api_key=st.secrets["api_key"])
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert on Australian political donation laws."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.TEMPERATURE,
                    max_tokens=self.config.MAX_TOKENS
                )
                return response.choices[0].message.content.strip()
            
            return "AI analysis not available (no API keys configured)"
            
        except Exception as e:
            logger.error(f"Legislation analysis error: {e}", exc_info=True)
            return f"Legal analysis error: {str(e)}"

class DatabaseAnalyzer:
    """AEC database analysis component."""
    
    def __init__(self, config: CrossAnalysisConfig):
        self.config = config
        self.db = None
        self.agent = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and SQL agent."""
        db_path = Path(self.config.DATABASE_PATH)
        if not db_path.exists():
            logger.warning(f"Database not found: {db_path}")
            return
