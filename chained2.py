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
from langchain import hub
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
        self._initialize_vector_db()
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB for legislation documents."""
        try:
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
            logger.error(f"Vector DB initialization error: {e}")
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
            
            # Generate analysis using Replicate LLaMA
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
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Legislation analysis error: {e}")
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
        
        try:
            self.db = SQLDatabase.from_uri(f"sqlite:///{self.config.DATABASE_PATH}")
            
            if 'api_key' in st.secrets:
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    openai_api_key=st.secrets["api_key"],
                    temperature=0
                )
                
                toolkit = SQLDatabaseToolkit(db=self.db, llm=llm)
                tools = toolkit.get_tools()
                
                prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
                system_message = prompt_template.format(dialect="SQLite", top_k=10)
                
                self.agent = create_react_agent(llm, tools, prompt=system_message)
                
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def analyze_declared_donations(self, donor_name: str) -> Dict[str, Any]:
        """Analyze declared donations for a specific donor."""
        if not self.agent:
            return {"error": "Database agent not available"}
        
        try:
            # Query for direct matches
            direct_query = f"""
            Find all donations by '{donor_name}' or any entity containing '{donor_name}' in the name. 
            Include donor name, recipient, amount, date, and any other relevant details.
            Also search for similar names or related entities.
            """
            
            direct_result = self.agent.invoke(
                {"messages": [HumanMessage(content=direct_query)]}
            )
            
            # Query for patterns and totals
            pattern_query = f"""
            Analyze donation patterns for '{donor_name}':
            1. Total donations across all years
            2. Which political parties received donations
            3. Frequency and timing of donations
            4. Any unusual patterns or large amounts
            """
            
            pattern_result = self.agent.invoke(
                {"messages": [HumanMessage(content=pattern_query)]}
            )
            
            return {
                "direct_matches": direct_result["messages"][-1].content if direct_result else "No direct matches",
                "pattern_analysis": pattern_result["messages"][-1].content if pattern_result else "No pattern analysis",
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Database analysis error: {e}")
            return {"error": str(e)}

class CrossAnalysisEngine:
    """Main cross-analysis engine that combines all components."""
    
    def __init__(self):
        self.config = CrossAnalysisConfig()
        self.logger = CrossAnalysisLogger(self.config)
        
        # Initialize analyzers
        self.web_analyzer = WebSearchAnalyzer(self.config)
        self.legislation_analyzer = LegislationAnalyzer(self.config)
        self.database_analyzer = DatabaseAnalyzer(self.config)
    
    def analyze_donor_compliance(self, donor_name: str) -> Dict[str, Any]:
        """Comprehensive donor compliance analysis."""
        st.write(f"🔍 **Starting comprehensive analysis of: {donor_name}**")
        
        results = {
            "donor_name": donor_name,
            "timestamp": datetime.now().isoformat(),
            "web_analysis": {},
            "legal_analysis": "",
            "database_analysis": {},
            "cross_analysis": "",
            "compliance_score": "Unknown"
        }
        
        # 1. Web Search Analysis
        with st.spinner("🌐 Searching news articles..."):
            web_sources, web_summary = self.web_analyzer.search_donor_news(donor_name)
            web_analysis = self.web_analyzer.analyze_web_results(donor_name, web_sources, web_summary)
            
            results["web_analysis"] = {
                "sources_found": len(web_sources),
                "analysis": web_analysis,
                "sources": web_sources
            }
            
            st.success(f"Found {len(web_sources)} news articles")
        
        # 2. Legal Requirements Analysis
        with st.spinner("📋 Analyzing legal requirements..."):
            legal_context = f"Donor: {donor_name}. Web coverage suggests: {web_analysis[:200]}"
            legal_analysis = self.legislation_analyzer.analyze_compliance_requirements(donor_name, legal_context)
            
            results["legal_analysis"] = legal_analysis
            st.success("Legal analysis completed")
        
        # 3. Database Analysis
        with st.spinner("💾 Querying AEC database..."):
            db_results = self.database_analyzer.analyze_declared_donations(donor_name)
            
            results["database_analysis"] = db_results
            if "error" not in db_results:
                st.success("Database analysis completed")
            else:
                st.warning(f"Database issue: {db_results['error']}")
        
        # 4. Cross-Analysis
        with st.spinner("🔬 Performing cross-analysis..."):
            cross_analysis = self._generate_cross_analysis(results)
            results["cross_analysis"] = cross_analysis
            results["compliance_score"] = self._calculate_compliance_score(results)
            
            st.success("Cross-analysis completed")
        
        # Log the analysis
        self.logger.log_analysis("donor_compliance", donor_name, results)
        
        return results
    
    def _generate_cross_analysis(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive cross-analysis."""
        try:
            if 'api_key' not in st.secrets:
                return "Cross-analysis requires OpenAI API key"
            
            client = openai.Client(api_key=st.secrets["api_key"])
            
            prompt = f"""
            Perform a comprehensive cross-analysis of political donation compliance for {results['donor_name']}.
            
            WEB/NEWS ANALYSIS:
            {results['web_analysis'].get('analysis', 'No web analysis')}
            
            LEGAL REQUIREMENTS:
            {results['legal_analysis']}
            
            DATABASE RECORDS:
            {results['database_analysis'].get('direct_matches', 'No database matches')}
            
            PATTERN ANALYSIS:
            {results['database_analysis'].get('pattern_analysis', 'No pattern analysis')}
            
            Provide a comprehensive analysis addressing:
            
            1. **TRANSPARENCY ASSESSMENT**: How well do public reports align with official records?
            
            2. **COMPLIANCE STATUS**: Based on legal requirements, is this donor meeting their obligations?
            
            3. **RED FLAGS**: Any discrepancies, missing disclosures, or concerning patterns?
            
            4. **RECOMMENDATIONS**: What actions should be taken for better compliance/transparency?
            
            5. **CONFIDENCE LEVEL**: Rate your confidence in this analysis (High/Medium/Low) and explain why.
            
            Be specific about any gaps between news coverage, legal requirements, and official filings.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert compliance analyst specializing in Australian political donation transparency and regulatory compliance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Cross-analysis generation error: {e}")
            return f"Cross-analysis error: {str(e)}"
    
    def _calculate_compliance_score(self, results: Dict[str, Any]) -> str:
        """Calculate a simple compliance score based on available data."""
        try:
            score_factors = []
            
            # Factor 1: Database records found
            if "error" not in results["database_analysis"]:
                if "no donations found" not in results["database_analysis"].get("direct_matches", "").lower():
                    score_factors.append("Records Found")
                else:
                    score_factors.append("No Records")
            
            # Factor 2: News coverage
            if results["web_analysis"]["sources_found"] > 0:
                score_factors.append("Public Coverage")
            
            # Factor 3: Legal analysis completeness
            if len(results["legal_analysis"]) > 100:
                score_factors.append("Legal Context")
            
            if len(score_factors) >= 3:
                return "High Confidence Analysis"
            elif len(score_factors) >= 2:
                return "Medium Confidence Analysis"
            else:
                return "Limited Data Available"
                
        except Exception as e:
            return "Score Calculation Error"
    
    def analyze_sector_compliance(self, sector: str) -> Dict[str, Any]:
        """Analyze compliance patterns across a sector (e.g., mining, banking)."""
        st.write(f"🏢 **Analyzing sector compliance: {sector}**")
        
        # This would be similar to donor analysis but across multiple entities
        # Implementation would depend on having sector classification in your database
        
        return {
            "sector": sector,
            "message": "Sector analysis feature - implementation depends on data structure"
        }

# Streamlit Interface
def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Political Donation Cross-Analysis",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Wazzup!!! Test Political Donation Cross-Analysis System")
    st.markdown("**Comprehensive analysis combining news coverage, legal requirements, and Test data**")
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if 'analysis_engine' not in st.session_state:
        with st.spinner("Initializing analysis systems..."):
            st.session_state.analysis_engine = CrossAnalysisEngine()
    
    engine = st.session_state.analysis_engine
    
    # System status check
    with st.expander("🔧 System Status", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            web_status = "✅ Ready" if 'tavily_key' in st.secrets else "❌ No API Key"
            st.metric("Web Search", web_status)
        
        with col2:
            ai_status = "✅ Ready" if 'api_key' in st.secrets else "❌ No API Key"
            st.metric("AI Analysis", ai_status)
        
        with col3:
            db_status = "✅ Ready" if Path(engine.config.DATABASE_PATH).exists() else "❌ Not Found"
            st.metric("AEC Database", db_status)
        
        with col4:
            doc_status = "✅ Ready" if Path(engine.config.LEGISLATION_DOC).exists() else "❌ Not Found"
            st.metric("Legal Docs", doc_status)
    
    st.markdown("---")
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        [
            "Individual Donor Analysis",
            "Sector Compliance Analysis",
            "Custom Cross-Reference"
        ]
    )
    
    if analysis_type == "Individual Donor Analysis":
        st.markdown("### 👤 Individual Donor Compliance Analysis")
        st.markdown("Analyze a specific donor across news coverage, legal requirements, and test records.")
        
        donor_name = st.text_input(
            "Enter donor name (individual or organization):",
            placeholder="e.g., DJ Ryan, DJ Arvee."
        )
        
        if st.button("🔍 Analyze Donor", type="primary", disabled=not donor_name.strip()):
            if donor_name.strip():
                results = engine.analyze_donor_compliance(donor_name.strip())
                
                # Display results
                st.markdown("---")
                st.markdown(f"## 📊 Analysis Results for: **{results['donor_name']}**")
                
                # Create tabs for different aspects
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📰 News Coverage", 
                    "📋 Legal Requirements", 
                    "💾 Test Records", 
                    "🔬 Cross-Analysis"
                ])
                
                with tab1:
                    st.markdown("### News & Media Coverage")
                    if results["web_analysis"]["sources_found"] > 0:
                        st.success(f"Found {results['web_analysis']['sources_found']} relevant articles")
                        st.markdown(results["web_analysis"]["analysis"])
                        
                        with st.expander("📑 Source Articles"):
                            for source in results["web_analysis"]["sources"]:
                                st.markdown(f"**[{source['title']}]({source['url']})**")
                                st.write(source['content'])
                                st.markdown("---")
                    else:
                        st.info("No recent news coverage found")
                
                with tab2:
                    st.markdown("### Legal Requirements & Compliance")
                    st.markdown(results["legal_analysis"])
                
                with tab3:
                    st.markdown("### Official AEC Records")
                    if "error" not in results["database_analysis"]:
                        st.markdown("**Direct Matches:**")
                        st.text(results["database_analysis"]["direct_matches"])
                        
                        st.markdown("**Pattern Analysis:**")
                        st.text(results["database_analysis"]["pattern_analysis"])
                    else:
                        st.error(f"Database error: {results['database_analysis']['error']}")
                
                with tab4:
                    st.markdown("### Comprehensive Cross-Analysis")
                    st.info(f"**Confidence Level:** {results['compliance_score']}")
                    st.markdown(results["cross_analysis"])
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("News Articles", results["web_analysis"]["sources_found"])
                    with col2:
                        records_status = "Found" if "error" not in results["database_analysis"] else "Error"
                        st.metric("Database Status", records_status)
                    with col3:
                        legal_status = "Analyzed" if len(results["legal_analysis"]) > 50 else "Limited"
                        st.metric("Legal Analysis", legal_status)
    
    elif analysis_type == "Sector Compliance Analysis":
        st.markdown("### 🏢 Sector Compliance Analysis")
        st.info("This feature analyzes compliance patterns across an entire industry sector.")
        
        sector = st.selectbox(
            "Select sector:",
            ["Mining", "Banking & Finance", "Property Development", "Energy", "Retail", "Other"]
        )
        
        if sector == "Other":
            sector = st.text_input("Enter sector name:")
        
        if st.button("📊 Analyze Sector", disabled=True):
            st.info("Sector analysis feature coming soon - requires enhanced database indexing")
    
    else:  # Custom Cross-Reference
        st.markdown("### 🔬 Custom Cross-Reference Analysis")
        st.markdown("Perform custom analysis with specific parameters.")
        
        custom_query = st.text_area(
            "Describe your analysis request:",
            placeholder="e.g., Compare mining industry donations in news vs AEC records for 2023"
        )
        
        if st.button("🔬 Perform Custom Analysis"):
            st.info("Custom analysis feature - would require query parsing and custom search logic")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>
        🔍 Political Donation Cross-Analysis System | 
        Combines Web Search, Legal Analysis & AEC Database | 
        Session ID: {session_id}
        </small>
    </div>
    """.format(session_id=st.session_state.session_id), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
