# streamlit_app.py - Run this file to start the Streamlit interface
# Usage: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
import traceback
from typing import Dict, Any, Optional, List, Union

# You'll need to import your fixed DataAnalystAgent class
# Make sure the data_analyst_agent_fixed.py file is in the same directory
try:
    from agent import DataAnalystAgent
except ImportError:
    st.write("❌ Could not import DataAnalystAgent. Make sure 'agent.py' is in the same directory.")
    st.stop()

def safe_get_summary(agent: DataAnalystAgent) -> Dict[str, Any]:
    """Safely get data summary with error handling"""
    try:
        return agent.get_data_summary()
    except Exception as e:
        return {"error": f"Failed to get summary: {str(e)}"}

def safe_create_visualizations(agent: DataAnalystAgent) -> List[Dict[str, Any]]:
    """Safely create visualizations with error handling"""
    try:
        return agent.create_visualizations()
    except Exception as e:
        return [{"error": f"Failed to create visualizations: {str(e)}"}]

def safe_answer_question(agent: DataAnalystAgent, question: str, chat_history: list) -> Dict[str, Any]:
    """Safely answer questions with error handling"""
    try:
        return agent.answer_question(question, chat_history)
    except Exception as e:
        return {"error": f"Failed to answer question: {str(e)}"}

def display_dataframe_safely(df: Any, use_container_width: bool = True) -> None:
    """Safely display dataframes with error handling"""
    try:
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df, use_container_width=use_container_width)
        else:
            st.write("⚠️ No data to display")
    except Exception as e:
        st.write(f"❌ Error displaying data: {str(e)}")

def show_error(message: str) -> None:
    """Display error message"""
    st.markdown(f"<div style='color: red; font-weight: bold;'>❌ {message}</div>", unsafe_allow_html=True)

def show_success(message: str) -> None:
    """Display success message"""
    st.markdown(f"<div style='color: green; font-weight: bold;'>✅ {message}</div>", unsafe_allow_html=True)

def show_warning(message: str) -> None:
    """Display warning message"""
    st.markdown(f"<div style='color: orange; font-weight: bold;'>⚠️ {message}</div>", unsafe_allow_html=True)

def main() -> None:
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="🤖 AI Data Analyst",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title
    st.markdown('<h1 class="main-header">🤖 AI Data Analyst Agent</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by Llama 3.1 via Together.ai** - Upload any document and get intelligent analysis!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key: str = st.text_input(
            "Together.ai API Key", 
            type="password",
            help="Get your API key from https://api.together.xyz/"
        )
        
        if api_key:
            # Initialize agent
            if 'agent' not in st.session_state or st.session_state.get('api_key') != api_key:
                with st.spinner("Initializing AI agent..."):
                    try:
                        st.session_state.agent = DataAnalystAgent(api_key)
                        st.session_state.api_key = api_key
                        show_success("AI Agent initialized!")
                    except Exception as e:
                        show_error(f"Failed to initialize: {str(e)}")
                        st.stop()
            else:
                show_success("AI Agent ready!")
        else:
            show_warning("Please enter your Together.ai API key")
            st.write("**Get your API key:**")
            st.write("1. Visit [Together.ai](https://api.together.xyz/)")
            st.write("2. Sign up and generate API key")
            st.write("3. Paste it above")
            st.stop()
        
        # Model info
        st.markdown("---")
        st.write("**Model:** Llama-3.1-8B-Instruct-Turbo")
        
        # Session stats
        if 'agent' in st.session_state:
            st.markdown("---")
            st.markdown("**Session Stats:**")
            agent: DataAnalystAgent = st.session_state.agent
            questions_count: int = len(getattr(agent, 'analysis_history', []))
            st.metric("Questions Asked", questions_count)
    
    # Main content
    if 'agent' in st.session_state:
        agent: DataAnalystAgent = st.session_state.agent
        
        # File upload section
        st.header("📁 Upload Your Document")
        
        uploaded_file = st.file_uploader(
            "Choose a file to analyze",
            type=['csv', 'xlsx', 'xls', 'pdf', 'docx', 'doc', 'txt', 'jpg', 'jpeg', 'png', 'bmp'],
            help="Supported: CSV, Excel, PDF, Word, Text, Images"
        )
        
        if uploaded_file:
            # Show file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**File:** {uploaded_file.name}")
            with col2:
                st.info(f"**Size:** {uploaded_file.size:,} bytes")
            with col3:
                file_type: str = getattr(uploaded_file, 'type', 'Unknown')
                st.info(f"**Type:** {file_type}")
            
            # Process the file
            if st.button("🚀 Analyze Document", type="primary"):
                with st.spinner("Processing your document..."):
                    try:
                        # Create temporary file
                        file_extension: str = uploaded_file.name.split('.')[-1] if '.' in uploaded_file.name else 'tmp'
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path: str = tmp_file.name
                        
                        # Load document
                        result: Union[Dict[str, Any], str] = agent.load_document(tmp_file_path)
                        
                        # Clean up
                        try:
                            os.unlink(tmp_file_path)
                        except OSError:
                            pass  # File might already be deleted
                        
                        if isinstance(result, dict) and result.get("status") == "success":
                            show_success(result.get('message', 'Document loaded successfully'))
                            st.session_state.document_loaded = True
                        else:
                            error_msg: str = result.get('message', str(result)) if isinstance(result, dict) else str(result)
                            show_error(error_msg)
                            
                    except Exception as e:
                        show_error(f"Error processing file: {str(e)}")
                        if st.checkbox("Show debug info"):
                            st.code(traceback.format_exc())
        
        # Show content if document is loaded
        if getattr(st.session_state, 'document_loaded', False):
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Summary", "💬 Chat with Agent", "📊 Visualizations", "📈 History"])
            
            with tab1:
                st.header("Data Summary")
                
                try:
                    summary: Dict[str, Any] = safe_get_summary(agent)
                    
                    if "error" in summary:
                        show_error(f"Error getting summary: {summary['error']}")
                    else:
                        if isinstance(agent.data, pd.DataFrame):
                            # Structured data display
                            col1, col2, col3, col4 = st.columns(4)
                            
                            # Safely get shape information
                            shape: Union[tuple, list] = summary.get('shape', (0, 0))
                            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                                rows, cols = shape[0], shape[1]
                            else:
                                rows = len(agent.data)
                                cols = len(agent.data.columns) if hasattr(agent.data, 'columns') else 0
                            
                            with col1:
                                st.metric("📊 Rows", f"{rows:,}")
                            with col2:
                                st.metric("📈 Columns", cols)
                            with col3:
                                missing_values: Dict[str, Any] = summary.get('missing_values', {})
                                if isinstance(missing_values, dict):
                                    missing_count: int = sum(missing_values.values())
                                else:
                                    missing_count = 0
                                st.metric("❌ Missing", f"{missing_count:,}")
                            with col4:
                                dtypes: Dict[str, Any] = summary.get('dtypes', {})
                                if isinstance(dtypes, dict):
                                    numeric_count: int = len([k for k, v in dtypes.items() if 'int' in str(v) or 'float' in str(v)])
                                else:
                                    numeric_count = 0
                                st.metric("🔢 Numeric", numeric_count)
                            
                            # Sample data
                            st.subheader("Sample Data")
                            sample_data: Any = summary.get('sample_data')
                            if sample_data and isinstance(sample_data, list):
                                try:
                                    sample_df: pd.DataFrame = pd.DataFrame(sample_data)
                                    display_dataframe_safely(sample_df)
                                except Exception:
                                    display_dataframe_safely(agent.data.head())
                            else:
                                display_dataframe_safely(agent.data.head())
                            
                            # Column info
                            dtypes = summary.get('dtypes', {})
                            if dtypes and isinstance(dtypes, dict):
                                st.subheader("Column Information")
                                try:
                                    missing_values = summary.get('missing_values', {})
                                    col_df: pd.DataFrame = pd.DataFrame({
                                        'Column': list(dtypes.keys()),
                                        'Data Type': [str(v) for v in dtypes.values()],
                                        'Missing': [missing_values.get(col, 0) for col in dtypes.keys()]
                                    })
                                    display_dataframe_safely(col_df)
                                except Exception as e:
                                    show_error(f"Error displaying column information: {str(e)}")
                        
                        else:
                            # Text document display
                            st.subheader("Document Information")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📄 Type", summary.get('type', 'Unknown'))
                            with col2:
                                st.metric("📝 Characters", f"{summary.get('length', 0):,}")
                            with col3:
                                st.metric("📖 Words", f"{summary.get('word_count', 0):,}")
                            
                            st.subheader("Document Preview")
                            preview_text: str = summary.get('preview', '')
                            st.text_area("Content", preview_text, height=300, disabled=True)
                            
                except Exception as e:
                    show_error(f"Error displaying summary: {str(e)}")
                    if st.checkbox("Show error details"):
                        st.code(traceback.format_exc())
            
            with tab2:
                st.header("💬 Chat with Agent")

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Ask a question about your data"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing..."):
                            response = safe_answer_question(agent, prompt, st.session_state.messages)
                            if "error" in response:
                                st.error(response["error"])
                            else:
                                st.markdown(response.get('llm_answer', 'No answer generated'))
                    st.session_state.messages.append({"role": "assistant", "content": response.get('llm_answer', 'No answer generated')})
            
            with tab3:
                st.header("Data Visualizations")
                
                if isinstance(agent.data, pd.DataFrame):
                    if st.button("🎨 Generate Visualizations"):
                        with st.spinner("Creating visualizations..."):
                            vizs: List[Dict[str, Any]] = safe_create_visualizations(agent)
                            
                            if vizs:
                                for viz in vizs:
                                    if isinstance(viz, dict) and "error" not in viz:
                                        title: str = viz.get("title", "Visualization")
                                        st.subheader(title)
                                        
                                        figure: Any = viz.get("figure")
                                        if figure:
                                            try:
                                                st.plotly_chart(figure, use_container_width=True)
                                            except Exception as e:
                                                show_error(f"Error displaying chart: {str(e)}")
                                        
                                        insights: Any = viz.get("insights")
                                        if insights:
                                            with st.expander("🔍 AI Insights"):
                                                st.write(insights)
                                    else:
                                        error_msg: str = viz.get("error", "Unknown visualization error") if isinstance(viz, dict) else str(viz)
                                        show_error(error_msg)
                            else:
                                show_warning("No visualizations were generated")
                else:
                    st.info("📊 Visualizations are only available for structured data (CSV/Excel files)")
            
            with tab4:
                st.header("Analysis History")
                
                history: List[Any] = getattr(agent, 'analysis_history', [])
                if history and isinstance(history, list):
                    for i, item in enumerate(reversed(history)):
                        if isinstance(item, dict):
                            question: str = item.get('question', 'Unknown question')
                            answer: str = item.get('answer', 'No answer')
                            context: str = item.get('context', '')
                            
                            question_preview: str = question[:60] + "..." if len(question) > 60 else question
                            
                            with st.expander(f"Q{len(history)-i}: {question_preview}"):
                                st.write("**Question:**", question)
                                st.write("**Answer:**", answer)
                                if context:
                                    st.write("**Context:**", context)
                else:
                    st.info("📝 No analysis history yet. Start asking questions!")

if __name__ == "__main__":
    main()