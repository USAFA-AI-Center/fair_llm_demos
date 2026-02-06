import streamlit as st
import sys
import os
from pathlib import Path

# Add the fair_llm directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Configure Streamlit page
st.set_page_config(
    page_title="WolframGPT Demo",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .wolfram-result {
        background-color: #fff3e0;
        border-left-color: #ff9800;
        font-family: monospace;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# Header
st.markdown('<h1 class="main-header">🧮 WolframGPT Demo</h1>', unsafe_allow_html=True)
st.markdown("**Chat with an AI that can use Wolfram|Alpha for mathematical computations!**")

# Sidebar
with st.sidebar:
    st.header("📋 About")
    st.markdown("""
    This demo uses **Ollama's gpt-oss-20b** model with advanced interpretation:
    
    🔍 **Smart Input Analysis**: Interprets your query to determine the best approach
    🧮 **Wolfram|Alpha Integration**: Automatically queries Wolfram for math problems
    📊 **Result Interpretation**: Explains Wolfram results in user-friendly terms
    💬 **General Chat**: Handles non-mathematical questions directly
    🚀 **Local AI**: Ollama provides fast, local AI inference
    
    **Example queries:**
    - "What is the integral of x^2?"
    - "Solve 2x + 5 = 13"
    - "Convert 100 miles to kilometers"
    - "What is the derivative of sin(x)?"
    - "How are you today?"
    """)
    
    st.header("🔧 Settings")
    max_tokens = st.slider("Max Response Length", 100, 1000, 512)
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Model loading section
if not st.session_state.model_loaded:
    st.info("🔄 **Ollama Connection Required**")
    st.markdown("""
    This demo requires Ollama to be running with the gpt-oss-20b model available.
    
    **To get started:**
    1. Make sure Ollama is installed and running
    2. Pull the gpt-oss-20b model: `ollama pull gpt-oss-20b`
    3. Click the "Test Connection" button below
    4. Once connected, you can start chatting with local AI!
    
    **Features after connection:**
    - Fast local AI inference
    - Smart query analysis and interpretation
    - Automatic Wolfram|Alpha integration
    - User-friendly result explanations
    """)
    
    if st.button("🚀 Test Connection", type="primary"):
        with st.spinner("🔄 Testing Ollama connection... This may take a moment."):
            try:
                # Import and test the model
                from fairlib.utils.WolframGPT import chat_with_tools
                
                # Test with a simple query
                test_response = chat_with_tools("Hello, are you working?")
                if "Error:" in test_response:
                    st.error(f"❌ Connection failed: {test_response}")
                else:
                    st.session_state.model_loaded = True
                    st.success("✅ Ollama connection successful!")
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Failed to connect to Ollama: {e}")
                st.info("💡 Make sure Ollama is running and the gpt-oss-20b model is available.")
                
                # Show detailed error information
                with st.expander("🔍 Error Details"):
                    st.code(str(e))
    
    # Show system requirements
    with st.expander("💻 System Requirements"):
        st.markdown("""
        **Minimum Requirements:**
        - 8GB+ RAM
        - 20GB+ free disk space
        - Python 3.8+
        - Ollama installed and running
        
        **Recommended:**
        - 16GB+ RAM
        - GPU with 8GB+ VRAM
        - Fast internet connection (for model download)
        
        **Setup Instructions:**
        ```bash
        # Install Ollama
        curl -fsSL https://ollama.ai/install.sh | sh
        
        # Pull the model
        ollama pull gpt-oss-20b
        
        # Install Python dependencies
        pip install streamlit python-dotenv requests
        ```
        """)
    
    st.stop()

# Chat interface (only shown after model is loaded)
st.header("💬 Chat with WolframGPT")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show Wolfram result and LaTeX expressions if available
        if message["role"] == "assistant" and "wolfram_result" in message:
            with st.expander("🧮 Mathematical Solution"):
                # Display final answer prominently
                if message.get('final_answer'):
                    st.markdown("**Final Answer:**")
                    st.latex(message['final_answer'])
                
                # Display step-by-step LaTeX solution
                if message.get('latex_steps'):
                    st.markdown("**Step-by-Step Solution:**")
                    for i, latex_step in enumerate(message['latex_steps'], 1):
                        if latex_step.strip():
                            st.markdown(f"**Step {i}:**")
                            st.latex(latex_step)
                
                # Display all LaTeX expressions
                if "latex_expressions" in message and message["latex_expressions"]:
                    st.markdown("**Mathematical Expressions:**")
                    for latex_expr in message["latex_expressions"]:
                        if latex_expr.strip():
                            st.latex(latex_expr)

# Chat input
if prompt := st.chat_input("Ask me anything! Try a math problem..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your query..."):
            try:
                from fairlib.utils.WolframGPT import chat_with_tools, interpret_user_input
                
                # Generate response
                result = chat_with_tools(prompt)
                
                # Handle both old string format and new dict format
                if isinstance(result, dict):
                    # Display main response
                    st.markdown(result['response'])
                    
                    # Show interpretation process
                    with st.expander("🔍 Query Analysis", expanded=False):
                        st.json(result['interpretation'])
                    
                    # Show Wolfram details if applicable
                    if result['requires_wolfram']:
                        # Display LaTeX solution prominently
                        st.markdown("### 🧮 Mathematical Solution")
                        
                        if result.get('final_answer'):
                            st.markdown("**Final Answer:**")
                            st.latex(result['final_answer'])
                        
                        if result.get('latex_steps'):
                            st.markdown("**Step-by-Step Solution:**")
                            for i, latex_step in enumerate(result['latex_steps'], 1):
                                if latex_step.strip():
                                    st.markdown(f"**Step {i}:**")
                                    st.latex(latex_step)
                        
                        if result['latex_expressions']:
                            st.markdown("**Mathematical Expressions:**")
                            for latex_expr in result['latex_expressions']:
                                if latex_expr.strip():
                                    st.latex(latex_expr)
                        
                        # Show technical details in a collapsed expander
                        with st.expander("🔍 Technical Details", expanded=False):
                            st.write(f"**Query Type:** {result['query_type']}")
                            st.write(f"**Confidence:** {result['confidence']:.2f}")
                            st.write(f"**Wolfram Query:** {result['wolfram_query']}")
                            st.write(f"**Wolfram Result:** {result['wolfram_result']}")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result['response'],
                        "wolfram_result": result.get('wolfram_result'),
                        "final_answer": result.get('final_answer'),
                        "latex_steps": result.get('latex_steps', []),
                        "latex_expressions": result.get('latex_expressions', []),
                        "query_type": result.get('query_type'),
                        "confidence": result.get('confidence')
                    })
                else:
                    # Handle old string format for backward compatibility
                    st.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>Powered by <strong>Ollama gpt-oss-20b</strong> and <strong>Wolfram|Alpha</strong></p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)