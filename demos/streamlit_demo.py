import streamlit as st
import sys
import os
import re
from pathlib import Path

# Add the fair_llm directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from fairlib.utils.WolframGPT import chat_with_tools
except ImportError as e:
    st.error(f"Failed to import WolframGPT: {e}")
    st.stop()

def preprocess_latex_in_text(text: str) -> str:
    """
    Preprocess text to properly format LaTeX expressions for Streamlit display.
    """
    if not text:
        return text
    
    processed_text = text
    
    # First, clean up any existing malformed LaTeX delimiters
    # Remove standalone $\displaystyle and $ that are not properly paired
    processed_text = re.sub(r'\$\s*\\displaystyle\s*(?![^$]*\$)', '', processed_text)
    processed_text = re.sub(r'(?<!\$)\s*\$\s*(?![^$]*\$)', '', processed_text)
    
    # Handle specific patterns in order of specificity
    
    # Pattern 1: Handle adjacent mathematical terms like "xequals4", "2equals5", etc.
    def replace_adjacent_math(match):
        content = match.group(0)
        # Replace "equals" with "=" in mathematical contexts, preserving spaces
        content = content.replace('equals', ' = ')
        return f'${content}$'
    
    processed_text = re.sub(r'[a-zA-Z]+\d*equals\d+', replace_adjacent_math, processed_text)
    
    # Pattern 1b: Handle more complex concatenated math expressions
    def replace_concatenated_math(match):
        content = match.group(0)
        # Handle various concatenated patterns in order of specificity
        if 'equals' in content:
            content = re.sub(r'([a-zA-Z]+)equals(\d+)', r'\1 = \2', content)  # xequals4 -> x = 4
        elif 'plus' in content:
            content = re.sub(r'([a-zA-Z]+)plus(\d+)', r'\1 + \2', content)  # xplus4 -> x + 4
        elif 'minus' in content:
            content = re.sub(r'([a-zA-Z]+)minus(\d+)', r'\1 - \2', content)  # xminus4 -> x - 4
        elif 'times' in content:
            content = re.sub(r'([a-zA-Z]+)times(\d+)', r'\1 \\cdot \2', content)  # xtimes4 -> x \cdot 4
        elif 'dividedby' in content:
            content = re.sub(r'([a-zA-Z]+)dividedby(\d+)', r'\1 / \2', content)  # xdividedby4 -> x / 4
        return f'${content}$'
    
    # More comprehensive pattern for concatenated mathematical expressions
    processed_text = re.sub(r'[a-zA-Z]+(?:equals|plus|minus|times|dividedby)\d+', replace_concatenated_math, processed_text)
    
    # Pattern 2: Handle general "equals" in mathematical contexts
    def replace_equals_in_math(match):
        content = match.group(0)
        content = content.replace('equals', ' = ')
        return f'${content}$'
    
    processed_text = re.sub(r'[a-zA-Z0-9\s]+equals[a-zA-Z0-9\s]+', replace_equals_in_math, processed_text)
    
    # Pattern 3: Expressions in square brackets like [\boxed{...}] or [x=4]
    def replace_square_brackets(match):
        content = match.group(1)
        # Only wrap if it contains mathematical symbols
        if any(char in content for char in ['^', '_', '+', '-', '*', '/', '=', '<', '>', '{', '}', '\\']):
            return f'${content}$'
        return match.group(0)
    
    processed_text = re.sub(r'\[([^]]+)\]', replace_square_brackets, processed_text)
    
    # Pattern 4: Double parentheses like ((x - 3)(x^2 + 3x - 6) = 0)
    def replace_double_parentheses(match):
        content = match.group(1)
        return f'${content}$'
    
    processed_text = re.sub(r'\(\(([^)]*[+\-*/^=<>{}()]+[^)]*)\)\)', replace_double_parentheses, processed_text)
    
    # Pattern 5: Expressions in single parentheses like (x^3 + 18 = 15x)
    def replace_parentheses(match):
        content = match.group(1)
        if any(char in content for char in ['^', '_', '+', '-', '*', '/', '=', '<', '>', '{', '}']):
            return f'${content}$'
        return match.group(0)
    
    processed_text = re.sub(r'\(([^)]+)\)', replace_parentheses, processed_text)
    
    # Pattern 6: Standalone LaTeX commands like \frac{...}{...}
    def replace_latex_commands(match):
        command = match.group(0)
        return f'${command}$'
    
    processed_text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}(?:\{[^}]*\})*', replace_latex_commands, processed_text)
    
    # Clean up any double-wrapped expressions
    def clean_double_wrapped(match):
        content = match.group(1)
        return f'${content}$'
    
    processed_text = re.sub(r'\$\$\displaystyle\s*([^$]+)\$\$', clean_double_wrapped, processed_text)
    
    # Clean up nested $ expressions
    def clean_nested(match):
        content = match.group(1)
        return f'${content}$'
    
    processed_text = re.sub(r'\$\displaystyle\s*\$\displaystyle\s*([^$]+)\$\$', clean_nested, processed_text)
    
    return processed_text

def extract_and_display_latex(text: str):
    """
    Extract LaTeX expressions from text and display them properly.
    """
    if not text:
        return
    
    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        # Check if paragraph contains LaTeX expressions
        if any(char in paragraph for char in ['^', '_', '\\', '{', '}']):
            # Try to extract LaTeX expressions
            latex_expressions = []
            remaining_text = paragraph
            
            # Find LaTeX expressions in various formats
            patterns = [
                r'\\[a-zA-Z]+\{[^}]*\}(?:\{[^}]*\})*',  # \frac{}{}, \sqrt{}, etc.
                r'[a-zA-Z]+\^[0-9]+',  # x^2, y^3, etc.
                r'[a-zA-Z]+_[0-9]+',   # x_1, y_2, etc.
                r'[^$]*[+\-*/^=<>{}()]+[^$]*',  # Mathematical expressions
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, remaining_text)
                for match in matches:
                    if match.strip() and len(match.strip()) > 1:
                        latex_expressions.append(match.strip())
                        remaining_text = remaining_text.replace(match, '', 1)
            
            # Display the paragraph with LaTeX expressions
            if latex_expressions:
                # Display any remaining text
                if remaining_text.strip():
                    st.markdown(remaining_text.strip())
                
                # Display LaTeX expressions
                for expr in latex_expressions:
                    if expr.strip():
                        try:
                            st.latex(expr.strip())
                        except:
                            # If LaTeX rendering fails, display as code
                            st.code(expr.strip())
            else:
                # No LaTeX expressions found, display as regular markdown
                st.markdown(paragraph)
        else:
            # No LaTeX expressions, display as regular markdown
            st.markdown(paragraph)

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
    This demo uses the **GPT-OSS-120B** model to:
    
    - Answer general questions
    - Perform mathematical computations using Wolfram|Alpha
    - Solve equations, integrals, derivatives
    - Convert units and more!
    
    **Example queries:**
    - "What is the integral of x^2?"
    - "Solve 2x + 5 = 13"
    - "Convert 100 miles to kilometers"
    - "What is the derivative of sin(x)?"
    """)
    
    st.header("🔧 Settings")
    max_tokens = st.slider("Max Response Length", 100, 1000, 512)
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Model loading status
if not st.session_state.model_loaded:
    with st.spinner("🔄 Loading GPT-OSS-120B model... This may take a moment on first run."):
        try:
            # Test if the model can be imported and used
            test_response = chat_with_tools("Hello, are you working?")
            st.session_state.model_loaded = True
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.info("💡 Make sure you have the required dependencies installed and sufficient system resources.")
            st.stop()

# Chat interface
st.header("💬 Chat with WolframGPT")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Preprocess LaTeX expressions in chat history
        if message["role"] == "assistant":
            processed_content = preprocess_latex_in_text(message["content"])
            st.markdown(processed_content)
        else:
            st.markdown(message["content"])
        
        # Show LaTeX expressions if available
        if message["role"] == "assistant" and "latex_expressions" in message:
            latex_expressions = message["latex_expressions"]
            if latex_expressions:
                for expr in latex_expressions:
                    if expr and expr.strip():
                        # Preprocess the expression to ensure proper LaTeX formatting
                        processed_expr = preprocess_latex_in_text(expr)
                        st.markdown(processed_expr)
        
        # Show structured extracted data if available
        if message["role"] == "assistant" and "extracted_data" in message:
            extracted_data = message["extracted_data"]
            if extracted_data and not extracted_data.get('fallback', False):
                with st.expander("🔍 Structured Solution Details"):
                    if 'problem_type' in extracted_data:
                        st.markdown(f"**Problem Type:** {extracted_data['problem_type']}")
                    if 'solution_method' in extracted_data:
                        st.markdown(f"**Solution Method:** {extracted_data['solution_method']}")
                    
                    if 'steps' in extracted_data and isinstance(extracted_data['steps'], list):
                        st.markdown("**Solution Steps:**")
                        for i, step in enumerate(extracted_data['steps'], 1):
                            if isinstance(step, dict):
                                st.markdown(f"**Step {i}:** {step.get('description', '')}")
                                if 'explanation' in step:
                                    st.markdown(f"*{step['explanation']}*")
                                if 'latex_expression' in step and step['latex_expression']:
                                    # Preprocess the LaTeX expression to ensure proper formatting
                                    processed_step_expr = preprocess_latex_in_text(step['latex_expression'])
                                    st.markdown(processed_step_expr)
                    
                    if 'verification' in extracted_data and extracted_data['verification']:
                        st.markdown(f"**Verification:** {extracted_data['verification']}")
                    
                    if 'alternative_methods' in extracted_data and extracted_data['alternative_methods']:
                        st.markdown(f"**Alternative Methods:** {', '.join(extracted_data['alternative_methods'])}")
        
        # Show Wolfram result if available
        if message["role"] == "assistant" and "wolfram_result" in message and message["wolfram_result"]:
            with st.expander("🔍 Wolfram|Alpha Result"):
                st.code(message["wolfram_result"])

# Chat input
if prompt := st.chat_input("Ask me anything! Try a math problem..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                response_dict = chat_with_tools(prompt)
                
                # Format the response properly for display
                if isinstance(response_dict, dict):
                    # Extract the main response content
                    main_response = response_dict.get('response', '')
                    
                    # Display the main response with LaTeX rendering
                    if main_response:
                        # Preprocess the response to properly format LaTeX expressions
                        processed_response = preprocess_latex_in_text(main_response)
                        st.markdown(processed_response)
                    
                    latex_expressions = response_dict.get('latex_expressions', [])
                    if latex_expressions:
                        st.markdown("**Mathematical Solution:**")
                        for expr in latex_expressions:
                            if expr and expr.strip():
                                processed_expr = preprocess_latex_in_text(expr)
                                st.markdown(processed_expr)
                    
                    # Show final answer if available and different from main response
                    final_answer = response_dict.get('final_answer', '')
                    if final_answer and final_answer != main_response and final_answer.strip():
                        st.markdown("**Final Answer:**")
                        # Preprocess the final answer to ensure proper LaTeX formatting
                        processed_final = preprocess_latex_in_text(final_answer)
                        st.markdown(processed_final)
                    
                    extracted_data = response_dict.get('extracted_data', {})
                    if extracted_data and not extracted_data.get('fallback', False):
                        with st.expander("🔍 Structured Solution Details"):
                            if 'problem_type' in extracted_data:
                                st.markdown(f"**Problem Type:** {extracted_data['problem_type']}")
                            if 'solution_method' in extracted_data:
                                st.markdown(f"**Solution Method:** {extracted_data['solution_method']}")
                            
                            if 'steps' in extracted_data and isinstance(extracted_data['steps'], list):
                                st.markdown("**Solution Steps:**")
                                for i, step in enumerate(extracted_data['steps'], 1):
                                    if isinstance(step, dict):
                                        st.markdown(f"**Step {i}:** {step.get('description', '')}")
                                        if 'explanation' in step:
                                            st.markdown(f"*{step['explanation']}*")
                                        if 'latex_expression' in step and step['latex_expression']:
                                            processed_step_expr = preprocess_latex_in_text(step['latex_expression'])
                                            st.markdown(processed_step_expr)
                            
                            if 'verification' in extracted_data and extracted_data['verification']:
                                st.markdown(f"**Verification:** {extracted_data['verification']}")
                            
                            if 'alternative_methods' in extracted_data and extracted_data['alternative_methods']:
                                st.markdown(f"**Alternative Methods:** {', '.join(extracted_data['alternative_methods'])}")
                    
                    # Store the formatted response for chat history
                    formatted_response = main_response
                    if latex_expressions:
                        formatted_response += "\n\n**Mathematical Solution:**\n"
                        for expr in latex_expressions:
                            if expr and expr.strip():
                                formatted_response += f"$$\n{expr}\n$$\n\n"
                    
                    if final_answer and final_answer != main_response and final_answer.strip():
                        formatted_response += f"**Final Answer:**\n$$\n{final_answer}\n$$"
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": formatted_response,
                        "wolfram_result": response_dict.get('wolfram_result', ''),
                        "latex_expressions": latex_expressions,
                        "extracted_data": extracted_data
                    })
                else:
                    # Fallback for non-dict responses
                    st.markdown(str(response_dict))
                    st.session_state.messages.append({"role": "assistant", "content": str(response_dict)})
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>Powered by <strong>GPT-OSS-120B</strong> and <strong>Wolfram|Alpha</strong></p>
    <p>Built with ❤️ using Streamlit</p>
</div>