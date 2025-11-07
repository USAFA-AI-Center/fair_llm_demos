#!/bin/bash

echo "🧮 Starting WolframGPT Streamlit Demo..."
echo "=================================="

# Check if we're in the right directory
if [ ! -f "streamlit_demo_simple.py" ]; then
    echo "❌ Error: streamlit_demo_simple.py not found"
    echo "Please run this script from the demos directory"
    exit 1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import torch, transformers, streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install torch transformers streamlit python-dotenv requests
fi

echo "✅ Dependencies ready!"
echo ""
echo "🚀 Starting Streamlit demo..."
echo "The demo will open in your browser at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the demo"
echo ""

# Run the streamlit demo
streamlit run streamlit_demo_simple.py

