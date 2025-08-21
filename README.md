# FAIR-LLM Demo Repository

This repository contains demonstration scripts for the FAIR-LLM (Flexible, Agnostic, and Interoperable Reasoning) Framework - a powerful system for building modular agentic AI applications.

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher (required by fair-llm)
- pip (Python package installer)
- Git
- GitHub account with access to the private FAIR-LLM repository
- API keys for OpenAI and Anthropic

### Installation

#### Step 1: Clone this Demo Repository

```bash
git clone https://github.com/yourusername/fair-llm-demos.git
cd fair-llm-demos
```

#### Step 2: Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### Step 3: Configure GitHub Access for Private Repository

Since the main `fair-llm` package is in a private repository, you need to authenticate with GitHub.

**Option A: Using Personal Access Token (Recommended)**

1. Generate a GitHub Personal Access Token:
   - Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Click "Personal access tokens" → "Tokens (classic)" → "Generate new token"
   - Give it a name (e.g., "fair-llm-access")
   - Select the `repo` scope (full control of private repositories)
   - Copy the token immediately (you won't see it again!)

2. Set the token as an environment variable:
   ```bash
   export GH_TOKEN=your_github_token_here
   ```

**Option B: Using SSH Key**

1. Ensure your SSH key is added to GitHub:
   ```bash
   ssh -T git@github.com  # Test SSH connection
   ```

#### Step 4: Install Dependencies

```bash
# Install all dependencies including the private fair-llm package (Option A):
pip install -r requirements.txt

# Or if using SSH (Option B):
pip install -r requirements-ssh.txt
```

#### Step 5: Configure API Keys

Set the required environment variables:

```bash
# Copy the environment template
cp .env.example .env

# Edit .env and add your actual API keys:
# OPENAI_API_KEY=your-actual-openai-key
# ANTHROPIC_API_KEY=your-actual-anthropic-key
```

Or export them directly:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

#### Step 6: Verify Installation

Run the verification script to ensure everything is properly configured:

```bash
python verify_setup.py
```

This will check:
- ✓ Python version (3.11+)
- ✓ Package installation
- ✓ API key configuration  
- ✓ FAIR-LLM import functionality
- ✓ Basic tool functionality

## 📚 Available Demos

### Core Demonstrations

1. **Single Agent with Calculator** (`demo_single_agent_calculator.py`)
   - Introduction to building a basic agent with tool usage
   - Learn the fundamental components: LLM, Tools, Memory, Planner

2. **Multi-Agent Collaboration** (`multi_agent_demo.py`)
   - Advanced hierarchical multi-agent system
   - Manager agent delegating to specialized workers
   - Real-world example: Bitcoin price research + calculation

3. **Advanced Calculator with Calculus** (`demo_advanced_calculator_calculus.py`)
   - Agent using multiple mathematical tools
   - Symbolic computation capabilities

4. **RAG from Documents** (`demo_rag_from_documents.py`)
   - Retrieval-Augmented Generation implementation
   - Document loading, splitting, and vector storage
   - Knowledge base querying

5. **Structured Output** (`demo_structured_output.py`)
   - Reliable JSON extraction from unstructured text
   - Pydantic validation and self-correction

6. **Model Comparison** (`demo_model_comparison.py`)
   - Compare responses from different LLM providers
   - Demonstrates the Model Abstraction Layer

### Running a Demo

```bash
# Basic single agent demo
python demo_single_agent_calculator.py

# Multi-agent collaboration demo
python multi_agent_demo.py

# Any other demo
python demo_name.py
```

## 🏗️ Framework Architecture Overview

The FAIR-LLM framework provides:

- **🤖 Advanced Agent Patterns**: ReAct (Reason+Act) cognitive cycles
- **🤝 Multi-Agent Collaboration**: Hierarchical manager-worker architectures
- **🧠 RAG Support**: Document grounding and knowledge base integration
- **🔌 Multiple LLM Support**: OpenAI, Anthropic, HuggingFace, Ollama
- **🧩 Modular Design**: Interface-driven architecture for easy customization
- **🛡️ Security**: Built-in input validation and sandboxed execution

## 🔧 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'fairlib'**
   - Ensure the private repository was installed correctly
   - Check your GitHub token has the correct permissions
   - Try reinstalling: `pip install --force-reinstall -r requirements.txt`

2. **API Key Errors**
   - Verify environment variables are set: `echo $OPENAI_API_KEY`
   - Ensure `.env` file is in the project root
   - Check API keys are valid and have sufficient credits

3. **Python Version Issues**
   - FAIR-LLM requires Python 3.11+
   - Check version: `python --version`
   - Consider using pyenv or conda for version management

4. **GitHub Authentication Failed**
   - Token expired or incorrect permissions
   - For private repo access, token needs `repo` scope
   - Try regenerating the token

### Getting Help

- Check the main [FAIR-LLM repository](https://github.com/USAFA-AI-Center/fair_llm) documentation
- Review the Developer's Guide: "A Guide to the FAIR Agentic Framework.docx"
- Open an issue in this demo repository for demo-specific problems
- Contact the development team for framework issues

## 📝 Project Structure

```
fair-llm-demos/
├── demos/
│   ├── multi_agent_demo.py    # Multi-agent collaboration
│   ├── demo_single_agent_calculator.py
│   └── ... (other demos)
├── .env.example               # Environment variable template
├── .env                       # Your actual env vars (git-ignored)
├── requirements.txt           # Dependencies with token auth
├── requirements-ssh.txt       # Dependencies with SSH auth
├── verify_setup.py           # Installation verification script
└── README.md                 # This file
```

## 🤝 Contributing

To contribute new demos:

1. Create a new demo file following the naming pattern `demo_*.py`
2. Include comprehensive comments explaining the concepts
3. Ensure all imports are from `fairlib`
4. Test with both OpenAI and Anthropic models
5. Submit a pull request with a description of what the demo teaches

## 📜 License

This demo repository is MIT licensed. The main FAIR-LLM framework license can be found in the private repository.

## 🙏 Acknowledgments

Developed by the USAFA AI Center team:
- Ryan R (rrabinow@uccs.edu)
- Austin W (austin.w@ardentinc.com)  
- Eli G (elijah.g@ardentinc.com)
- Chad M (Chad.Mello@afacademy.af.edu)