# FAIR-LLM Demos

A collection of runnable demos for the [FAIR-LLM framework](https://github.com/USAFA-AI-Center/fair_llm), showcasing everything from basic single-agent setups to multi-agent orchestration, RAG pipelines, MCP tool integration, Streamlit web UIs, and distributed vLLM inference.

## 🚀 Quick Installation

### Prerequisites
- Python 3.12 or higher
- Git

### Step 1: Clone the Repository
```bash
git clone git@github.com:USAFA-AI-Center/fair_llm_demos.git
cd fair_llm_demos
```

### Step 2: Create a Virtual Environment
**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- `fair-llm` - The core FAIR-LLM package
- `python-dotenv` - For environment variable management
- `rich` - For beautiful terminal output
- `anthropic` - For Anthropic Claude integration
- `faiss-cpu` - For vector search capabilities
- `seaborn` - For data visualization
- `pytest` - For testing

### Step 4: Verify Installation
Run the verification script:
```bash
python verify_setup.py
```

You will see output verifying the proper installation of dependent packages.

### Step 5: Set Up Your Hugging Face Token

Most demos use HuggingFace models, which require a free authentication token to download model weights.

#### Create a Hugging Face Account
1. Go to [huggingface.co](https://huggingface.co)
2. Click **Sign Up** in the top right corner
3. Create your free account using your email address
4. Verify your email address

#### Generate Your Access Token
1. Once logged in, click on your **profile picture** in the top right
2. Select **Settings** from the dropdown menu
3. In the left sidebar, click on **Access Tokens**
4. Click **New token** button
5. Give your token a name (e.g., "FAIR-LLM Demos")
6. Select **Read** role (this is sufficient for running the demos)
7. Click **Generate token**
8. **Important:** Copy your token immediately - you won't be able to see it again!

#### Configure Your Token
Log in to HuggingFace from the command line so the adapter can authenticate automatically:
```bash
huggingface-cli login
```
Paste your token when prompted. This stores it securely in your environment and no changes to the demo scripts are needed.

**Security Note:** Keep your token private! Don't share it or commit it to version control.

### Step 6: Configure API Keys (Optional)

Some demos use external APIs. See `google_api_setup.md` in the repo root for instructions on setting up Google Custom Search Engine (CSE) keys, which are needed for web search and web-search-based demos. OpenAI and Anthropic API keys should be configured in the main `fairlib` package's `settings.yml`.

---

## 🎯 Running the Demos

All demos are in the `demos/` folder. Run them from the repo root:

### Beginner — Start Here

```bash
# The "Hello, World!" of FAIR-LLM. A single agent with a calculator tool.
python demos/demo_single_agent_calculator.py

# A step up: multiple, more complex math tools.
python demos/demo_advanced_calculator_calculus.py

# Compare responses side-by-side across different LLM providers.
python demos/demo_model_comparison.py
```

### HuggingFace Adapter Deep-Dives

```bash
# Walkthrough of the HuggingFace adapter with transformers v4-compatible code paths.
python demos/demo_huggingface_v4.py

# Walkthrough of v5-specific features (SDPA attention, async streaming, BatchEncoding).
python demos/demo_huggingface_v5.py
```

### RAG (Retrieval-Augmented Generation)

```bash
# Ground an agent in your own documents using ChromaDB.
python demos/demo_rag_from_documents.py

# RAG using the FAISS vector store backend (recommended for most deployments).
python demos/demo_faiss_rag_from_readme.py
```

### Structured Output

```bash
# Extract clean, Pydantic-validated JSON from unstructured text, with a self-correction loop.
python demos/demo_structured_output.py
```

### Multi-Agent Orchestration

```bash
# A hierarchical manager-worker team solving a complex task.
python demos/demo_multi_agent.py
```

### Autograding (Committee of Agents)

```bash
# A committee of agents grading student essays against a rubric.
python demos/demo_committee_of_agents_essay_autograder.py

# A committee of agents grading student code submissions.
python demos/demo_committee_of_agents_coding_autograder.py
```

Support files (rubrics, sample submissions) for the autograder demos are in `demos/essay_autograder_files/` and `demos/coding_autograder_files/`.

### Agent Config Export & Load

```bash
# Save a complete agent configuration to JSON and reload it as a functional agent.
# Useful for sharing configs, version control, and prompt optimization workflows.
python demos/demo_agent_config_export_load.py
```

### Web Search + Plotting

```bash
# An agent that uses web search and graphing tools to answer data-driven questions.
python demos/demo_web_search_plot_agent.py
```

### MCP (Model Context Protocol) Integration

These demos require the `mcp` package (`pip install mcp`) and are located in `demos/mcp/`.

```bash
# Single agent that combines local tools with tools from an MCP server.
python demos/mcp/demo_mcp_single_agent.py

# Agent tool-calling demo using both SSE (remote) and stdio MCP transports.
python demos/mcp/demo_mcp_agent_tool_calling.py

# A full multi-agent research team with MCP-provided web search (Brave Search via Docker)
# and graceful fallback to built-in Google CSE search.
python demos/mcp/demo_multi_agent_research_team.py
```

### Streamlit Web UI

A browser-based chat interface for interacting with an agent. Run from the `demos/` directory:

```bash
cd demos
bash run_streamlit_demo.sh

# Or launch directly:
streamlit run demos/demo_streamlit_simple.py   # Simpler version
streamlit run demos/demo_streamlit.py          # Full version with LaTeX rendering
```

The Streamlit demo opens at `http://localhost:8501` in your browser.

### Distributed Inference (vLLM Load Balancer)

These demos connect to a running `vllm_load_manager` backend for distributed GPU inference instead of running models locally. They require a vLLM cluster to be running and accessible.

```bash
# Single agent using a load-balanced vLLM backend.
python demos/demo_single_agent_calculator_load_balancer.py

# Multiple agents hitting the same vLLM cluster in parallel (async).
python demos/demo_multi_agent_load_balancer.py
```

---

## 📦 Upgrading

To upgrade to the latest versions:
```bash
pip install --upgrade -r requirements.txt

# Or just upgrade fair-llm
pip install --upgrade fair-llm
```

---

## 🐛 Troubleshooting

### Virtual Environment Not Activated
Make sure your virtual environment is activated before installing or running:
```bash
# You should see (.venv) at the beginning of your terminal prompt
# If not, activate it:
source .venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows
```

### Missing Dependencies
If you get import errors, ensure all requirements are installed:
```bash
pip install -r requirements.txt --force-reinstall
```

### Python Version Issues
Verify you're using Python 3.12 or higher:
```bash
python --version
```

If you have multiple Python versions, you may need to use `python3` or `python3.12` explicitly.

### HuggingFace Authentication Errors
If you see authentication errors when running demos:
- Run `huggingface-cli login` and paste your token when prompted
- Ensure your token has at least **Read** permissions
- Generate a new token if needed and re-run `huggingface-cli login`

### MCP Demos Failing
- Install the MCP package: `pip install mcp`
- For `demo_multi_agent_research_team.py`, the Brave Search MCP server must be running in Docker. The demo will fall back to Google CSE automatically if it is unavailable.

### Streamlit Not Found
```bash
pip install streamlit
```

### Load Balancer Demos Failing
Ensure a `vllm_load_manager` instance is running and accessible on your network before starting these demos.

---

## 📚 What's Included

```
fair_llm_demos/
├── demos/
│   ├── mcp/
│   │   ├── demo_mcp_agent_tool_calling.py
│   │   ├── demo_mcp_single_agent.py
│   │   ├── demo_multi_agent_research_team.py
│   │   └── mcp_filesystem_server.py
│   ├── essay_autograder_files/        # Rubrics & sample essays for autograder demo
│   ├── coding_autograder_files/       # Rubrics & sample code for autograder demo
│   ├── demo_advanced_calculator_calculus.py
│   ├── demo_agent_config_export_load.py
│   ├── demo_committee_of_agents_coding_autograder.py
│   ├── demo_committee_of_agents_essay_autograder.py
│   ├── demo_faiss_rag_from_readme.py
│   ├── demo_huggingface_v4.py
│   ├── demo_huggingface_v5.py
│   ├── demo_model_comparison.py
│   ├── demo_multi_agent.py
│   ├── demo_multi_agent_load_balancer.py
│   ├── demo_rag_from_documents.py
│   ├── demo_single_agent_calculator.py
│   ├── demo_single_agent_calculator_load_balancer.py
│   ├── demo_streamlit.py
│   ├── demo_streamlit_simple.py
│   ├── demo_structured_output.py
│   ├── demo_web_search_plot_agent.py
│   └── run_streamlit_demo.sh
├── A Guide to the FAIR Agentic Framework.pdf
├── google_api_setup.md
├── pyproject.toml
├── requirements.txt
├── verify_setup.py
└── README.md
```

---

## 🎉 Next Steps

1. Run `python verify_setup.py` to confirm everything is working
2. Configure your HuggingFace token with `huggingface-cli login`
3. Start with `demos/demo_single_agent_calculator.py` and work your way up
4. Read **"A Guide to the FAIR Agentic Framework.pdf"** (included in this repo) for a deep-dive into the framework's design and capabilities

---

## 👥 Contributors

Developed by the USAFA AI Center team:
- Ryan R (rrabinow@uccs.edu)
- Austin W (austin.w@ardentinc.com)
- Eli G (elijah.g@ardentinc.com)
- Chad M (Chad.Mello@afacademy.af.edu)
