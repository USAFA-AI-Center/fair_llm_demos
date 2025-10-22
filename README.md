# FAIR-LLM Installation Guide

## 🚀 Quick Installation

### Prerequisites
- Python 3.8 or higher
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
- `fair-llm>=0.1` - The core FAIR-LLM package
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

## 🎯 Running the Demos

Once installed, try the demo scripts:

### Single Agent Calculator Demo
```bash
# Basic functionality
python demos/demo_single_agent_calculator.py
```

### RAG Enhanced Agent Demo
```bash
# Agent grounded with RAG
python demos/demo_rag_from_documents.py
```

## 📦 Upgrading

To upgrade to the latest versions:
```bash
# Upgrade all packages
pip install --upgrade -r requirements.txt

# Or just upgrade fair-llm
pip install --upgrade fair-llm
```

## 🐛 Troubleshooting

### Virtual Environment Not Activated
Make sure your virtual environment is activated before installing or running:
```bash
# You should see (venv) at the beginning of your terminal prompt
# If not, activate it:
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Missing Dependencies
If you get import errors, ensure all requirements are installed:
```bash
pip install -r requirements.txt --force-reinstall
```

### Python Version Issues
Verify you're using Python 3.8 or higher:
```bash
python --version
```

If you have multiple Python versions, you may need to use `python3` instead of `python`.

## 📚 What's Included

After installation, you'll have:
- ✅ The complete FAIR-LLM framework
- ✅ Multi-agent orchestration capabilities
- ✅ Document processing tools
- ✅ Complete demo applications

## 🎉 Next Steps

1. Run `python verify_setup.py` to confirm everything is working
2. Explore the `demos/` folder for examples
3. Try running the demo applications
4. Start building your own multi-agent applications!

## 👥 Contributors

Developed by the USAFA AI Center team:
- Ryan R (rrabinow@uccs.edu)
- Austin W (austin.w@ardentinc.com)
- Eli G (elijah.g@ardentinc.com)
- Chad M (Chad.Mello@afacademy.af.edu)