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

### Step 5: Set Up Your Hugging Face Token

The demo files use Hugging Face models, which require a free authentication token.

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

#### Add Token to Demo Files
Each demo file contains a line that looks like this:
```python
HuggingFaceAdapter("dolphin3-qwen25-0.5b", auth_token="")
```

Simply paste your token between the empty quotes:
```python
HuggingFaceAdapter("dolphin3-qwen25-0.5b", auth_token="hf_YourTokenHere")
```

**Security Note:** Keep your token private! Don't share it or commit it to version control.

## 🎯 Running the Demos

Once installed and your Hugging Face token is configured, try the demo scripts:

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

### Hugging Face Authentication Errors
If you see authentication errors when running demos:
- Verify your token is correctly pasted in the demo file
- Ensure there are no extra spaces in the token string
- Check that your token has at least **Read** permissions
- Generate a new token if needed and replace the old one

## 📚 What's Included

After installation, you'll have:
- ✅ The complete FAIR-LLM framework
- ✅ Multi-agent orchestration capabilities
- ✅ Document processing tools
- ✅ Complete demo applications

## 🎉 Next Steps

1. Run `python verify_setup.py` to confirm everything is working
2. Set up your Hugging Face token (see Step 5 above)
3. Explore the `demos/` folder for examples
4. Try running the demo applications
5. Start building your own multi-agent applications!

## 👥 Contributors

Developed by the USAFA AI Center team:
- Ryan R (rrabinow@uccs.edu)
- Austin W (austin.w@ardentinc.com)
- Eli G (elijah.g@ardentinc.com)
- Chad M (Chad.Mello@afacademy.af.edu)