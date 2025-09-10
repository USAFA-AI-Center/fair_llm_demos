# Google Custom Search API Setup Guide

## 🎯 Overview

This guide will walk you through setting up Google Custom Search Engine (CSE) API credentials for the multi-agent demo. The entire process is **FREE** for up to 100 searches per day.

**Time Required:** ~15 minutes  
**Cost:** Free (with limits)  
**Prerequisites:** A Google account

---

## 📋 What You'll Need

1. **Google Cloud Project** - To manage your API access
2. **Custom Search API Key** - To authenticate your requests
3. **Search Engine ID** - To identify your custom search engine

---

## 🚀 Step-by-Step Instructions

### Part 1: Create a Google Cloud Project

1. **Go to Google Cloud Console**
   - Navigate to: https://console.cloud.google.com
   - Sign in with your Google account

2. **Create a New Project**
   - Click the project dropdown at the top of the page
   - Click "New Project"
   - Enter a project name (e.g., "fair-demos")
   - No need to give an organization
   - Click "Create"
   - Wait ~30 seconds for the project to be created

3. **Make Sure Your New Project is Selected**
   - The project name should appear in the top bar
   - If not, click the dropdown and select your new project

### Part 2: Enable the Custom Search API

1. **Open the API Library**
   - In the left sidebar, click "APIs & Services" → "Library"
   - Or go directly to: https://console.cloud.google.com/apis/library
      - if you use the above link remember to re-select your project

2. **Search for Custom Search API**
   - In the search bar, type "Custom Search API"
   - Click on "Custom Search API"

3. **Enable the API**
   - Click the blue "ENABLE" button
   - Wait for the API to be enabled (~10 seconds)

### Part 3: Create API Credentials

1. **Go to Credentials Page**
   - After enabling the API, click "+ CREATE CREDENTIALS" at the top (blue text near the name of your project)
   - Select API key

2. **Create an API Key Instead**
   - Select "None" for aplication restrictions
   - Slect "Don't restrict key" for API restrictions
   - Click "Create" (your API key will be generated immediately)
   - **IMPORTANT:** Copy this key now! You'll need it later
   - Click "Close" or "Done"

### Part 4: Create a Custom Search Engine

1. **Go to Programmable Search Engine**
   - Navigate to: https://programmablesearchengine.google.com/cse/all
   - Or search for "Google Programmable Search Engine"

2. **Create a New Search Engine**
   - Click "Add" or "Get started"
   - Fill in the form:
     - **Search engine name:** "Web Search Demo Tool"
     - **What to search:** Choose "Search the entire web"
     - **Search settings:** Leave as default
   - Click "Create"

3. **Get Your Search Engine ID**
   - After creation, you'll see a "Search engine ID"
   - Copy this ID (looks like: "a1b2c3d4e5f6g7h8i9")
      - it is in the first line here: <script async src="https://cse.google.com/cse.js?cx=YOUR_ENGINE_ID">

### Part 5: Configure Your Environment

1. **Create a `.env` file** in your project directory:

```bash
# Create the file
touch .env
cp .env.example .env
```

2. **Add your credentials** to the `.env` file:

```env
# Google Custom Search API Configuration
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Google CSE keys and settings
GOOGLE_CSE_SEARCH_API=AI-xxxxxxxxx
GOOGLE_CSE_SEARCH_ENGINE_ID=90-xxxxxxxxxxx
```

3. **Test Your Setup**
   - Run the demo: `python demo_multi_agent.py`
   - You should see sources coming from real websites
      - "source": "Google"

---

## 🎓 Understanding API Limits

### Free Tier Limits
- **100 searches per day** (resets at midnight Pacific Time)
- **10 results per search** maximum
- No credit card required

### What Counts as a Search?
- Each call to the search tool = 1 search
- Pagination (getting more results) = additional searches

### Monitoring Usage
- Check your usage at: https://console.cloud.google.com/apis/api/customsearch.googleapis.com/metrics
- Set up alerts if desired

---

## 🔍 Troubleshooting

### Common Issues and Solutions

**Issue: "API key not valid" error**
- Solution: Double-check your API key is copied correctly
- Ensure the Custom Search API is enabled in your project

**Issue: "Invalid search engine ID"**
- Solution: Verify you're using the Search Engine ID, not the name
- The ID should be alphanumeric (e.g., "a1b2c3d4e5f6g7h8i9")

**Issue: "Quota exceeded" error**
- Solution: You've hit the 100/day limit
- Wait until tomorrow or use mock mode for testing

**Issue: Demo still uses mock search**
- Solution: Ensure both environment variables are set
- Check for typos in variable names
- Restart your Python environment after changing `.env`

---

## 💡 Tips for Students

1. **Start with Mock Mode**: Test your multi-agent logic with mock data first
2. **Share API Keys Carefully**: Never commit `.env` files to git
3. **Monitor Usage**: 100 searches go quickly when debugging
4. **Use Caching**: The framework caches results to save API calls

---

## 📚 Additional Resources

- [Google Custom Search API Documentation](https://developers.google.com/custom-search/v1/overview)
- [Search Engine Configuration Options](https://support.google.com/programmable-search/answer/4513882)

---

## 🎉 Success Checklist

Before running the demo with real search:

- [ ] Created Google Cloud Project
- [ ] Enabled Custom Search API
- [ ] Created and copied API Key
- [ ] Created Custom Search Engine
- [ ] Copied Search Engine ID
- [ ] Created `.env` file with both values
- [ ] Tested the demo and saw "Real Google Search API detected"

---

*Last updated: Wed Sep 10 2025*