# demo_web_search_plot_agent.py
import os
import asyncio
import datetime
from typing import List, Any, Dict

from fairlib.modules.agent.multi_agent_runner import _create_default_manager_prompt_builder

"""
This script serves as a hands-on tutorial and demonstration of the framework's
most advanced feature: Hierarchical Multi-Agent Collaboration.

We will construct a "team" of AI agents with a clear reporting structure:
1.  A "Manager" Agent: Its job is to understand a complex user request and
    delegate sub-tasks to the appropriate worker.
2.  "Worker" Agents: Each worker has a specialized role and a specific tool,
    allowing it to excel at one type of task.

The scenario: A user wants to perform a task that requires both real-time
information gathering (research) and graph generation (code creation and execution). No single
agent can solve this alone, but by collaborating, the team can deliver a
comprehensive solution.

IMPORTANT NOTE: This demo does not consistently create graphing outputs. It is very dependent upon the results
outputted from the web_searcher. If it is able to hit a good link for the data extractor, a graph can be generated in
demos/outputs/. If you want to get a graph out of this demo, keep running it. Further improvements need to be
made to the data_extractor tool and the graphing tool in FAIR_LLM. A code validation agent may also allow
more consistent graphing outputs.
"""

# --- Step 1: Import all necessary components ---
from fairlib import (
    settings,
    OpenAIAdapter,
    ManagerPlanner,
    ToolRegistry,
    WebSearcherTool,
    ToolExecutor,
    WorkingMemory,
    ReActPlanner,
    SimpleAgent, 
    HierarchicalAgentRunner,
    GraphingTool,
    WebDataExtractor,
    BasicSecurityManager,
    Example,
    RoleDefinition,
    FormatInstruction,
    WorkerInstruction,
    PromptBuilder,
    AgentCapability
)

from dotenv import load_dotenv
load_dotenv()

settings.api_keys.openai_api_key = os.getenv("OPENAI_API_KEY")
settings.api_keys.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

class AgentDescriptionBuilder:
    """Builds comprehensive, structured descriptions for agents"""
    
    @staticmethod
    def build_description(capability: AgentCapability) -> str:
        """Generate a detailed description from structured capability"""
        
        description = f"""
AGENT: {capability.name}
PRIMARY FUNCTION: {capability.primary_function}

CAPABILITIES:
{chr(10).join(f'- {cap}' for cap in capability.capabilities)}

LIMITATIONS:
{chr(10).join(f'- {lim}' for lim in capability.limitations)}

INPUT FORMAT: {capability.input_format}
OUTPUT FORMAT: {capability.output_format}

TOOLS AVAILABLE: {', '.join(capability.tools)}

EXAMPLE TASKS THIS AGENT CAN HANDLE:
{chr(10).join(f'- {task}' for task in capability.example_tasks)}

KEYWORDS FOR DELEGATION: {', '.join(capability.delegation_keywords)}
"""
        return description.strip()


# Define agent capabilities
RESEARCHER_CAPABILITY = AgentCapability(
    name="Researcher",
    primary_function="Search the internet for current information",
    capabilities=[
        "Search the web for real-time information",
        "Find current prices, news, and facts",
        "Locate data sources, APIs, and datasets",
        "Discover relevant URLs and documentation",
        "Search for multiple related topics"
    ],
    limitations=[
        "Cannot extract data from URLs (only finds them)",
        "Cannot process or parse website content",
        "Cannot create visualizations",
        "Returns search results, not extracted data"
    ],
    input_format="Natural language search queries",
    output_format="JSON array of search results with titles, URLs, and snippets",
    example_tasks=[
        "Find current bitcoin prices",
        "Search for climate data sources",
        "Locate NASA temperature datasets",
        "Find stock market information",
        "Search for scientific research papers"
    ],
    delegation_keywords=["search", "find", "locate", "discover", "look up", "current", "latest", "real-time"],
    tools=["web_searcher"]
)

DATA_EXTRACTOR_CAPABILITY = AgentCapability(
    name="DataExtractor",
    primary_function="Extract actual data values from any web source using intelligent multi-strategy approach",
    capabilities=[
        "Automatically detect content type and choose extraction strategy",
        "Construct API calls from parameter documentation",
        "Follow download links to get data files",
        "Extract from HTML tables, lists, and embedded content",
        "Parse CSV, JSON, Excel, PDF formats automatically",
        "Use LLM to extract data from unstructured pages",
        "Try multiple strategies until actual data is found",
        "Handle any data domain (finance, climate, sports, etc.)"
    ],
    limitations=[
        "Cannot search for new URLs (needs URLs from Researcher)",
        "Cannot create visualizations",
        "May require multiple attempts for complex sources",
        "Some sites may require authentication"
    ],
    input_format="JSON array of search results with titles, URLs, and snippets",
    output_format="Structured data with actual values, metadata about extraction strategies used",
    example_tasks=[
        "Extract stock prices from finance websites",
        "Get weather data from meteorological services",
        "Pull statistics from government databases",
        "Extract sports scores from results pages",
        "Get economic indicators from central banks",
        "Parse research data from academic sources"
    ],
    delegation_keywords=["extract", "get data", "fetch", "download", "parse", "retrieve", "pull data", "obtain values"],
    tools=["web_data_extractor"]
)

GRAPHER_CAPABILITY = AgentCapability(
    name="Grapher",
    primary_function="Create visualizations from structured data",
    capabilities=[
        "Generate appropriate plot types automatically",
        "Create line plots, bar charts, scatter plots, histograms",
        "Handle time series visualizations",
        "Add professional styling and annotations",
        "Save high-resolution plots",
        "Process custom visualization instructions",
        "Accept multiple data formats (columns/rows, separate arrays, etc.)"
    ],
    limitations=[
        "Requires structured data",
        "Cannot search or extract data",
        "Cannot analyze plot meaning",
        "Needs data from DataExtractor agent"
    ],
    input_format="Structured data JSON (flexible formats accepted: {'columns': [...], 'rows': [...]}, {'x': [...], 'y': [...]}, or {'field1': [...], 'field2': [...]})",
    output_format="Plot metadata including file path and visualization details",
    example_tasks=[
        "Plot temperature over time",
        "Create bitcoin price chart",
        "Visualize data trends",
        "Generate scatter plot of correlations",
        "Make bar chart comparing categories",
        "Plot months vs anomalies data"
    ],
    delegation_keywords=["plot", "graph", "visualize", "chart", "diagram", "draw", "create visualization"],
    tools=["graphing_tool"]
)


# --- Step 3: Enhanced Manager Prompt Builder ---
class EnhancedManagerPromptBuilder:
    """Creates enhanced prompts for manager agents with proper delegation rules"""
    
    @staticmethod
    def create_delegation_rules_as_role(agents: Dict[str, Any]) -> RoleDefinition:
        """Create delegation rules as an enhanced role definition"""
        
        role_text = """You are a Manager Agent responsible for coordinating specialized worker agents to complete complex tasks.

CORE RESPONSIBILITIES:
1. Analyze user requests and break them into subtasks
2. Delegate each subtask to the most appropriate worker agent
3. Coordinate the workflow between agents

DELEGATION PRINCIPLES:
- Each agent has ONE specific job - use them accordingly
- Follow the natural data flow: Search → Extract → Process → Summarize
- Never skip agents in the workflow
- Always wait for agent results before proceeding

WORKFLOW PATTERNS:
1. Data Discovery Flow: Researcher → DataExtractor → Grapher → Summarizer
2. Direct Data Flow: If user provides URLs → DataExtractor → Grapher → Summarizer
3. Analysis Flow: If data already exists → Grapher → Summarizer

AGENT CAPABILITIES:
"""
        
        # Add specific agent capabilities
        for name, agent in agents.items():
            if hasattr(agent, 'capability') and agent.capability:
                cap = agent.capability
                role_text += f"\n{name.upper()}:"
                role_text += f"\n  Primary Function: {cap.primary_function}"
                role_text += f"\n  Use When: Task contains keywords: {', '.join(cap.delegation_keywords)}"
                role_text += f"\n  Input: {cap.input_format}"
                role_text += f"\n  Output: {cap.output_format}"
                role_text += f"\n  Tools: {', '.join(cap.tools)}"
        
        return RoleDefinition(role_text)
    
    @staticmethod
    def create_strict_format_instructions() -> List[FormatInstruction]:
        """Create format instructions that prevent output errors"""
        
        return [
            FormatInstruction("""CRITICAL: You MUST output in this EXACT format:

Line 1: Thought: [your reasoning about what to do next]
Line 2: Action: {"tool_name": "delegate", "tool_input": {"worker_name": "[AgentName]", "task": "[specific task description]"}}

RULES:
- The Action line MUST contain valid JSON only
- Never add extra text after the JSON
- Use double quotes for all JSON strings
- Always use tool_name "delegate" for worker delegation
- For final answer use: {"tool_name": "final_answer", "tool_input": "[your complete answer]"}"""),
            
            FormatInstruction("""COMMON MISTAKES TO AVOID:
WRONG: Action: Using tool 'delegate' with input...
RIGHT: Action: {"tool_name": "delegate", "tool_input": {...}}

WRONG: Action: Researcher should search for data. I will delegate this task to the Researcher.
RIGHT: Action: {"tool_name": "delegate", "tool_input": {"worker_name": "Researcher", "task": "search for data"}}

WRONG: Mixed format output
RIGHT: Exactly two lines - Thought then Action with JSON""")
        ]
    
    @staticmethod
    def create_enhanced_worker_instructions(agents: Dict[str, Any]) -> List[WorkerInstruction]:
        """Create enhanced worker instructions with capability details"""
        
        instructions = []
        
        for name, agent in agents.items():
            if hasattr(agent, 'capability') and agent.capability:
                cap = agent.capability
                
                # Create detailed description
                description = f"{cap.primary_function}. "
                description += f"CAPABILITIES: {', '.join(cap.capabilities)}. "
                description += f"INPUT_FORMAT: {cap.input_format}. "
                description += f"OUTPUT_FORMAT: {cap.output_format}. "
                description += f"EXAMPLE_USAGE: {', '.join(cap.example_tasks)}"
                
                instructions.append(WorkerInstruction(name, description))
            else:
                # Fallback to basic description
                instructions.append(WorkerInstruction(name, agent.role_description))
        
        return instructions


def enhance_manager_prompt_builder(prompt_builder: PromptBuilder, agents: Dict[str, Any]) -> PromptBuilder:
    """
    Properly enhance an existing prompt builder with delegation rules
    """
    
    # 1. Replace or enhance the role definition
    enhanced_role = EnhancedManagerPromptBuilder.create_delegation_rules_as_role(agents)
    prompt_builder.role_definition = enhanced_role
    
    # 2. Replace worker instructions with enhanced versions
    prompt_builder.worker_instructions = EnhancedManagerPromptBuilder.create_enhanced_worker_instructions(agents)
    
    # 3. Add strict format instructions
    prompt_builder.format_instructions = EnhancedManagerPromptBuilder.create_strict_format_instructions()
    
    return prompt_builder

# Add these generic examples to your prompt builder

def add_generic_data_extraction_examples(prompt_builder: PromptBuilder):
    """Add generic examples that work for any data extraction scenario"""
    
    # Example 1: Handling API documentation generically
    prompt_builder.examples.append(Example(
        """User Request: Get data about [any topic]

Thought: The user wants data about [topic]. I'll search for relevant data sources.
Action: {"tool_name": "delegate", "tool_input": {"worker_name": "Researcher", "task": "search for [topic] data download CSV JSON API statistics dataset"}}

Tool Observation: Result from Researcher: [{"title": "Data Source", "url": "https://example.com/data", "snippet": "Access data about [topic]"}]

Thought: Found a potential data source. I'll have the DataExtractor attempt to get the actual data values.
Action: {"tool_name": "delegate", "tool_input": {"worker_name": "DataExtractor", "task": "Extract actual data values about [topic] from https://example.com/data. If you find API documentation, construct the appropriate API call. If you find download links, follow them to get the data."}}

Tool Observation: Result from DataExtractor: {
  "status": "partial",
  "data_found": false,
  "documentation_found": [{"columns": ["parameter", "type"], "rows": [...]}],
  "extraction_strategies": ["Direct fetch", "Analyzing content", "Searching for data links"]
}

Thought: The DataExtractor found documentation but not actual data. Let me search for more direct data sources or specific file downloads.
Action: {"tool_name": "delegate", "tool_input": {"worker_name": "Researcher", "task": "search for [topic] data direct download filetype:csv OR filetype:json OR filetype:xlsx raw data"}}"""
    ))
    
    # Example 2: Successfully extracting data
    prompt_builder.examples.append(Example(
        """User Request: Show me a chart of [any metric]

Thought: I need to find data about [metric] to create a chart. Let me search for data sources.
Action: {"tool_name": "delegate", "tool_input": {"worker_name": "Researcher", "task": "search for [metric] data statistics historical current values download"}}

Tool Observation: Result from Researcher: [{"url": "https://data.example.com/metrics"}]

Thought: I'll extract the data from this source.
Action: {"tool_name": "delegate", "tool_input": {"worker_name": "DataExtractor", "task": "Extract data about [metric] from https://data.example.com/metrics. Get actual numerical values, time periods, or categories as appropriate."}}

Tool Observation: Result from DataExtractor: {
  "status": "success",
  "data_found": true,
  "extracted_data": [{
    "columns": ["period", "value"],
    "rows": [["2023-Q1", "42.5"], ["2023-Q2", "45.1"], ...],
    "data_type": "time_series"
  }]
}

Thought: Great! I have the data. Now I'll create a visualization. I will delegate this task to the Grapher.
Action: {"tool_name": "delegate", "tool_input": {"worker_name": "Grapher", "task": "Create an appropriate chart for this [metric] data. Choose the best visualization type based on the data structure."}}"""
    ))
    
    # Example 3: Handling various data formats
    prompt_builder.examples.append(Example(
        """User Request: Analyze [topic] trends

Thought: To analyze trends, I need historical data about [topic]. Let me search for comprehensive datasets.
Action: {"tool_name": "delegate", "tool_input": {"worker_name": "Researcher", "task": "search for [topic] historical data trends over time statistics database"}}

Tool Observation: Result from Researcher: Multiple data sources found

Thought: I'll try to extract data from the most promising source.
Action: {"tool_name": "delegate", "tool_input": {"worker_name": "DataExtractor", "task": "Extract trend data about [topic]. Look for time-based data showing changes over months or years. Handle any format: tables, CSV downloads, JSON APIs, or embedded data."}}

Tool Observation: Result from DataExtractor: {
  "status": "success",
  "data_found": true,
  "extracted_data": [{
    "columns": ["date", "category", "value", "change"],
    "rows": [...],
    "data_type": "categorical_time_series"
  }],
  "extraction_strategies": ["Direct fetch", "Found data table in HTML", "SUCCESS: Extracted from HTML table"]
}

Thought: Excellent! I have trend data with multiple dimensions. I'll visualize this to show the trends clearly.
Action: {"tool_name": "delegate", "tool_input": {"worker_name": "Grapher", "task": "Create a multi-line or grouped chart showing trends by category over time. Use appropriate colors and labels."}}

Tool Observation: Result from Grapher: {"status": "success", "plot_metadata": {"file_path": "./outputs/plot_trends.png"}}

Thought: I have completed the full analysis with data, visualization, and insights.
Action: {"tool_name": "final_answer", "tool_input": "I've analyzed the [topic] trends with the following findings: [summary]. The visualization has been saved as 'plot_trends.png' and shows [key visual insights]. The data reveals [main patterns and conclusions]."}"""
    ))


def add_generic_manager_guidance(prompt_builder: PromptBuilder):
    """Add generic guidance for handling any type of data request"""
    
    guidance = FormatInstruction("""
GENERIC DATA EXTRACTION STRATEGIES:

When users request data about ANY topic:
1. Search broadly first: Include keywords like "data", "statistics", "download", "API", "dataset"
2. Be persistent: If first attempt returns documentation, search for direct downloads
3. Try multiple sources: Different sites structure data differently
4. Let DataExtractor handle complexity: It will try multiple strategies automatically

Common patterns by request type:
- "Show me X over time" → Search for historical/time-series data
- "Compare X and Y" → Search for datasets containing both variables
- "Current X statistics" → Search for real-time or recent data
- "Analyze X trends" → Search for historical data with multiple time points

DataExtractor capabilities:
- Constructs API calls from documentation
- Follows download links automatically
- Extracts from HTML tables, embedded data
- Handles CSV, JSON, Excel, PDF formats
- Uses LLM to extract data from unstructured pages

NEVER give up after one attempt - always try alternative searches or sources.
""")
    
    prompt_builder.format_instructions.append(guidance)
    
    return prompt_builder


# --- Step 4: Create Enhanced Agent Function ---
def create_enhanced_agent(
    llm,
    tools,
    capability: AgentCapability,
    stateless: bool = True
):
    """
    Create an agent with enhanced structured description
    """
    tool_registry = ToolRegistry()
    for tool in tools:
        tool_registry.register_tool(tool)
    
    planner = ReActPlanner(llm, tool_registry)
    executor = ToolExecutor(tool_registry)
    memory = WorkingMemory()
    
    # Create agent with capability
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=memory,
        stateless=stateless
    )
    
    # Set capability and auto-generate description
    agent.capability = capability
    agent.role_description = AgentDescriptionBuilder.build_description(capability)
    
    return agent


# --- Step 5: Main Function ---
async def main():
    print("Initializing fairlib.core.components...")
    
    llm = OpenAIAdapter(
       api_key=settings.api_keys.openai_api_key,
       model_name=settings.models["openai_gpt4"].model_name
    )

    web_search_config = {
        "google_api_key": settings.search_engine.google_cse_search_api,
        "google_search_engine_id": settings.search_engine.google_cse_search_engine_id,
        "cache_ttl": settings.search_engine.web_search_cache_ttl,
        "cache_max_size": settings.search_engine.web_search_cache_max_size,
        "max_results": settings.search_engine.web_search_max_results,
    }

    # Create agents with enhanced capabilities
    researcher = create_enhanced_agent(
        llm,
        [WebSearcherTool(config=web_search_config)],
        RESEARCHER_CAPABILITY
    )

    data_extractor = create_enhanced_agent(
        llm,
        [WebDataExtractor(llm=llm)],
        DATA_EXTRACTOR_CAPABILITY
    )

    grapher = create_enhanced_agent(
        llm,
        [GraphingTool(
            security_manager=BasicSecurityManager(),
            llm=llm,
            output_dir="./outputs"
        )],
        GRAPHER_CAPABILITY
    )

    workers = {
        "Researcher": researcher, 
        "DataExtractor": data_extractor, 
        "Grapher": grapher
    }

    manager_memory = WorkingMemory()

    # Create and enhance prompt builder
    prompt_builder = PromptBuilder()
    prompt_builder = enhance_manager_prompt_builder(prompt_builder, workers)
    prompt_builder = add_generic_manager_guidance(prompt_builder)
    add_generic_data_extraction_examples(prompt_builder)

    # Add custom example with correct format
    prompt_builder.examples.append(Example(
        """User Request: Find the price of bitcoin over the past 5 years and plot the data.

Thought: The user wants historical bitcoin price data and a visualization. This requires: 1) researching current bitcoin price sources, 2) extracting the actual data, 3) creating a plot, and 4) summarizing the results. I'll start by delegating to the Researcher.
Action: {"tool_name": "delegate", "tool_input": {"worker_name": "Researcher", "task": "Search for reliable sources of bitcoin price history over the past 5 years (2019-2024). Find APIs or datasets with daily/weekly price data."}}

Tool Observation: Result from Researcher: {
  "query": "bitcoin price history past 5 years API dataset daily weekly price data",
  "enhanced_query": "bitcoin price history past 5 years API dataset daily weekly price data 2024",
  "search_type": "general",
  "results": [
    {
      "title": "Bitcoin USD (BTC-USD) Price History & Historical Data - Yahoo Finance",
      "url": "https://finance.yahoo.com/quote/BTC-USD/history",
      "display_url": "finance.yahoo.com",
      "snippet": "Download historical data for Bitcoin USD. Available in daily, weekly or monthly format back to when Bitcoin USD (BTC-USD) was first traded.",
      "source": "Google",
      "total_results": "12500",
      "relevance_score": 35
    },
    {
      "title": "CoinGecko API Documentation",
      "url": "https://www.coingecko.com/en/api/documentation",
      "display_url": "www.coingecko.com",
      "snippet": "Free cryptocurrency API with historical price data. Get bitcoin prices with /coins/{id}/market_chart endpoint.",
      "source": "Google",
      "relevance_score": 32
    }
  ],
  "timestamp": "2024-11-29T10:30:00.000000"
}

Thought: Found good sources. Yahoo Finance has downloadable historical data. I'll extract the bitcoin price data from there.
Action: {"tool_name": "delegate", "tool_input": {"worker_name": "DataExtractor", "task": "Extract bitcoin historical price data from https://finance.yahoo.com/quote/BTC-USD/history - get daily closing prices for the past 5 years (2019-2024)"}}

Tool Observation: Result from DataExtractor: {
  "status": "success",
  "extracted_data": [{
    "source_url": "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range",
    "data_type": "time_series",
    "columns": ["date", "price_usd"],
    "rows": [
      ["2019-01-01", "3746.71"],
      ["2020-01-01", "7200.17"],
      ["2021-01-01", "29374.15"],
      ["2022-01-01", "47686.81"],
      ["2023-01-01", "16547.91"],
      ["2024-01-01", "42265.33"]
    ],
    "metadata": {"data_points": 1827, "source": "CoinGecko API"}
  }],
  "summary": "Successfully extracted 1,827 daily price points for bitcoin from January 2019 to January 2024."
}

Thought: The DataExtractor successfully retrieved the price data. Now I need to create a visualization. I'll delegate to the Grapher with specific instructions for the plot.
Action: {"tool_name": "delegate", "tool_input": {"worker_name": "Grapher", "task": "Create a line plot of bitcoin prices over the past 5 years using the extracted data. Use these specifications: 1) X-axis: dates from 2019-2024, 2) Y-axis: price in USD (use log scale if needed), 3) Title: 'Bitcoin Price History (2019-2024)', 4) Add grid lines and format prices with commas, 5) Save the plot with high resolution."}}

Tool Observation: Result from Grapher: {
  "status": "success",
  "plot_metadata": {
    "file_path": "./outputs/plot_20240627_103045.png",
    "file_name": "plot_20240627_103045.png",
    "plot_type": "line",
    "dimensions": {"width": 1200, "height": 800},
    "title": "Bitcoin Price History (2019-2024)"
  },
  "data_analysis": {
    "suggested_plot_type": "line",
    "has_time_series": true,
    "num_rows": 1827
  }
}

Thought: I have successfully coordinated all agents to complete the task. The Researcher found data sources, the DataExtractor retrieved the price history, and the Grapher created a visualization. I can now provide the final answer to the user.
Action: {"tool_name": "final_answer", "tool_input": "I've successfully analyzed bitcoin's price over the past 5 years and created a visualization. Here's what I found:\n\nBitcoin's price journey from 2019-2024 shows extreme volatility. Starting at $3,747 (Jan 2019), it reached an all-time high of $69,000 (Nov 2021) before crashing to $17,600 (June 2022). Current price: ~$42,265, representing a 1,028% gain over 5 years.\n\nI've created a detailed line plot saved as 'plot_20240627_103045.png' that visualizes this roller-coaster trajectory using a logarithmic scale to better show percentage changes. The plot includes 1,827 daily price points and highlights major peaks and troughs.\n\nKey patterns visible in the data include: the 2020-2021 bull run (driven by institutional adoption), the 2022 bear market (following rate hikes), and recent recovery in 2023-2024. The visualization clearly shows bitcoin's characteristic boom-bust cycles, with roughly 2-year periods between major peaks."}"""
    ))

    manager_planner = ManagerPlanner(llm, workers, prompt_builder)
    manager_agent = SimpleAgent(llm, manager_planner, None, manager_memory) 

    team_runner = HierarchicalAgentRunner(manager_agent, workers)

    # Test query
    user_query = "I want to generate a plot showing the temperature of the earth over the last 10 years."

    print(f"\n{'='*100}")
    print(f"User Query: {user_query}")
    print(f"{'='*100}\n")

    final_answer = await team_runner.arun(user_query)
    
    # Display the final result
    print("\n✅ --- FINAL Synthesized Answer ---")
    print(final_answer)
    print(f"\n{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())