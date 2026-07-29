# demo_web_search_plot_agent.py

"""
Multi-agent research-and-plot pipeline: a real model managing a team of
specialist agents through the workers-as-tools path.

The scenario is unchanged from the original hierarchical version of this
demo: a user asks for a plot that requires real-time information
gathering (research), data extraction, and graph generation. No single
agent can solve this alone. What changed is the team wiring. There is no
dedicated orchestrator any more: each specialist is an ordinary stateless
SimpleAgent wrapped in a WorkerAgentTool, and the manager is a plain
SimpleAgent built by build_worker_manager, whose planner is the
batch-capable MultiActionReActPlanner and whose tools are the workers.
Delegation is an ordinary typed tool call.

What it shows:
  - WorkerAgentTool adapts each specialist agent into a typed tool. The
    manager model learns what each worker is for from the tool
    description in its rendered tool catalog; the structured
    AgentCapability definitions below generate those descriptions.
  - build_worker_manager is pure wiring: MultiActionReActPlanner plus
    ToolExecutor over a registry of worker tools. Application prompt
    content (role, workflow guidance, examples) rides in through a
    PromptBuilder; the planner merges its own mandatory JSON format
    instructions, so no hand-written delegation format is needed.
  - The Researcher and DataExtractor workers declare
    SideEffect.READ_ONLY: they look things up and mutate nothing, so
    independent delegations to them can be issued together in one turn
    and dispatched concurrently by the side-effect-aware executor. This
    pipeline is mostly dependent (search feeds extraction feeds
    plotting), so the prompt tells the manager to batch only truly
    independent lookups.
  - The Grapher worker keeps the conservative EXTERNAL default: it
    executes generated plotting code and writes image files to
    ./outputs, so its delegations run as a sequential barrier.

Requirements: a Google CSE API key and search engine ID in settings, and
a local HuggingFace model (the first run downloads weights). The run is
driven end to end by a real small instruct model, so expect stochastic
imperfection: it may take extra turns, retry searches, or fall short of
a clean plot. That is the honest behavior of the system, not a scripted
transcript.

Run:
    python demos/demo_web_search_plot_agent.py
"""

import asyncio
import os
from typing import Dict, List

# --- Step 1: Import all necessary components ---
from fairlib import (
    AgentCapability,
    BasicSecurityManager,
    Example,
    FormatInstruction,
    GraphingTool,
    HuggingFaceAdapter,
    PromptBuilder,
    ReActPlanner,
    RoleDefinition,
    SimpleAgent,
    ToolExecutor,
    ToolRegistry,
    WebDataExtractor,
    WebSearcherTool,
    WorkerAgentTool,
    WorkingMemory,
    build_worker_manager,
    settings,
)
from fairlib.core.interfaces.llm import AbstractChatModel
from fairlib.core.interfaces.tools import AbstractTool, SideEffect


class AgentDescriptionBuilder:
    """Builds comprehensive, structured descriptions for agents.

    The generated text becomes each WorkerAgentTool's description, which
    is what the manager model reads in its rendered tool catalog when it
    decides where to delegate.
    """

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


# --- Step 3: Manager Prompt Content ---
class EnhancedManagerPromptBuilder:
    """Creates the application prompt content for the fan-out manager.

    Only role, workflow, and strategy text lives here. The output format
    (the mandatory thought-plus-actions JSON) belongs to the
    MultiActionReActPlanner inside build_worker_manager, which merges its
    own format instructions; a hand-written delegation format would
    drift out of sync with the planner's parser.
    """

    @staticmethod
    def create_delegation_rules_as_role(capabilities: Dict[str, AgentCapability]) -> RoleDefinition:
        """Create delegation rules as an enhanced role definition"""

        role_text = """You are a Manager Agent responsible for coordinating specialized worker agents to complete complex tasks. Each worker is available to you as a tool: calling the tool delegates a subtask to that worker and the observation is the worker's final answer.

CORE RESPONSIBILITIES:
1. Analyze user requests and break them into subtasks
2. Delegate each subtask to the most appropriate worker tool
3. Coordinate the workflow between workers

DELEGATION PRINCIPLES:
- Each worker has ONE specific job - use them accordingly
- Follow the natural data flow: Search -> Extract -> Plot
- Never skip workers in the workflow
- Each stage of the pipeline depends on the previous stage's output, so delegate one stage per turn and wait for its observation
- Batch several delegations in one turn ONLY when they are truly independent (for example, two searches on unrelated subjects); independent read-only delegations are dispatched concurrently

WORKFLOW PATTERNS:
1. Data Discovery Flow: Researcher -> DataExtractor -> Grapher
2. Direct Data Flow: If user provides URLs -> DataExtractor -> Grapher
3. Analysis Flow: If data already exists -> Grapher

WORKER CAPABILITIES:
"""

        # Add specific worker capabilities
        for name, cap in capabilities.items():
            role_text += f"\n{name.upper()}:"
            role_text += f"\n  Primary Function: {cap.primary_function}"
            role_text += f"\n  Use When: Task contains keywords: {', '.join(cap.delegation_keywords)}"
            role_text += f"\n  Input: {cap.input_format}"
            role_text += f"\n  Output: {cap.output_format}"
            role_text += f"\n  Tools: {', '.join(cap.tools)}"

        return RoleDefinition(role_text)


def enhance_manager_prompt_builder(
    prompt_builder: PromptBuilder, capabilities: Dict[str, AgentCapability]
) -> PromptBuilder:
    """Install the manager role definition on an existing prompt builder.

    Only the role is installed: what each worker is for is carried by the
    WorkerAgentTool descriptions in the rendered tool catalog, and the
    delegation format is owned by the planner.
    """
    prompt_builder.role_definition = (
        EnhancedManagerPromptBuilder.create_delegation_rules_as_role(capabilities)
    )
    return prompt_builder


# Generic examples for the prompt builder. Each model response in an
# example is the planner's mandatory JSON shape: one object with a
# 'thought' string and an 'actions' array of {tool_name, tool_input}.

def add_generic_data_extraction_examples(prompt_builder: PromptBuilder):
    """Add generic examples that work for any data extraction scenario"""

    # Example 1: Handling API documentation generically, with persistence
    prompt_builder.examples.append(Example(
        """User Request: Get data about [any topic]

{"thought": "The user wants data about [topic]. Extraction depends on URLs I do not have yet, so this turn is a single search delegation.", "actions": [{"tool_name": "Researcher", "tool_input": "search for [topic] data download CSV JSON API statistics dataset"}]}

Observation: [Researcher] [{"title": "Data Source", "url": "https://example.com/data", "snippet": "Access data about [topic]"}]

{"thought": "Found a potential data source. I will have the DataExtractor attempt to get the actual data values.", "actions": [{"tool_name": "DataExtractor", "tool_input": "Extract actual data values about [topic] from https://example.com/data. If you find API documentation, construct the appropriate API call. If you find download links, follow them to get the data."}]}

Observation: [DataExtractor] {"status": "partial", "data_found": false, "documentation_found": [{"columns": ["parameter", "type"], "rows": ["..."]}], "extraction_strategies": ["Direct fetch", "Analyzing content", "Searching for data links"]}

{"thought": "The DataExtractor found documentation but not actual data. Let me search for more direct data sources or specific file downloads.", "actions": [{"tool_name": "Researcher", "tool_input": "search for [topic] data direct download filetype:csv OR filetype:json OR filetype:xlsx raw data"}]}"""
    ))

    # Example 2: Successfully extracting data and plotting it
    prompt_builder.examples.append(Example(
        """User Request: Show me a chart of [any metric]

{"thought": "I need to find data about [metric] to create a chart. Let me search for data sources.", "actions": [{"tool_name": "Researcher", "tool_input": "search for [metric] data statistics historical current values download"}]}

Observation: [Researcher] [{"url": "https://data.example.com/metrics"}]

{"thought": "I will extract the data from this source.", "actions": [{"tool_name": "DataExtractor", "tool_input": "Extract data about [metric] from https://data.example.com/metrics. Get actual numerical values, time periods, or categories as appropriate."}]}

Observation: [DataExtractor] {"status": "success", "data_found": true, "extracted_data": [{"columns": ["period", "value"], "rows": [["2023-Q1", "42.5"], ["2023-Q2", "45.1"]], "data_type": "time_series"}]}

{"thought": "I have the data. Now I will create a visualization.", "actions": [{"tool_name": "Grapher", "tool_input": "Create an appropriate chart for this [metric] data. Choose the best visualization type based on the data structure."}]}"""
    ))

    # Example 3: Completing the pipeline and finishing
    prompt_builder.examples.append(Example(
        """User Request: Analyze [topic] trends

{"thought": "To analyze trends, I need historical data about [topic]. Let me search for comprehensive datasets.", "actions": [{"tool_name": "Researcher", "tool_input": "search for [topic] historical data trends over time statistics database"}]}

Observation: [Researcher] Multiple data sources found

{"thought": "I will try to extract data from the most promising source.", "actions": [{"tool_name": "DataExtractor", "tool_input": "Extract trend data about [topic]. Look for time-based data showing changes over months or years. Handle any format: tables, CSV downloads, JSON APIs, or embedded data."}]}

Observation: [DataExtractor] {"status": "success", "data_found": true, "extracted_data": [{"columns": ["date", "category", "value", "change"], "rows": ["..."], "data_type": "categorical_time_series"}], "extraction_strategies": ["Direct fetch", "Found data table in HTML", "SUCCESS: Extracted from HTML table"]}

{"thought": "I have trend data with multiple dimensions. I will visualize this to show the trends clearly.", "actions": [{"tool_name": "Grapher", "tool_input": "Create a multi-line or grouped chart showing trends by category over time. Use appropriate colors and labels."}]}

Observation: [Grapher] {"status": "success", "plot_metadata": {"file_path": "./outputs/plot_trends.png"}}

{"thought": "I have completed the full analysis with data and a visualization.", "actions": [{"tool_name": "final_answer", "tool_input": "I've analyzed the [topic] trends: [summary]. The visualization has been saved as 'plot_trends.png' and shows [key visual insights]. The data reveals [main patterns and conclusions]."}]}"""
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
- "Show me X over time" -> Search for historical/time-series data
- "Compare X and Y" -> Search for datasets containing both variables
- "Current X statistics" -> Search for real-time or recent data
- "Analyze X trends" -> Search for historical data with multiple time points

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


# --- Step 4: Worker Construction ---
def create_worker(
    llm: AbstractChatModel,
    tools: List[AbstractTool],
    stateless: bool = True,
) -> SimpleAgent:
    """Build an ordinary stateless worker agent around its own tools.

    This is the same construction a standalone agent uses; nothing about
    a worker is manager-specific until WorkerAgentTool wraps it. Workers
    are stateless so each delegation is planned fresh, not against the
    history of previous delegations.
    """
    tool_registry = ToolRegistry()
    for tool in tools:
        tool_registry.register_tool(tool)

    planner = ReActPlanner(llm, tool_registry)
    executor = ToolExecutor(tool_registry)
    memory = WorkingMemory()

    return SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=memory,
        stateless=stateless
    )


# --- Step 5: Main Function ---
async def main():
    # check if the web search tool can be used
    if not settings.search_engine.google_cse_search_api or not settings.search_engine.google_cse_search_engine_id:
        print("A google search engine API key as well as search engine ID needs to be set to run this demo. Exiting...")
        return

    print("Initializing fairlib.core.components...")

    # The manager drives a three-worker fan-out under the strict batch JSON
    # contract; the 0.5b model emits parseable turns too rarely to hold that
    # contract, so the default is the 3b. Set FAIR_LLM_DEMO_MODEL to try
    # another model.
    llm = HuggingFaceAdapter(
        os.environ.get("FAIR_LLM_DEMO_MODEL", "dolphin3-qwen25-3b")
    )

    web_search_config = {
        "google_api_key": settings.search_engine.google_cse_search_api,
        "google_search_engine_id": settings.search_engine.google_cse_search_engine_id,
        "cache_ttl": settings.search_engine.web_search_cache_ttl,
        "cache_max_size": settings.search_engine.web_search_cache_max_size,
        "max_results": settings.search_engine.web_search_max_results,
    }

    # Ordinary specialist agents; the whole team shares one loaded model.
    researcher = create_worker(llm, [WebSearcherTool(config=web_search_config)])

    data_extractor = create_worker(llm, [WebDataExtractor(llm=llm)])

    grapher = create_worker(llm, [GraphingTool(
        security_manager=BasicSecurityManager(),
        llm=llm,
        output_dir="./outputs"
    )])

    # Wrap each worker as a typed tool. The capability-derived description
    # is what the manager model reads in its rendered tool catalog.
    # READ_ONLY on the Researcher and DataExtractor is the author's
    # assertion that they only look things up, which lets independent
    # delegations to them run concurrently in one turn. The Grapher keeps
    # the conservative EXTERNAL default because it executes generated
    # plotting code and writes image files to ./outputs, so its
    # delegations run as a sequential barrier.
    worker_tools = [
        WorkerAgentTool(
            researcher,
            name="Researcher",
            description=AgentDescriptionBuilder.build_description(RESEARCHER_CAPABILITY),
            side_effect=SideEffect.READ_ONLY,
        ),
        WorkerAgentTool(
            data_extractor,
            name="DataExtractor",
            description=AgentDescriptionBuilder.build_description(DATA_EXTRACTOR_CAPABILITY),
            side_effect=SideEffect.READ_ONLY,
        ),
        WorkerAgentTool(
            grapher,
            name="Grapher",
            description=AgentDescriptionBuilder.build_description(GRAPHER_CAPABILITY),
        ),
    ]

    capabilities = {
        "Researcher": RESEARCHER_CAPABILITY,
        "DataExtractor": DATA_EXTRACTOR_CAPABILITY,
        "Grapher": GRAPHER_CAPABILITY,
    }

    manager_memory = WorkingMemory()

    # Create and enhance prompt builder with application content only;
    # the manager planner merges its mandatory JSON format instructions.
    prompt_builder = PromptBuilder()
    prompt_builder = enhance_manager_prompt_builder(prompt_builder, capabilities)
    prompt_builder = add_generic_manager_guidance(prompt_builder)
    add_generic_data_extraction_examples(prompt_builder)

    # Add a full worked example of the complete pipeline
    prompt_builder.examples.append(Example(
        """User Request: Find the price of bitcoin over the past 5 years and plot the data.

{"thought": "The user wants historical bitcoin price data and a visualization. This requires: 1) researching data sources, 2) extracting the actual data, 3) creating a plot. Extraction depends on the search results, so I start with a single delegation to the Researcher.", "actions": [{"tool_name": "Researcher", "tool_input": "Search for reliable sources of bitcoin price history over the past 5 years (2019-2024). Find APIs or datasets with daily/weekly price data."}]}

Observation: [Researcher] {"query": "bitcoin price history past 5 years API dataset daily weekly price data", "results": [{"title": "Bitcoin USD (BTC-USD) Price History & Historical Data - Yahoo Finance", "url": "https://finance.yahoo.com/quote/BTC-USD/history", "snippet": "Download historical data for Bitcoin USD. Available in daily, weekly or monthly format back to when Bitcoin USD (BTC-USD) was first traded."}, {"title": "CoinGecko API Documentation", "url": "https://www.coingecko.com/en/api/documentation", "snippet": "Free cryptocurrency API with historical price data. Get bitcoin prices with /coins/{id}/market_chart endpoint."}]}

{"thought": "Found good sources. Yahoo Finance has downloadable historical data. I will extract the bitcoin price data from there.", "actions": [{"tool_name": "DataExtractor", "tool_input": "Extract bitcoin historical price data from https://finance.yahoo.com/quote/BTC-USD/history - get daily closing prices for the past 5 years (2019-2024)"}]}

Observation: [DataExtractor] {"status": "success", "extracted_data": [{"source_url": "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range", "data_type": "time_series", "columns": ["date", "price_usd"], "rows": [["2019-01-01", "3746.71"], ["2020-01-01", "7200.17"], ["2021-01-01", "29374.15"], ["2022-01-01", "47686.81"], ["2023-01-01", "16547.91"], ["2024-01-01", "42265.33"]], "metadata": {"data_points": 1827, "source": "CoinGecko API"}}], "summary": "Successfully extracted 1,827 daily price points for bitcoin from January 2019 to January 2024."}

{"thought": "The DataExtractor successfully retrieved the price data. Now I need to create a visualization. I will delegate to the Grapher with specific instructions for the plot.", "actions": [{"tool_name": "Grapher", "tool_input": "Create a line plot of bitcoin prices over the past 5 years using the extracted data. Use these specifications: 1) X-axis: dates from 2019-2024, 2) Y-axis: price in USD (use log scale if needed), 3) Title: 'Bitcoin Price History (2019-2024)', 4) Add grid lines and format prices with commas, 5) Save the plot with high resolution."}]}

Observation: [Grapher] {"status": "success", "plot_metadata": {"file_path": "./outputs/plot_20240627_103045.png", "file_name": "plot_20240627_103045.png", "plot_type": "line", "title": "Bitcoin Price History (2019-2024)"}, "data_analysis": {"suggested_plot_type": "line", "has_time_series": true, "num_rows": 1827}}

{"thought": "I have coordinated all workers to complete the task. The Researcher found data sources, the DataExtractor retrieved the price history, and the Grapher created a visualization. I can now provide the final answer.", "actions": [{"tool_name": "final_answer", "tool_input": "I've analyzed bitcoin's price over the past 5 years and created a visualization. Starting at $3,747 (Jan 2019), it reached an all-time high of $69,000 (Nov 2021) before crashing to $17,600 (June 2022); current price ~$42,265, a 1,028% gain over 5 years. I've created a line plot saved as 'plot_20240627_103045.png' covering 1,827 daily price points on a logarithmic scale, which highlights the 2020-2021 bull run, the 2022 bear market, and the 2023-2024 recovery."}]}"""
    ))

    # The manager is a plain SimpleAgent over the worker tools: no
    # dedicated orchestrator class, no separate runner. Multi-worker
    # delegation in one turn is an ordinary ToolCallBatch, so independent
    # READ_ONLY delegations are dispatched concurrently while the
    # Grapher's EXTERNAL delegations act as sequential barriers.
    manager = build_worker_manager(
        llm,
        worker_tools,
        prompt_builder=prompt_builder,
        memory=manager_memory,
    )

    # Test query
    user_query = "I want to generate a plot showing the temperature of the earth over the last 10 years."

    print(f"\n{'='*100}")
    print(f"User Query: {user_query}")
    print(f"{'='*100}\n")

    final_answer = await manager.arun(user_query)

    # Display the final result
    print("\n--- FINAL Synthesized Answer ---")
    print(final_answer)
    print(f"\n{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
