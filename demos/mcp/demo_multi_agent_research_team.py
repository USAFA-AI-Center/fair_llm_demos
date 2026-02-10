# demo_multi_agent_research_team_showcase.py
"""
================================================================================
     FAIR-LLM Showcase: Multi-Agent Research Team (with MCP Integration)
================================================================================

PURPOSE:
    This demo is a TEACHING TOOL designed to walk students through every major
    class available in the FAIR-LLM framework.

    By the end of this demo, you will understand how to:
    - Build custom prompts with the PromptBuilder system
    - Create specialized worker agents with focused tool sets
    - Orchestrate a hierarchical manager-worker team
    - Connect to external MCP servers for tool interoperability
    - Use someone ELSE's tools (like Brave Search) through MCP

THE RESEARCH TEAM:
==================
    Manager  (coordinates the effort, delegates sub-tasks)
       |
       +---> Researcher  (web search via MCP / Google CSE fallback)
       +---> Analyst     (mathematical calculations)
       +---> Writer      (synthesizes findings into a report, no tools)

MCP VALUE PROPOSITION:
======================
    The Researcher agent does NOT implement its own web search. Instead, it
    connects to someone else's Brave Search MCP server running in a Docker
    container. This demonstrates INTEROPERABILITY: you can leverage any
    MCP-compatible tool without writing a single line of integration code.

    If Brave Search is unavailable, the demo gracefully falls back to the
    built-in WebSearcherTool (Google CSE). If neither is available, only the
    Analyst and Writer workers will be active.

PREREQUISITES:
==============
    Optional (for MCP web search):
        pip install mcp
        docker run -d -p 8080:8080 \\
          -e BRAVE_API_KEY="YOUR_KEY" \\
          --name brave-search-mcp \\
          shoofio/brave-search-mcp-sse:latest

    Optional (for Google CSE fallback):
        Set google_cse_search_api and google_cse_search_engine_id in
        fairlib/config/settings.yml

RUN:
    python demos/demo_multi_agent_research_team_showcase.py

================================================================================
"""
import asyncio
import os
import sys
from typing import Optional

# ==============================================================================
# SECTION 1: THE FAIRLIB IMPORT CATALOG
# ==============================================================================
# Everything below can be imported directly from `fairlib`. This single import
# point is powered by lazy-loading (see fairlib/__init__.py)
#
# We organize them here by category so you can see the full toolkit at a glance.

# --- 1a. Configuration ---
# `settings` is a validated Pydantic object loaded from fairlib/config/settings.yml.
# It gives you access to API keys, model configs, search engine settings, etc.
from fairlib import settings

# --- 1b. Core Data Types ---
# These are the fundamental data structures that flow through every component.
# `Message` is the universal currency of communication between agents and LLMs.
# `Thought`, `Action`, `Observation`, `FinalAnswer` are the ReAct loop primitives.
# `Document` is used in RAG pipelines for chunked text with metadata.
from fairlib import Message, Thought, Action, Observation, FinalAnswer, Document

# --- 1c. Prompt Engineering ---
# The PromptBuilder system lets you construct structured prompts from composable
# pieces. Each piece is a `PromptItem` subclass that renders to a string.
from fairlib import (
    PromptBuilder,          # The main builder that assembles all pieces
    RoleDefinition,         # Defines who the agent IS (its persona/goal)
    ToolInstruction,        # Describes a single available tool
    WorkerInstruction,      # Describes a single available worker agent
    FormatInstruction,      # Rules for how the LLM should format output
    Example,                # Few-shot examples to guide behavior
    AgentCapability,        # Structured description of what an agent can do
)

# --- 1d. Agent Classes ---
# `SimpleAgent` is the core ReAct agent that thinks, acts, and observes in a loop.
# `HierarchicalAgentRunner` orchestrates a manager + multiple worker agents.
from fairlib import SimpleAgent, HierarchicalAgentRunner

# --- 1e. Planners ---
# Planners are the "brain" — they take history and produce the next Thought+Action.
# `ReActPlanner`       - Standard planner using JSON format (for capable models)
# `SimpleReActPlanner` - Lightweight planner using text key-value format (for small models)
# `ManagerPlanner`     - Specialized planner that only delegates or gives final answers
from fairlib import ReActPlanner, SimpleReActPlanner, ManagerPlanner

# --- 1f. Memory ---
# Memory systems store conversation history and retrieved context.
# `WorkingMemory` is short-term, in-context memory (most common for demos).
# `LongTermMemory` is RAG-backed memory using a vector store + retriever.
from fairlib import WorkingMemory

# --- 1g. Model Abstraction Layer (MAL) ---
# The MAL lets you swap LLM providers without changing any agent code.
# Each adapter conforms to the same `AbstractChatModel` interface.
#   OpenAIAdapter      - GPT-4, GPT-3.5, etc.
#   AnthropicAdapter   - Claude 3, Claude 3.5, etc.
#   HuggingFaceAdapter - Local transformer models (v4 AND v5 compatible)
#   OllamaAdapter      - Local Ollama models
#   LoadBalancerAdapter - Distributes requests across multiple adapters
from fairlib import HuggingFaceAdapter

# --- 1h. Tool Components ---
# Tools give agents the ability to interact with the world.
# `ToolRegistry` holds a collection of tools.
# `ToolExecutor` runs tools by name and returns results.
from fairlib import ToolRegistry, ToolExecutor

# --- 1i. Built-in Tools ---
# Ready-to-use tools that ship with fairlib.
#   SafeCalculatorTool  - AST-based safe math evaluation
#   AdvancedCalculusTool - Calculus operations (integrals, derivatives)
#   WebSearcherTool     - Google Custom Search Engine integration
#   WeatherTool         - Weather data retrieval
#   GraphingTool        - Creates matplotlib visualizations
#   WebDataExtractor    - Structured data extraction from web pages
#   CodeExecutionTool   - Sandboxed Python code execution
#   KnowledgeBaseQueryTool - RAG query interface
#   GradeEssayFromRubricTool  - Essay autograding
#   GradeCodeFromRubricTool   - Code autograding
from fairlib import SafeCalculatorTool, WebSearcherTool

# --- 1j. MCP (Model Context Protocol) ---
# MCP lets your agents use tools hosted on EXTERNAL servers.
# This is the key to INTEROPERABILITY — you can plug in anyone's tools.
#   MCPClient          - Low-level connection to an MCP server
#   MCPToolAdapter     - Wraps an MCP tool to look like a local AbstractTool
#   MCPToolRegistry    - Manages tools from multiple MCP servers
#   MCPServerConfig    - Pydantic config for connecting to an MCP server
#   CompositeToolRegistry - Merges local tools + MCP tools into one registry
from fairlib import MCPServerConfig, CompositeToolRegistry

# --- 1k. Advanced Prompt Components (imported from sub-module) ---
# These are specialized prompt items for manager agents.
from fairlib.core.prompts import (
    ManagerPromptBuilder,       # Builder optimized for manager delegation prompts
    StrictFormatInstruction,    # Explicit DO/DON'T rules for smaller models
    DelegationExample,          # Single-turn delegation example for few-shot
    EnhancedWorkerInstruction,  # Rich worker descriptions with capability details
)


# ==============================================================================
# SECTION 2: HELPER FUNCTIONS
# ==============================================================================

def print_section(title: str, width: int = 70):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step(step_num: int, description: str):
    """Print a numbered step."""
    print(f"\n  [{step_num}] {description}")
    print("  " + "-" * 50)


async def setup_brave_search_mcp(url: Optional[str] = None):
    """
    Attempt to connect to a Brave Search MCP server via SSE.

    This demonstrates using someone ELSE's tool through MCP.
    The Brave Search server is a Docker container that exposes web search
    as an MCP tool — our agent doesn't need to know anything about the
    Brave API; it just sends a search query and gets results back.

    Returns:
        MCPToolRegistry if successful, None otherwise.
    """
    sse_url = url or os.environ.get("BRAVE_MCP_SSE_URL", "http://localhost:8080/sse")

    try:
        from fairlib.modules.mcp.client.mcp_tool_registry import MCPToolRegistry

        mcp_config = MCPServerConfig(
            name="brave-search",
            transport="sse",
            url=sse_url,
            timeout=30
        )

        mcp_registry = MCPToolRegistry(tool_prefix="brave")
        await mcp_registry.add_server(mcp_config)
        tools = list(mcp_registry.get_all_tools().keys())
        print(f"    Connected to Brave Search MCP (SSE): {tools}")
        return mcp_registry

    except ImportError:
        print("    MCP library not installed (pip install mcp). Skipping.")
        return None
    except Exception as e:
        print(f"    Could not connect to Brave Search at {sse_url}: {e}")
        return None


def create_worker_agent(
    llm,
    tools,
    role_description: str,
    use_simple_planner: bool = False,
    mcp_registry=None,
    max_steps: int = 5,
):
    """
    Factory function to create a specialized worker agent.

    This function demonstrates the standard pattern for building an agent:
        1. Create a ToolRegistry and register tools
        2. (Optional) Combine with MCP tools via CompositeToolRegistry
        3. Create a Planner with the registry
        4. Create a ToolExecutor with the registry
        5. Create WorkingMemory
        6. Assemble into a SimpleAgent

    Args:
        llm:                The language model adapter (any MAL adapter works)
        tools:              List of local tool instances
        role_description:   What this agent does (shown to the manager)
        use_simple_planner: If True, use SimpleReActPlanner (better for small models)
        mcp_registry:       Optional MCPToolRegistry to merge with local tools
        max_steps:          Max reasoning steps before the agent gives up
    """
    # Step 1: Create a local tool registry
    local_registry = ToolRegistry()
    for tool in tools:
        local_registry.register_tool(tool)

    # Step 2: Optionally merge with MCP tools
    if mcp_registry is not None:
        registry = CompositeToolRegistry([local_registry, mcp_registry])
    else:
        registry = local_registry

    # Step 3: Create the planner
    if use_simple_planner:
        planner = SimpleReActPlanner(llm, registry)
    else:
        planner = ReActPlanner(llm, registry)

    # Step 4: Create the executor
    executor = ToolExecutor(registry)

    # Step 5: Create memory
    memory = WorkingMemory()

    # Step 6: Assemble the agent
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=memory,
        max_steps=max_steps,
        stateless=True,  # Workers clear memory between tasks
    )
    agent.role_description = role_description
    return agent


# ==============================================================================
# SECTION 3: BUILDING THE RESEARCH TEAM
# ==============================================================================

async def build_research_team(llm):
    """
    Construct a 3-worker research team with a manager coordinator.

    This function demonstrates:
    - Creating agents with different tool configurations
    - MCP integration for external web search
    - Graceful fallback when MCP is unavailable
    - ManagerPlanner with custom PromptBuilder
    - AgentCapability for structured worker descriptions
    - HierarchicalAgentRunner for orchestration
    """
    print_section("BUILDING THE RESEARCH TEAM")

    # ------------------------------------------------------------------
    # Step 1: Set up MCP for the Researcher's web search
    # ------------------------------------------------------------------
    print_step(1, "Connecting to external MCP servers")
    print("    Attempting to connect to Brave Search MCP server...")
    print("    (This uses someone ELSE's web search tool via MCP!)")

    brave_registry = await setup_brave_search_mcp()

    # Determine what search capability we have
    has_brave_search = brave_registry is not None
    has_google_search = (
        settings.search_engine.google_cse_search_api
        and settings.search_engine.google_cse_search_engine_id
    )

    search_source = "none"
    researcher_tools = []
    researcher_mcp = None

    if has_brave_search:
        search_source = "Brave Search (MCP/SSE)"
        researcher_mcp = brave_registry
    elif has_google_search:
        search_source = "Google CSE (local tool)"
        web_search_config = {
            "google_api_key": settings.search_engine.google_cse_search_api,
            "google_search_engine_id": settings.search_engine.google_cse_search_engine_id,
            "cache_ttl": settings.search_engine.web_search_cache_ttl,
            "cache_max_size": settings.search_engine.web_search_cache_max_size,
            "max_results": settings.search_engine.web_search_max_results,
        }
        researcher_tools.append(WebSearcherTool(config=web_search_config))
    else:
        print("    WARNING: No search capability available.")
        print("    The Researcher will operate without web search tools.")

    print(f"    Search source: {search_source}")

    # ------------------------------------------------------------------
    # Step 2: Define AgentCapability for each worker
    # ------------------------------------------------------------------
    # AgentCapability provides a STRUCTURED way to describe what each worker
    # can do. This is used to generate rich worker instructions for the
    # manager's prompt and can also be used for automatic agent selection.
    print_step(2, "Defining agent capabilities (AgentCapability)")

    researcher_capability = AgentCapability(
        name="Researcher",
        primary_function="Searches the web to find current, real-time information.",
        capabilities=[
            "Find current prices, statistics, and data",
            "Look up recent news and developments",
            "Discover facts and background information",
        ],
        limitations=[
            "Cannot perform calculations",
            "Cannot write long-form content",
        ],
        input_format="A clear search query or research question",
        output_format="Raw search results and key findings",
        example_tasks=[
            "Find the current price of Bitcoin",
            "What are the latest trends in AI agents?",
        ],
        delegation_keywords=["search", "find", "look up", "research", "current"],
        tools=["web_search"],
    )

    analyst_capability = AgentCapability(
        name="Analyst",
        primary_function="Performs mathematical calculations and data analysis.",
        capabilities=[
            "Arithmetic operations (add, subtract, multiply, divide)",
            "Percentage calculations",
            "Unit conversions",
        ],
        limitations=[
            "Cannot search the web",
            "Cannot write reports",
        ],
        input_format="A mathematical expression or calculation request",
        output_format="Numerical result with explanation",
        example_tasks=[
            "Calculate 5000 / 67432.50",
            "What is 15% of 250?",
        ],
        delegation_keywords=["calculate", "compute", "math", "how many", "percentage"],
        tools=["safe_calculator"],
    )

    writer_capability = AgentCapability(
        name="Writer",
        primary_function="Synthesizes information into clear, well-organized reports.",
        capabilities=[
            "Summarize complex information",
            "Write structured reports with sections",
            "Combine findings from multiple sources",
        ],
        limitations=[
            "Cannot search the web",
            "Cannot perform calculations",
            "Has NO tools — relies entirely on the LLM",
        ],
        input_format="Raw findings and data to synthesize",
        output_format="A clear, organized written summary or report",
        example_tasks=[
            "Summarize the research findings into a brief report",
            "Write a comparison of two datasets",
        ],
        delegation_keywords=["summarize", "write", "report", "synthesize", "explain"],
        tools=[],
    )

    # Print the detailed capability descriptions
    for cap in [researcher_capability, analyst_capability, writer_capability]:
        print(f"\n    {cap.to_detailed_description()[:80]}...")

    # ------------------------------------------------------------------
    # Step 3: Create the worker agents
    # ------------------------------------------------------------------
    print_step(3, "Creating specialized worker agents")

    # RESEARCHER: Uses MCP web search OR local Google CSE
    researcher = create_worker_agent(
        llm,
        researcher_tools,
        role_description=(
            "A web research specialist. Search the web ONCE, then immediately "
            "return the results as your final answer. Do NOT search multiple times. "
            "One search is enough — return what you find."
        ),
        mcp_registry=researcher_mcp,
        max_steps=3,
    )
    researcher.capability = researcher_capability
    print(f"    Researcher agent ready [{search_source}]")

    # ANALYST: Uses SafeCalculatorTool for math
    analyst = create_worker_agent(
        llm,
        [SafeCalculatorTool()],
        role_description=(
            "A mathematical analyst. You have ONLY one tool: safe_calculator. "
            "Use it to evaluate math expressions. Do NOT try to use any other tool. "
            "If given a number to calculate, pass the expression to safe_calculator "
            "and return the result as your final answer."
        ),
        max_steps=3,
    )
    analyst.capability = analyst_capability
    print("    Analyst agent ready [SafeCalculatorTool]")

    # WRITER: No tools — relies on the LLM's own writing ability
    writer = create_worker_agent(
        llm,
        [],  # No tools!
        role_description=(
            "A writing specialist. You have NO tools available. "
            "Your job is to take information given to you and write a clear, "
            "organized summary or report. Just write your response directly "
            "as the final answer — do NOT try to use any tools."
        ),
    )
    writer.capability = writer_capability
    print("    Writer agent ready [no tools, LLM only]")

    # ------------------------------------------------------------------
    # Step 4: Build a custom ManagerPromptBuilder
    # ------------------------------------------------------------------
    # The ManagerPromptBuilder is a specialized PromptBuilder that adds
    # strict formatting rules for smaller models. It ensures the manager
    # only outputs "delegate" or "final_answer" actions.
    print_step(4, "Building the manager's prompt (ManagerPromptBuilder)")

    workers = {
        "Researcher": researcher,
        "Analyst": analyst,
        "Writer": writer,
    }

    # We use strict_mode=False so we can add StrictFormatInstruction items
    # MANUALLY below. With strict_mode=True, the builder adds them automatically
    # behind the scenes — but we want students to see every class in action.
    manager_builder = ManagerPromptBuilder(strict_mode=False)
    manager_builder.set_role(
        "You are the manager of a research team. Your job is to break down "
        "complex user requests into sub-tasks and delegate them to the right "
        "worker. You do NOT perform tasks yourself — you coordinate the team. "
        "Always delegate research tasks to the Researcher, math tasks to the "
        "Analyst, and writing/summary tasks to the Writer."
    )
    manager_builder.set_workflow(["Researcher", "Analyst", "Writer"])

    # Add enhanced worker instructions from capabilities
    for cap in [researcher_capability, analyst_capability, writer_capability]:
        manager_builder.worker_instructions.append(
            EnhancedWorkerInstruction(cap)
        )

    # --- StrictFormatInstruction ---
    # These provide explicit DO/DON'T formatting rules that help smaller models
    # (3B-7B) produce correctly structured output. Each static method returns
    # a pre-built instruction targeting a specific failure mode.
    #
    #   .json_output_rules()        - "Output flat JSON, no code fences"
    #   .delegation_rules(workers)  - "Only use delegate or final_answer"
    #   .correct_format_example()   - "Correct vs incorrect format examples"

    manager_builder.format_instructions.append(
        StrictFormatInstruction.json_output_rules()
    )
    manager_builder.format_instructions.append(
        StrictFormatInstruction.delegation_rules(["Researcher", "Analyst", "Writer"])
    )
    manager_builder.format_instructions.append(
        StrictFormatInstruction.correct_format_example()
    )
    # Also add the workflow guidance (normally auto-added by strict_mode)
    manager_builder.add_workflow_guidance()

    # Add delegation examples using DelegationExample directly.
    # DelegationExample is a subclass of Example that creates single-turn
    # few-shot examples showing the manager how to delegate. Each example
    # shows ONE decision point: a context, a thought, and an action.
    #
    # There are three static factory methods for common patterns:
    #   .initial_delegation()   - First step: user request → delegate
    #   .followup_delegation()  - Middle step: observation → delegate next
    #   .final_answer_example() - Last step: observation → final answer

    # Example 1: The manager receives the user request and delegates to Researcher
    manager_builder.examples.append(
        DelegationExample.initial_delegation(
            user_request="Find the current price of gold and calculate how many ounces I can buy with $10,000",
            thought="I need to find the price of gold first. The Researcher can search for this.",
            worker_name="Researcher",
            task="Find the current price of one ounce of gold in USD",
        )
    )

    # Example 2: The manager receives the Researcher's result and delegates to Analyst
    manager_builder.examples.append(
        DelegationExample.followup_delegation(
            observation="Result from Researcher: Gold is currently $2,350 per ounce.",
            thought="Now I need the Analyst to calculate how many ounces $10,000 can buy.",
            worker_name="Analyst",
            task="Calculate 10000 / 2350",
        )
    )

    # Example 3: The manager has all the information and provides the final answer
    manager_builder.examples.append(
        DelegationExample.final_answer_example(
            observation="Result from Analyst: 10000 / 2350 = 4.2553",
            thought="I have all the information. Let me provide the final answer.",
            answer="Based on the current price of $2,350/oz, you can buy approximately 4.26 ounces of gold with $10,000.",
        )
    )

    # Preview the built prompt (for educational purposes)
    preview = manager_builder.build_system_prompt_string()
    print(f"\n    Manager prompt preview ({len(preview)} chars):")
    # Show first few lines
    for line in preview.split("\n")[:6]:
        print(f"      | {line}")
    print("      | ...")

    # ------------------------------------------------------------------
    # Step 5: Create the Manager Agent with ManagerPlanner
    # ------------------------------------------------------------------
    print_step(5, "Creating the Manager agent (ManagerPlanner)")

    # The ManagerPlanner takes our custom builder and automatically merges
    # mandatory format instructions (ensuring the parser always works).
    manager_planner = ManagerPlanner(
        llm=llm,
        workers=workers,
        prompt_builder=manager_builder,  # Our custom builder!
    )

    manager_agent = SimpleAgent(
        llm=llm,
        planner=manager_planner,
        tool_executor=None,  # Manager never executes tools directly
        memory=WorkingMemory(),
    )
    print("    Manager agent ready [ManagerPlanner + custom prompt]")

    # ------------------------------------------------------------------
    # Step 6: Assemble the team with HierarchicalAgentRunner
    # ------------------------------------------------------------------
    print_step(6, "Assembling the team (HierarchicalAgentRunner)")

    team = HierarchicalAgentRunner(
        manager_agent=manager_agent,
        workers=workers,
        max_steps=15,
    )
    print("    Team assembled!")

    return team, brave_registry


# ==============================================================================
# SECTION 4: RUNNING THE DEMO
# ==============================================================================

async def run_preset_demo(team):
    """Run a preset query to showcase the team in action."""
    print_section("RUNNING PRESET DEMO QUERY")

    query = (
        "I have a budget of $5,000. Find the current price of Bitcoin "
        "and calculate exactly how many Bitcoins I can afford. "
        "Then write a brief summary of the investment."
    )

    print(f"\n  Query: {query}\n")
    print("-" * 70)

    result = await team.arun(query)

    print("\n" + "=" * 70)
    print("  FINAL RESEARCH REPORT")
    print("=" * 70)
    print(result)
    return result


async def run_interactive(team):
    """Run in interactive mode, accepting queries from the user."""
    print_section("INTERACTIVE MODE")
    print("\n  The research team is ready for your queries!")
    print("  The team consists of:")
    print("    - Manager:    Coordinates the research effort")
    print("    - Researcher: Searches the web for information")
    print("    - Analyst:    Performs mathematical calculations")
    print("    - Writer:     Synthesizes findings into reports")
    print("\n  Example queries:")
    print('    - "Find the price of Ethereum and calculate how many I can buy with $2,000"')
    print('    - "Research the latest AI trends and write a brief summary"')
    print('    - "What is 15% of 8,500?"')
    print("\n  Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("  Research Request: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("  Shutting down. Goodbye!")
                break

            print("\n" + "-" * 70)
            result = await team.arun(user_input)

            print("\n" + "=" * 70)
            print("  FINAL RESEARCH REPORT")
            print("=" * 70)
            print(result)
            print()

        except KeyboardInterrupt:
            print("\n\n  Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n  Error: {e}\n")


async def main():
    """
    Main entry point for the Multi-Agent Research Team Showcase.
    """
    print_section("FAIR-LLM SHOWCASE: Multi-Agent Research Team")
    print("""
  This demo walks you through EVERY major class in the FAIR-LLM framework
  while building a functional 3-worker research team.

  Classes demonstrated:
    Core:      Message, Thought, Action, FinalAnswer, Document, settings
    Prompts:   PromptBuilder, ManagerPromptBuilder, RoleDefinition,
               FormatInstruction, Example, AgentCapability,
               StrictFormatInstruction, DelegationExample,
               EnhancedWorkerInstruction, WorkerInstruction, ToolInstruction
    Agents:    SimpleAgent, HierarchicalAgentRunner
    Planners:  SimpleReActPlanner, ManagerPlanner
    Memory:    WorkingMemory
    MAL:       HuggingFaceAdapter (supports transformers v4 AND v5)
    Tools:     ToolRegistry, ToolExecutor, SafeCalculatorTool, WebSearcherTool
    MCP:       MCPServerConfig, CompositeToolRegistry, MCPToolRegistry
    """)

    # ------------------------------------------------------------------
    # Initialize the LLM
    # ------------------------------------------------------------------
    print_step(0, "Initializing the LLM (HuggingFaceAdapter)")
    print("    Using local HuggingFace model via the Model Abstraction Layer.")
    print("    (You could swap this for OpenAIAdapter, AnthropicAdapter, or")
    print("     OllamaAdapter without changing ANY agent code.)")

    # Qwen 2.5 14B Instruct: strong instruction following and JSON output,
    # which is critical for the manager's structured delegation format.
    # Requires ~28 GB VRAM in fp16 (fits on A6000/A100/etc.).
    # For smaller GPUs, try "qwen25-7b" (~14 GB) or "dolphin3-qwen25-3b" (~6 GB).
    #
    # max_new_tokens=512 gives the model enough room to generate complete
    # JSON actions. The default (256) is too short when the system prompt
    # plus conversation history is long.
    llm = HuggingFaceAdapter("qwen25-14b", max_new_tokens=512)
    print(f"    LLM ready: {llm.model_name}")

    # ------------------------------------------------------------------
    # Build and run the team
    # ------------------------------------------------------------------
    team, brave_registry = await build_research_team(llm)

    # Choose mode based on command-line args
    if "--preset" in sys.argv:
        await run_preset_demo(team)
    else:
        await run_interactive(team)

    # ------------------------------------------------------------------
    # Cleanup MCP connections
    # ------------------------------------------------------------------
    if brave_registry:
        print("\n  Cleaning up MCP connections...")
        await brave_registry.close_all()
        print("    Brave Search (SSE) closed.")

    print_section("DEMO COMPLETE")
    print("""
  KEY TAKEAWAYS:
  ==============
  1. FAIRLIB IMPORTS: Everything comes from `from fairlib import ...`
  2. MAL LAYER:       Swap LLM providers without changing agent code
  3. PROMPTBUILDER:   Compose prompts from structured, reusable pieces
  4. AGENTS:          SimpleAgent is the workhorse; workers are stateless
  5. MULTI-AGENT:     ManagerPlanner + HierarchicalAgentRunner = team
  6. MCP:             Use anyone's tools via MCPServerConfig + SSE/stdio
  7. GRACEFUL:        Always fall back when external services are unavailable
    """)


if __name__ == "__main__":
    asyncio.run(main())
