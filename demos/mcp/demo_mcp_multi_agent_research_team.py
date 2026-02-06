# demo_mcp_multi_agent_research_team.py
"""
================================================================================
                MCP Multi-Agent Research Team Demo
================================================================================

This demo showcases the power of combining MCP (Model Context Protocol) with
hierarchical multi-agent collaboration to create a research team.

HIGHLIGHTS:
===========
- Demonstrates HYBRID MCP architecture: SSE (remote) + stdio (local) together
- Uses third-party Brave Search MCP server via SSE for web search
- Uses local filesystem MCP server via stdio for file access
- Shows real-world interoperability with external MCP tools

THE RESEARCH TEAM:
==================

1. MANAGER AGENT (Coordinator)
   - Receives complex research requests from the user
   - Breaks down tasks and delegates to specialized workers
   - Synthesizes final answers from worker results

2. RESEARCHER AGENT
   - Specializes in finding information via web search
   - Uses Brave Search MCP tools (via SSE) for independent web search
   - Can also use local filesystem tools for context
   - Returns relevant findings to the manager

3. SUMMARIZER AGENT
   - Specializes in condensing and synthesizing information
   - Takes raw data/findings and produces clear summaries
   - Uses MCP filesystem tools to save reports if needed

NOTE: Brave Search MCP returns text results that can be directly injected into
LLM context, so no separate data extraction agent is needed.

THE MCP ADVANTAGE:
==================
With MCP integration, the team can:
- Access local files for context (reading existing reports, data files)
- Save research outputs to the filesystem
- Connect to any MCP-compatible tool server (databases, APIs, etc.)
- Use remote tools via SSE (Server-Sent Events) transport

PREREQUISITES:
==============
1. Install MCP library:
   pip install mcp

2. Start Brave Search MCP server (SSE):
   docker run -d -p 8080:8080 \\
     -e BRAVE_API_KEY="YOUR_BRAVE_API_KEY" \\
     --name brave-search-mcp \\
     shoofio/brave-search-mcp-sse:latest

   Get your free API key at: https://brave.com/search/api/
   (Free tier: 2,000 queries/month)

3. (Optional) Set environment variable for Brave SSE URL:
   export BRAVE_MCP_SSE_URL="http://localhost:8080/sse"

ARCHITECTURE:
=============
                    +------------------+
                    |  Manager Agent   |
                    +--------+---------+
                             |
                 +-----------+-----------+
                 |                       |
          +------v------+         +------v------+
          |  Researcher |         | Summarizer  |
          +------+------+         +------+------+
                 |                       |
          +------v------+         +------v------+
          | Brave Search|         |  Filesystem |
          |   (SSE)     |         |   (stdio)   |
          +-------------+         +-------------+
               ^                        ^
               |                        |
          Remote Server            Local Server
          (Docker)                 (Python)

EXAMPLE QUERY:
"Research the latest developments in AI agents and provide a summary
 with key insights and trends."
"""
import asyncio
import os
import sys
from typing import Optional, Tuple

from fairlib import (
    settings,
    HuggingFaceAdapter,
    ToolRegistry,
    WebSearcherTool,
    ToolExecutor,
    WorkingMemory,
    SimpleReActPlanner,
    SimpleAgent,
    ManagerPlanner,
    HierarchicalAgentRunner,
    MCPServerConfig,
)

from fairlib.modules.mcp import create_mcp_enhanced_registry
from fairlib.core.prompts import PromptBuilder, RoleDefinition, Example

# Default Brave Search SSE URL (can be overridden via environment variable)
DEFAULT_BRAVE_SSE_URL = "http://localhost:8080/sse"


def create_mcp_tool_prompt_builder():
    """
    Create a prompt builder for agents that use MCP tools.

    MCP tools work exactly like any other tool - the agent just provides
    a simple string as tool_input, and the MCPToolAdapter handles converting
    it to whatever format the MCP server expects.
    """
    builder = PromptBuilder()

    builder.role_definition = RoleDefinition(
        "You are a helpful assistant that uses tools to complete tasks. "
        "You MUST use the available tools to find information. "
        "Always call a tool when you need information you don't have."
    )

    # Note: We do NOT add custom format instructions.
    # The SimpleReActPlanner automatically merges mandatory format instructions.
    # MCP tools use simple string inputs, just like any other tool.

    builder.examples.append(Example(
        "# Example - Web Search:\n"
        "user: What is quantum computing?\n"
        "assistant: "
        "Thought: I need to search the web to find information about quantum computing.\n"
        "Action:\n"
        "tool_name: brave_brave-search_brave_web_search\n"
        "tool_input: what is quantum computing\n"
    ))

    builder.examples.append(Example(
        "# Example - Reading a file:\n"
        "user: Read the README.md file\n"
        "assistant: "
        "Thought: I need to read the README.md file to see its contents.\n"
        "Action:\n"
        "tool_name: fs_filesystem_read_file\n"
        "tool_input: README.md\n"
    ))

    return builder


def create_worker_agent(llm, tools, role_description, use_mcp_registry=None):
    """
    Factory function to create a specialized worker agent.

    Args:
        llm: The language model to use
        tools: List of tool instances for this agent
        role_description: Description of the agent's role (for manager context)
        use_mcp_registry: Optional pre-configured MCP registry to use

    Returns:
        Configured SimpleAgent with the specified tools and role
    """
    tool_registry = ToolRegistry()
    for tool in tools:
        tool_registry.register_tool(tool)

    # If MCP registry provided, we'll combine them
    if use_mcp_registry is not None:
        from fairlib.modules.action.tools.composite_registry import CompositeToolRegistry
        combined_registry = CompositeToolRegistry([tool_registry, use_mcp_registry])
        registry = combined_registry
    else:
        registry = tool_registry

    # Use MCP-optimized prompt builder when MCP tools are available
    # This helps smaller models properly format tool calls with JSON input
    if use_mcp_registry is not None:
        prompt_builder = create_mcp_tool_prompt_builder()
        planner = SimpleReActPlanner(llm, registry, prompt_builder=prompt_builder)
    else:
        planner = SimpleReActPlanner(llm, registry)

    executor = ToolExecutor(registry)
    memory = WorkingMemory()

    agent = SimpleAgent(llm, planner, executor, memory, stateless=True, max_steps=5)
    agent.role_description = role_description
    return agent


async def setup_filesystem_mcp(project_dir: str):
    """
    Set up MCP registry with local filesystem access via stdio transport.

    Uses our Python-based MCP filesystem server (no Node.js required).
    Returns None if MCP is not available.
    """
    mcp_server_script = os.path.join(os.path.dirname(__file__), "mcp_filesystem_server.py")

    try:
        from fairlib.modules.mcp.client.mcp_tool_registry import MCPToolRegistry

        mcp_config = MCPServerConfig(
            name="filesystem",
            transport="stdio",
            command=sys.executable,
            args=[mcp_server_script, project_dir],
            timeout=30
        )

        mcp_registry = MCPToolRegistry(tool_prefix="fs")
        await mcp_registry.add_server(mcp_config)
        tools = list(mcp_registry.get_all_tools().keys())
        print(f"  [stdio] Filesystem MCP tools: {tools}")
        return mcp_registry

    except ImportError:
        print("  [stdio] MCP library not installed. Skipping filesystem tools.")
        return None
    except Exception as e:
        print(f"  [stdio] Could not connect to filesystem MCP server: {e}")
        return None


async def setup_brave_search_sse(url: Optional[str] = None):
    """
    Set up MCP registry with Brave Search via SSE transport.

    Connects to a remote Brave Search MCP server running as a Docker container
    or other SSE-compatible service.

    Args:
        url: SSE endpoint URL. Defaults to BRAVE_MCP_SSE_URL env var or localhost:8080/sse

    Returns:
        MCPToolRegistry with Brave Search tools, or None if connection fails.
    """
    sse_url = url or os.environ.get("BRAVE_MCP_SSE_URL", DEFAULT_BRAVE_SSE_URL)

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
        print(f"  [SSE] Brave Search MCP tools: {tools}")
        return mcp_registry

    except ImportError:
        print("  [SSE] MCP library not installed. Skipping Brave Search.")
        return None
    except Exception as e:
        print(f"  [SSE] Could not connect to Brave Search at {sse_url}: {e}")
        print("        Make sure the Brave Search MCP server is running:")
        print("        docker run -d -p 8080:8080 -e BRAVE_API_KEY=... shoofio/brave-search-mcp-sse:latest")
        return None


async def setup_all_mcp_registries(project_dir: str) -> Tuple[Optional['MCPToolRegistry'], Optional['MCPToolRegistry']]:
    """
    Set up all MCP registries for the demo.

    Returns:
        Tuple of (filesystem_registry, brave_search_registry)
        Either may be None if setup fails.
    """
    print("\nSetting up MCP connections...")
    print("-" * 40)

    # Set up filesystem MCP (stdio - local)
    fs_registry = await setup_filesystem_mcp(project_dir)

    # Set up Brave Search MCP (SSE - remote)
    brave_registry = await setup_brave_search_sse()

    print("-" * 40)

    if fs_registry is None and brave_registry is None:
        print("WARNING: No MCP servers connected. Demo will have limited functionality.")
    else:
        connected = []
        if fs_registry:
            connected.append("Filesystem (stdio)")
        if brave_registry:
            connected.append("Brave Search (SSE)")
        print(f"Connected MCP servers: {', '.join(connected)}")

    return fs_registry, brave_registry


async def main():
    """
    Main function to set up and run the multi-agent research team.
    """
    print("=" * 70)
    print("         MCP Multi-Agent Research Team Demo")
    print("=" * 70)
    print()
    print("This demo showcases HYBRID MCP architecture:")
    print("  - Brave Search via SSE (remote Docker container)")
    print("  - Filesystem via stdio (local Python server)")
    print()

    # --- Step 1: Initialize the LLM ---
    print("Initializing LLM (local HuggingFace model)...")
    print("  (First run may take a moment to download the model)")
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")

    # --- Step 2: Set up MCP registries (hybrid: SSE + stdio) ---
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fs_registry, brave_registry = await setup_all_mcp_registries(project_dir)

    # --- Step 3: Create the Research Team ---
    print("\nBuilding the research team...")

    # Check if we have Brave Search available via MCP
    has_brave_search = brave_registry is not None

    # 3a. RESEARCHER AGENT - Web search specialist (uses Brave Search SSE ONLY)
    # Give Researcher ONLY web search tools to avoid confusion with filesystem
    researcher_tools = []

    # Fallback to Google CSE if no Brave Search
    if not has_brave_search:
        has_google_search = (
            settings.search_engine.google_cse_search_api and
            settings.search_engine.google_cse_search_engine_id
        )
        if has_google_search:
            print("  (Using Google CSE as fallback - Brave Search not available)")
            web_search_config = {
                "google_api_key": settings.search_engine.google_cse_search_api,
                "google_search_engine_id": settings.search_engine.google_cse_search_engine_id,
                "cache_ttl": settings.search_engine.web_search_cache_ttl,
                "cache_max_size": settings.search_engine.web_search_cache_max_size,
                "max_results": settings.search_engine.web_search_max_results,
            }
            researcher_tools.append(WebSearcherTool(config=web_search_config))

    # Researcher gets ONLY Brave Search (SSE) - no filesystem to avoid tool confusion
    researcher = create_worker_agent(
        llm,
        researcher_tools,
        role_description=(
            "A web research specialist. Your ONLY job is to search the web for information. "
            "You have access to web search tools. Use them to find current information, "
            "news, statistics, and facts. Return the search results you find."
        ),
        use_mcp_registry=brave_registry  # ONLY web search, no filesystem
    )
    print("  - Researcher agent ready" + (" (with Brave Search SSE)" if has_brave_search else ""))

    # 3b. SUMMARIZER AGENT - Synthesis and summarization (no tools needed)
    # The summarizer uses the LLM's natural ability to synthesize information.
    # It receives text from the Researcher and produces summaries.
    summarizer = create_worker_agent(
        llm,
        [],  # No tools needed - relies on LLM capabilities
        role_description=(
            "A summarization specialist. You have NO tools available. "
            "Your job is simple: read the task given to you and write a clear summary. "
            "Do NOT try to use any tools. Just write your summary directly as the final answer."
        ),
        use_mcp_registry=None  # No tools needed for summarization
    )
    print("  - Summarizer agent ready")

    # --- Step 4: Create the Manager Agent ---
    workers = {
        "Researcher": researcher,
        "Summarizer": summarizer
    }

    manager_memory = WorkingMemory()
    manager_planner = ManagerPlanner(llm, workers)
    manager_agent = SimpleAgent(llm, manager_planner, None, manager_memory)
    print("  - Manager agent ready")

    # --- Step 5: Create the Team Runner ---
    team = HierarchicalAgentRunner(manager_agent, workers, max_steps=15)

    # --- Step 6: Interactive Mode ---
    print("\n" + "=" * 70)
    print("Research Team is ready!")
    print("=" * 70)
    print("\nMCP Connections:")
    if brave_registry:
        print("  - Brave Search (SSE): Connected at", os.environ.get("BRAVE_MCP_SSE_URL", DEFAULT_BRAVE_SSE_URL))
    else:
        print("  - Brave Search (SSE): Not connected")
    if fs_registry:
        print("  - Filesystem (stdio): Connected")
    else:
        print("  - Filesystem (stdio): Not connected")
    print("\nThe team consists of:")
    print("  - Manager: Coordinates the research effort")
    print("  - Researcher: Searches the web via Brave Search MCP (SSE)")
    print("  - Summarizer: Synthesizes findings into reports")
    print("\nExample queries:")
    print('  - "Research the latest trends in AI and summarize the key developments"')
    print('  - "What is the Model Context Protocol? Summarize its key features."')
    print('  - "What are the top programming languages in 2025? Summarize the trends."')
    print("\nType 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Research Request: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Shutting down research team. Goodbye!")
                break

            print("\n" + "-" * 50)
            print("Manager is coordinating the research team...")
            print("-" * 50 + "\n")

            result = await team.arun(user_input)

            print("\n" + "=" * 50)
            print("FINAL RESEARCH REPORT")
            print("=" * 50)
            print(result)
            print()

        except KeyboardInterrupt:
            print("\n\nShutting down. Goodbye!")
            break
        except Exception as e:
            print(f"\nError during research: {e}\n")

    # Cleanup MCP connections
    print("\nCleaning up MCP connections...")
    if fs_registry:
        await fs_registry.close_all()
        print("  - Filesystem (stdio) closed")
    if brave_registry:
        await brave_registry.close_all()
        print("  - Brave Search (SSE) closed")


async def demo_preset_query():
    """
    Run a preset demo query to showcase the research team capabilities.

    This demo uses a web search query to demonstrate Brave Search SSE integration.
    """
    print("=" * 70)
    print("    MCP Research Team - Preset Demo (SSE + stdio)")
    print("=" * 70)
    print()

    print("Initializing LLM...")
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")

    # Set up hybrid MCP registries
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fs_registry, brave_registry = await setup_all_mcp_registries(project_dir)

    # Create minimal team for demo
    # Researcher gets ONLY Brave Search - no filesystem to avoid tool confusion
    researcher = create_worker_agent(
        llm, [],
        "A web research specialist. Search the web for information and return results.",
        use_mcp_registry=brave_registry  # ONLY web search
    )

    # Summarizer doesn't need tools - just synthesizes information passed to it
    summarizer = create_worker_agent(
        llm, [],
        "A summarization specialist. You have NO tools. Just write a summary directly as your final answer.",
        use_mcp_registry=None  # No tools needed
    )

    workers = {"Researcher": researcher, "Summarizer": summarizer}
    manager_planner = ManagerPlanner(llm, workers)
    manager = SimpleAgent(llm, manager_planner, None, WorkingMemory())
    team = HierarchicalAgentRunner(manager, workers)

    # Choose query based on available tools
    if brave_registry:
        query = (
            "Search the web for the latest developments in AI agents and "
            "provide a brief summary of the key trends."
        )
    else:
        query = (
            "Read the README.md file from our project and provide a brief summary "
            "of what the FAIR-LLM framework is designed to do."
        )

    print(f"\nQuery: {query}\n")
    print("-" * 50)

    result = await team.arun(query)

    print("\n" + "=" * 50)
    print("RESULT:")
    print("=" * 50)
    print(result)

    # Cleanup
    if fs_registry:
        await fs_registry.close_all()
    if brave_registry:
        await brave_registry.close_all()


if __name__ == "__main__":
    # Run interactive mode by default
    # Use demo_preset_query() for a quick non-interactive demo
    asyncio.run(main())
