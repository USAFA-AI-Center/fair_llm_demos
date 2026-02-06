# demo_mcp_agent_tool_calling.py
"""
================================================================================
            MCP Agent Tool Calling Demo - SSE + stdio
================================================================================

This demo showcases agents automatically invoking MCP tools to complete tasks.
It demonstrates the full agentic loop:

1. Agent receives a query
2. Agent reasons about what tool to use
3. Agent calls MCP tool (via SSE or stdio)
4. Agent receives tool observation
5. Agent synthesizes final answer

ARCHITECTURE:
=============
    +-----------------+
    |   User Query    |
    +--------+--------+
             |
    +--------v--------+
    |  Research Agent |
    |  (ReAct Loop)   |
    +--------+--------+
             |
    +--------v--------+     +------------------+
    | Tool Decision   |---->| Brave Search     |
    | (Thought/Action)|     | (SSE - Remote)   |
    +-----------------+     +------------------+
             |
    +--------v--------+     +------------------+
    | Tool Execution  |---->| Filesystem       |
    |                 |     | (stdio - Local)  |
    +-----------------+     +------------------+
             |
    +--------v--------+
    |  Final Answer   |
    +-----------------+

PREREQUISITES:
==============
1. pip install mcp
2. Start Brave Search MCP server:
   docker run -d -p 8080:8080 -e BRAVE_API_KEY="YOUR_KEY" \\
     --name brave-search-mcp shoofio/brave-search-mcp-sse:latest
"""
import asyncio
import os
import sys

from fairlib import (
    HuggingFaceAdapter,
    ToolExecutor,
    WorkingMemory,
    SimpleReActPlanner,
    SimpleAgent,
    MCPServerConfig,
)
from fairlib.core.prompts import PromptBuilder, RoleDefinition, Example

# Default Brave Search SSE URL
DEFAULT_BRAVE_SSE_URL = "http://localhost:8080/sse"


def create_research_agent_prompt_builder():
    """
    Create a prompt builder for a research agent.

    This uses the STANDARD tool_input format (simple strings).
    MCP tools look exactly the same as any other tool to the agent -
    the MCPToolAdapter handles converting simple strings to the
    JSON format that MCP servers expect.
    """
    builder = PromptBuilder()

    builder.role_definition = RoleDefinition(
        "You are a helpful research assistant that uses tools to find information. "
        "You MUST use the available tools to answer questions. "
        "Always call a search tool when you need information from the web."
    )

    # Note: We do NOT add special format instructions here.
    # The SimpleReActPlanner will automatically merge its mandatory
    # format instructions, which use simple string tool_input.

    builder.examples.append(Example(
        "# Example - Web Search:\n"
        "user: What are the latest AI trends?\n"
        "assistant: "
        "Thought: I need to search the web to find the latest AI trends.\n"
        "Action:\n"
        "tool_name: brave_brave-search_brave_web_search\n"
        "tool_input: latest AI trends 2025\n"
    ))

    builder.examples.append(Example(
        "# Example - Reading a file:\n"
        "user: What's in the README file?\n"
        "assistant: "
        "Thought: I need to read the README file to see its contents.\n"
        "Action:\n"
        "tool_name: fs_filesystem_read_file\n"
        "tool_input: README.md\n"
    ))

    return builder


async def setup_mcp_connections():
    """Set up both SSE (Brave Search) and stdio (filesystem) MCP connections."""
    from fairlib.modules.mcp.client.mcp_tool_registry import MCPToolRegistry
    from fairlib.modules.action.tools.composite_registry import CompositeToolRegistry

    print("\n" + "=" * 60)
    print("Setting up MCP Connections")
    print("=" * 60)

    registries = []

    # 1. Set up Brave Search via SSE
    sse_url = os.environ.get("BRAVE_MCP_SSE_URL", DEFAULT_BRAVE_SSE_URL)
    try:
        brave_config = MCPServerConfig(
            name="brave-search",
            transport="sse",
            url=sse_url,
            timeout=30
        )
        brave_registry = MCPToolRegistry(tool_prefix="brave")
        await brave_registry.add_server(brave_config)
        tools = list(brave_registry.get_all_tools().keys())
        print(f"[SSE] Brave Search connected at {sse_url}")
        print(f"      Tools: {tools}")
        registries.append(brave_registry)
    except Exception as e:
        print(f"[SSE] Brave Search not available: {e}")
        brave_registry = None

    # 2. Set up Filesystem via stdio
    mcp_server_script = os.path.join(os.path.dirname(__file__), "mcp_filesystem_server.py")
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        fs_config = MCPServerConfig(
            name="filesystem",
            transport="stdio",
            command=sys.executable,
            args=[mcp_server_script, project_dir],
            timeout=30
        )
        fs_registry = MCPToolRegistry(tool_prefix="fs")
        await fs_registry.add_server(fs_config)
        tools = list(fs_registry.get_all_tools().keys())
        print(f"[stdio] Filesystem connected")
        print(f"        Tools: {tools}")
        registries.append(fs_registry)
    except Exception as e:
        print(f"[stdio] Filesystem not available: {e}")
        fs_registry = None

    print("=" * 60)

    if not registries:
        return None, None, None

    # Combine registries
    if len(registries) > 1:
        combined = CompositeToolRegistry(registries)
    else:
        combined = registries[0]

    return combined, brave_registry, fs_registry


async def run_agent_demo(llm, registry, query: str):
    """Run the agent with a query and show the full agentic loop."""
    print("\n" + "-" * 60)
    print(f"QUERY: {query}")
    print("-" * 60)

    # Create agent with MCP-optimized prompting
    prompt_builder = create_research_agent_prompt_builder()
    planner = SimpleReActPlanner(llm, registry, prompt_builder=prompt_builder)
    executor = ToolExecutor(registry)
    memory = WorkingMemory()
    agent = SimpleAgent(llm, planner, executor, memory, stateless=True, max_steps=5)

    # Run the agent
    result = await agent.arun(query)

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result)
    print()

    return result


async def main():
    """Main demo function."""
    print("=" * 70)
    print("       MCP Agent Tool Calling Demo")
    print("       Demonstrating SSE + stdio MCP Integration")
    print("=" * 70)

    # Initialize LLM
    print("\nLoading LLM (Dolphin-3B)...")
    llm = HuggingFaceAdapter("dolphin3-qwen25-3b")

    # Set up MCP connections
    combined_registry, brave_registry, fs_registry = await setup_mcp_connections()

    if not combined_registry:
        print("\nERROR: No MCP servers available. Please start the Brave Search server:")
        print("  docker run -d -p 8080:8080 -e BRAVE_API_KEY=... shoofio/brave-search-mcp-sse:latest")
        return

    # Show all available tools
    all_tools = combined_registry.get_all_tools()
    print(f"\nAgent has access to {len(all_tools)} tools:")
    for name in all_tools.keys():
        print(f"  - {name}")

    # Demo queries
    queries = [
        "Search the web for what is the Model Context Protocol (MCP)",
        "List the files in the current directory",
    ]

    # Only run queries for available tools
    if brave_registry:
        await run_agent_demo(llm, combined_registry, queries[0])

    if fs_registry:
        await run_agent_demo(llm, combined_registry, queries[1])

    # Interactive mode
    print("\n" + "=" * 70)
    print("Interactive Mode - Type your queries (or 'q' to quit)")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nQuery: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            await run_agent_demo(llm, combined_registry, user_input)
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

    # Cleanup
    print("\nCleaning up MCP connections...")
    if fs_registry:
        await fs_registry.close_all()
    if brave_registry:
        await brave_registry.close_all()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
