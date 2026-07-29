# demo_multi_agent_research_team_showcase.py
"""
FAIR-LLM Showcase: Multi-Agent Research Team (with MCP Integration).

PURPOSE:
    This demo is a TEACHING TOOL designed to walk students through every major
    class available in the FAIR-LLM framework.

    By the end of this demo, you will understand how to:
    - Build custom prompts with the PromptBuilder system
    - Create specialized worker agents with focused tool sets
    - Wrap worker agents as typed tools with WorkerAgentTool
    - Assemble a fan-out manager with build_worker_manager
    - Observe delegation through the typed event bus
    - Connect to external MCP servers for tool interoperability
    - Use someone ELSE's tools (like Brave Search) through MCP

THE RESEARCH TEAM:
    Manager (a plain SimpleAgent whose tools ARE the workers)
       delegates to:
       Researcher  (web search via MCP / Google CSE fallback)
       Analyst     (mathematical calculations)
       Writer      (synthesizes findings into a report, no tools)

    The manager is not a special orchestrator class. Each worker is an
    ordinary stateless SimpleAgent wrapped in a WorkerAgentTool, and the
    manager is a plain SimpleAgent built by build_worker_manager over those
    worker tools. The manager's model reads what each worker is for from
    the rendered tool catalog (the WorkerAgentTool descriptions), so
    delegating a subtask is an ordinary typed tool call. Because all three
    workers here are declared SideEffect.READ_ONLY, independent
    delegations issued in one turn are dispatched concurrently by the
    side-effect-aware executor; dependent subtasks (search, then compute
    on the result, then write it up) still chain naturally across turns.

What it shows:
  - WorkerAgentTool adapts any BaseAgent into a typed tool; the tool
    description is what the manager model reads, and the declared
    SideEffect (READ_ONLY here) is what permits concurrent fan-out.
  - build_worker_manager is pure wiring: MultiActionReActPlanner plus
    ToolExecutor over a registry of worker tools, one shared event bus,
    returning an ordinary SimpleAgent.
  - Workers are ordinary stateless SimpleAgents with their own planners
    and tools - the same agents you would build standalone.
  - Instrumentation rides the shared AgentEventBus: the scheduler's
    grouping decision (ToolBatchScheduledEvent) and per-delegation timing
    (ToolCallPreEvent/ToolCallPostEvent, keyed by (step, call_index)).
  - MCP interoperability: the Researcher does NOT implement its own web
    search; it connects to someone else's Brave Search MCP server. If
    Brave Search is unavailable, the demo gracefully falls back to the
    built-in WebSearcherTool (Google CSE); if neither is available, only
    the Analyst and Writer workers have tools.

A note on stochasticity: a real local model drives the manager and every
worker. Delegation quality and whether independent subtasks fan out in a
single turn depend on the model's output on a given run; the run completes
either way.

PREREQUISITES:
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
    python demos/mcp/demo_multi_agent_research_team.py
"""
import asyncio
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# ==============================================================================
# SECTION 1: THE FAIRLIB IMPORT CATALOG
# ==============================================================================
# Everything below can be imported directly from fairlib. This single import
# point is powered by lazy-loading (see fairlib/__init__.py)
#
# We organize them here by category so you can see the full toolkit at a glance.

# --- 1a. Configuration ---
# settings is a validated Pydantic object loaded from fairlib/config/settings.yml.
# It gives you access to API keys, model configs, search engine settings, etc.
from fairlib import settings

# --- 1b. Core Data Types ---
# These are the fundamental data structures that flow through every component.
# Message is the universal currency of communication between agents and LLMs.
# Thought, Action, Observation, FinalAnswer are the ReAct loop primitives.
# Document is used in RAG pipelines for chunked text with metadata.

# --- 1c. Prompt Engineering ---
# The PromptBuilder system lets you construct structured prompts from composable
# pieces. Each piece is a PromptItem subclass that renders to a string.
# RoleDefinition states who the agent is; the planner merges its own mandatory
# format instructions on top, so application prompt content stays declarative.
from fairlib import PromptBuilder, RoleDefinition

# --- 1d. Agent Classes ---
# SimpleAgent is the core ReAct agent that thinks, acts, and observes in a loop.
# WorkerAgentTool wraps any agent as a typed tool a manager can call.
# build_worker_manager wires a fan-out manager as a plain SimpleAgent whose
# tools are the workers.
from fairlib import SimpleAgent, WorkerAgentTool, build_worker_manager

# --- 1e. Planners ---
# Planners are the "brain" — they take history and produce the next Thought+Action.
# ReActPlanner       - Standard planner using JSON format (for capable models)
# SimpleReActPlanner - Lightweight planner using text key-value format (for small models)
# The manager's batch-capable MultiActionReActPlanner is constructed for us
# inside build_worker_manager; we never instantiate it here.
from fairlib import ReActPlanner, SimpleReActPlanner

# --- 1f. Memory ---
# Memory systems store conversation history and retrieved context.
# WorkingMemory is short-term, in-context memory (most common for demos).
# LongTermMemory is RAG-backed memory using a vector store + retriever.
from fairlib import WorkingMemory

# --- 1g. Model Abstraction Layer (MAL) ---
# The MAL lets you swap LLM providers without changing any agent code.
# Each adapter conforms to the same AbstractChatModel interface.
#   OpenAIAdapter      - GPT-4, GPT-3.5, etc.
#   AnthropicAdapter   - Claude 3, Claude 3.5, etc.
#   HuggingFaceAdapter - Local transformer models (v4 AND v5 compatible)
#   OllamaAdapter      - Local Ollama models
#   LoadBalancerAdapter - Distributes requests across multiple adapters
from fairlib import HuggingFaceAdapter

# --- 1h. Tool Components ---
# Tools give agents the ability to interact with the world.
# ToolRegistry holds a collection of tools.
# ToolExecutor runs tools by name and returns results.
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

# --- 1k. Observability ---
# Every step of the manager's loop emits typed events on a shared bus.
# Subscribing is how consumers observe the system; nothing is inferred by
# string-matching output.
from fairlib import AgentEventBus
from fairlib.core.events import (
    ToolBatchScheduledEvent,
    ToolCallPostEvent,
    ToolCallPreEvent,
)

# --- 1l. Tool Contract Types ---
# SideEffect is each tool's dispatch classification. For worker tools it is
# the author's assertion about the wrapped agent: READ_ONLY delegations may
# run concurrently within a turn; the conservative default EXTERNAL is a
# sequential barrier.
from fairlib.core.interfaces.tools import SideEffect
from fairlib.core.message import OBSERVATION_PREFIX


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

    Nothing here is manager-specific: this is the same construction a
    standalone agent uses. A worker only becomes delegable when it is
    wrapped in a WorkerAgentTool (Section 3, Step 3), and the manager
    learns what the worker is for from that wrapper's description, not
    from anything on the agent itself.

    Args:
        llm:                The language model adapter (any MAL adapter works)
        tools:              List of local tool instances
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
    return SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=memory,
        max_steps=max_steps,
        stateless=True,  # Workers clear memory between delegations
    )


# ==============================================================================
# SECTION 2b: EVENT-BUS INSTRUMENTATION
# ==============================================================================
# The manager and its executor share one AgentEventBus. Subscribing to the
# typed events is the supported way to watch delegation happen:
#   ToolBatchScheduledEvent - how the executor grouped a turn's delegations
#                             (READ_ONLY groups run in parallel; barriers
#                             run alone)
#   ToolCallPreEvent        - a delegation is starting
#   ToolCallPostEvent       - a delegation finished, with its observation
# Pre/Post events for the same call share (step, call_index): step is the
# manager's loop step and call_index is the call's position within the
# turn. That pair is the correlation key - two delegations in one turn can
# target the SAME worker tool, so keying by tool name would collide.

class DelegationReporter:
    """Bus subscriber narrating the manager's fan-out as it happens.

    A class because it carries state: per-call start times keyed by
    (step, call_index) - a turn may delegate to the same worker twice, so
    keying by tool name would collide - and the finished-delegation log
    the demo prints at the end.
    """

    def __init__(self) -> None:
        self.starts: Dict[Tuple[Optional[int], int], float] = {}
        self.log: List[str] = []

    def on_batch_scheduled(self, event: ToolBatchScheduledEvent) -> None:
        print(f"\n  [scheduler] step {event.step}: "
              f"{event.batch_size} delegation(s) this turn:")
        for i, group in enumerate(event.groups, start=1):
            how = "PARALLEL" if group.parallel else "sequential"
            print(f"    group {i}: {how:10} [{group.side_effect.value}] "
                  f"{', '.join(group.tool_names)}")

    def on_delegation_start(self, event: ToolCallPreEvent) -> None:
        self.starts[(event.step, event.call_index)] = time.perf_counter()
        print(f"  [delegate] step {event.step} call {event.call_index}: "
              f"-> {event.tool_name}")

    def on_delegation_done(self, event: ToolCallPostEvent) -> None:
        started = self.starts.pop((event.step, event.call_index), None)
        duration = time.perf_counter() - started if started is not None else 0.0
        snippet = event.observation[:90].replace("\n", " ")
        self.log.append(
            f"step {event.step} call {event.call_index} {event.tool_name}: "
            f"{duration:.1f}s ok={event.succeeded}"
        )
        print(f"  [result]   step {event.step} call {event.call_index}: "
              f"{event.tool_name} in {duration:.1f}s -> {snippet}")


# ==============================================================================
# SECTION 3: BUILDING THE RESEARCH TEAM
# ==============================================================================

async def build_research_team(llm):
    """
    Construct a 3-worker research team behind a fan-out manager.

    This function demonstrates:
    - Creating agents with different tool configurations
    - MCP integration for external web search
    - Graceful fallback when MCP is unavailable
    - WorkerAgentTool wrapping (descriptions + SideEffect declarations)
    - PromptBuilder role content for the manager
    - Event-bus instrumentation of delegation
    - build_worker_manager assembling the manager as a plain SimpleAgent
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
    # Step 2: Create the worker agents
    # ------------------------------------------------------------------
    # Each worker is an ordinary stateless SimpleAgent with its own planner
    # and tools - exactly the agent you would build standalone. Nothing
    # about a worker is team-specific yet.
    print_step(2, "Creating specialized worker agents")

    # RESEARCHER: Uses MCP web search OR local Google CSE
    researcher = create_worker_agent(
        llm,
        researcher_tools,
        mcp_registry=researcher_mcp,
        max_steps=3,
    )
    print(f"    Researcher agent ready [{search_source}]")

    # ANALYST: Uses SafeCalculatorTool for math
    analyst = create_worker_agent(
        llm,
        [SafeCalculatorTool()],
        max_steps=3,
    )
    print("    Analyst agent ready [SafeCalculatorTool]")

    # WRITER: No tools — relies on the LLM's own writing ability
    writer = create_worker_agent(
        llm,
        [],  # No tools!
    )
    print("    Writer agent ready [no tools, LLM only]")

    # ------------------------------------------------------------------
    # Step 3: Wrap each worker as a typed tool (WorkerAgentTool)
    # ------------------------------------------------------------------
    # This is the whole workers-as-tools trick. Each description below is
    # what the manager's model reads in its rendered tool catalog, so write
    # it the way you would brief a colleague on when to call this worker.
    #
    # SideEffect is the author's assertion about the wrapped agent:
    # - The Researcher only retrieves (web search mutates nothing and
    #   needs no ordering), the Analyst only computes, and the Writer only
    #   generates text. All three are therefore declared READ_ONLY, which
    #   is what allows independent delegations issued in one manager turn
    #   to run concurrently.
    # - A worker that mutated shared state (wrote files, posted to an API)
    #   would keep the conservative default, EXTERNAL: a sequential
    #   barrier, because overlapping mutations can interleave badly.
    print_step(3, "Wrapping workers as typed tools (WorkerAgentTool)")

    worker_tools = [
        WorkerAgentTool(
            researcher,
            name="researcher",
            description=(
                "Delegate a web research subtask, phrased as a complete "
                "question, to a research specialist that can search the web "
                "for current prices, statistics, news, and facts. It cannot "
                "do math and cannot write reports. It searches once and "
                "returns what it finds."
            ),
            side_effect=SideEffect.READ_ONLY,
        ),
        WorkerAgentTool(
            analyst,
            name="analyst",
            description=(
                "Delegate a math subtask, phrased as a complete request with "
                "the numbers included, to an analyst whose only tool is a "
                "safe calculator. It evaluates arithmetic expressions, "
                "percentages, and conversions. It cannot search the web and "
                "cannot write reports."
            ),
            side_effect=SideEffect.READ_ONLY,
        ),
        WorkerAgentTool(
            writer,
            name="writer",
            description=(
                "Delegate a writing subtask, including ALL the findings and "
                "numbers to use, to a writing specialist with no tools. It "
                "synthesizes the material you give it into a clear, organized "
                "summary or report. It cannot search and cannot calculate, so "
                "the subtask text must contain everything it needs."
            ),
            side_effect=SideEffect.READ_ONLY,
        ),
    ]
    for tool in worker_tools:
        print(f"    {tool.name}: side_effect={tool.side_effect.value}")

    # ------------------------------------------------------------------
    # Step 4: Application prompt content for the manager (PromptBuilder)
    # ------------------------------------------------------------------
    # fairlib ships no prompt content. The manager's planner auto-generates
    # the tool catalog from the worker tools' schemas and descriptions and
    # merges its own mandatory format instructions; the application only
    # supplies role content. Compare this with Step 3: the per-worker
    # guidance lives on the tools, not in a hand-written roster.
    print_step(4, "Building the manager's role prompt (PromptBuilder)")

    manager_builder = PromptBuilder()
    manager_builder.role_definition = RoleDefinition(
        "You are the manager of a research team. Break the user's request "
        "into subtasks and delegate each one to the right worker tool. You "
        "do NOT perform tasks yourself - you coordinate the team. Delegate "
        "research to the researcher, math to the analyst, and writing to the "
        "writer. When subtasks are independent of each other, delegate them "
        "together in the same turn; when one subtask needs another's result, "
        "wait for that result before delegating the next. When all subtasks "
        "are done, combine the results into your final answer."
    )
    print("    Manager role defined (tool catalog is auto-generated from")
    print("    the WorkerAgentTool schemas and descriptions).")

    # ------------------------------------------------------------------
    # Step 5: Shared event bus (AgentEventBus)
    # ------------------------------------------------------------------
    print_step(5, "Wiring delegation observability (AgentEventBus)")

    bus = AgentEventBus()
    reporter = DelegationReporter()
    bus.subscribe(ToolBatchScheduledEvent, reporter.on_batch_scheduled)
    bus.subscribe(ToolCallPreEvent, reporter.on_delegation_start)
    bus.subscribe(ToolCallPostEvent, reporter.on_delegation_done)
    print("    Subscribed to ToolBatchScheduledEvent, ToolCallPreEvent,")
    print("    and ToolCallPostEvent on the manager's bus.")

    # ------------------------------------------------------------------
    # Step 6: Assemble the manager (build_worker_manager)
    # ------------------------------------------------------------------
    # build_worker_manager is pure wiring: a batch-capable planner
    # (MultiActionReActPlanner) plus a ToolExecutor over the worker tools,
    # sharing one event bus, assembled into an ordinary SimpleAgent. There
    # is no orchestrator class: delegation IS the agent's normal tool
    # loop, so a turn that delegates to several READ_ONLY workers at once
    # is dispatched concurrently by the side-effect-aware executor.
    print_step(6, "Assembling the manager (build_worker_manager)")

    manager = build_worker_manager(
        llm,
        worker_tools,
        prompt_builder=manager_builder,
        events=bus,
        max_steps=15,
    )
    print("    Manager ready: a plain SimpleAgent whose tools are the workers.")

    return manager, brave_registry, reporter


# ==============================================================================
# SECTION 4: RUNNING THE DEMO
# ==============================================================================

async def run_preset_demo(manager, reporter):
    """Run a preset query to showcase the team in action."""
    print_section("RUNNING PRESET DEMO QUERY")

    query = (
        "I have a budget of $5,000. Find the current price of Bitcoin "
        "and calculate exactly how many Bitcoins I can afford. "
        "Then write a brief summary of the investment."
    )

    print(f"\n  Query: {query}\n")
    print("  Note: these subtasks depend on each other (the math needs the")
    print("  price; the summary needs both), so expect the manager to chain")
    print("  them across turns. A request with independent parts can be")
    print("  delegated in one turn and run concurrently.")
    print("-" * 70)

    result = await manager.arun(query)

    print("\n" + "=" * 70)
    print("  FINAL RESEARCH REPORT")
    print("=" * 70)
    print(result)

    print("\n  Delegations this run (from the event bus):")
    for line in reporter.log:
        print(f"    {line}")

    print("\n  Manager memory (one observation per delegation, in call order):")
    for message in manager.memory.get_history():
        if message.content.startswith(OBSERVATION_PREFIX):
            print(f"    {message.content[:140]}")
    return result


async def run_interactive(manager):
    """Run in interactive mode, accepting queries from the user."""
    print_section("INTERACTIVE MODE")
    print("\n  The research team is ready for your queries!")
    print("  The team consists of:")
    print("    - Manager:    A plain SimpleAgent that delegates via worker tools")
    print("    - Researcher: Searches the web for information")
    print("    - Analyst:    Performs mathematical calculations")
    print("    - Writer:     Synthesizes findings into reports")
    print("\n  Example queries:")
    print('    - "Find the price of Ethereum and calculate how many I can buy with $2,000"')
    print('    - "Research the latest AI trends and write a brief summary"')
    print('    - "What is 15% of 8,500?"')
    print('    - "Separately: find the price of Bitcoin, and compute 5000 / 3.14"')
    print("      (independent subtasks like these can fan out in one turn)")
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
            result = await manager.arun(user_input)

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
  This demo walks you through the major classes in the FAIR-LLM framework
  while building a functional 3-worker research team behind a fan-out
  manager.

  Classes demonstrated:
    Core:      Message, Thought, Action, FinalAnswer, Document, settings
    Prompts:   PromptBuilder, RoleDefinition (the worker tool catalog is
               auto-generated from the WorkerAgentTool schemas)
    Agents:    SimpleAgent, WorkerAgentTool, build_worker_manager
    Planners:  ReActPlanner, SimpleReActPlanner (workers);
               MultiActionReActPlanner (wired inside build_worker_manager)
    Memory:    WorkingMemory
    Events:    AgentEventBus, ToolBatchScheduledEvent, ToolCallPreEvent,
               ToolCallPostEvent
    MAL:       HuggingFaceAdapter (supports transformers v4 AND v5)
    Tools:     ToolRegistry, ToolExecutor, SafeCalculatorTool,
               WebSearcherTool, SideEffect
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
    # which is critical for the manager's structured multi-action format.
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
    manager, brave_registry, reporter = await build_research_team(llm)

    # Choose mode based on command-line args
    if "--preset" in sys.argv:
        await run_preset_demo(manager, reporter)
    else:
        await run_interactive(manager)

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
  3. PROMPTBUILDER:   Compose prompts from structured, reusable pieces;
                      tool catalogs are generated, never hand-written
  4. AGENTS:          SimpleAgent is the workhorse; workers are stateless
  5. MULTI-AGENT:     WorkerAgentTool + build_worker_manager = team;
                      the manager is itself just a SimpleAgent, and
                      READ_ONLY delegations can fan out concurrently
  6. OBSERVABILITY:   Subscribe to typed events on the shared bus;
                      correlate Pre/Post by (step, call_index)
  7. MCP:             Use anyone's tools via MCPServerConfig + SSE/stdio
  8. GRACEFUL:        Always fall back when external services are unavailable
    """)


if __name__ == "__main__":
    asyncio.run(main())
