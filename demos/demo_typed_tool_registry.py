# demo_typed_tool_registry.py

"""
Scoping an agent's toolset with a typed registry and tool groups.

A registry can hold every tool an application owns, but a single agent should
only see the tools its job needs - least privilege, the same principle that
keeps a tool that touches the filesystem on a short leash. This demo builds one
broad registry (a calculator plus the four file tools), bundles just the
read-only search tools into a "file_search" group, and hands that group - and
nothing else - to an agent as its complete toolset. The agent then answers a
question about a small project tree it has to navigate, proving the scoped
toolset is enough to do the job and that the agent never had reach to anything
outside the group.

Two parts of the registry contract carry this:
- get(ToolType) assembles the group with type-safe lookups, so the wiring is
  checked at the call site rather than keyed on fragile strings.
- get_group(name) returns a registry view holding just the group's tools, which
  drops straight into a ToolExecutor and a planner exactly like any registry.

This is an end-to-end agent run on a local model. Run:
    python demos/demo_typed_tool_registry.py
"""

import asyncio
import tempfile
from pathlib import Path

from fairlib import (
    HuggingFaceAdapter,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    SimpleAgent,
    SimpleReActPlanner,
    RoleDefinition,
    SafeCalculatorTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
)

# The local model that drives the agent.
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

# A small self-contained project the agent will navigate to answer the question.
FIXTURE_FILES = {
    "README.md": (
        "# Storefront\n\n"
        "A toy order-processing package. Pricing lives under billing/.\n"
    ),
    "billing/invoice.py": (
        "TAX_RATE = 0.08\n\n"
        "def compute_total(subtotal, shipping):\n"
        '    """Return the grand total: subtotal plus tax plus shipping."""\n'
        "    tax = subtotal * TAX_RATE\n"
        "    return subtotal + tax + shipping\n"
    ),
    "catalog/products.py": (
        "PRODUCTS = {\n"
        '    "widget": 9.99,\n'
        '    "gadget": 14.50,\n'
        "}\n"
    ),
}

QUESTION = (
    "Which file defines the function compute_total, and what does that function "
    "return? Give the file path and a one-sentence description of the return value."
)


def build_fixture(root: Path) -> None:
    """Write the sample project tree under root."""
    for relative, content in FIXTURE_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


async def main() -> None:
    print(f"Loading {MODEL_NAME} via the HuggingFaceAdapter (first run downloads weights)...")
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=512)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_fixture(root)

        # The full registry an application might own: a calculator and the four
        # file tools, every file tool confined to the fixture root.
        registry = ToolRegistry()
        registry.register_tool(SafeCalculatorTool())
        for tool in (ListDirTool(root), GlobTool(root), GrepTool(root), ReadFileTool(root)):
            registry.register_tool(tool)
        print(f"Full registry: {list(registry.get_all_tools())}")

        # Bundle only the read-only search tools into a group. get(ToolType) does
        # the lookup type-safely - the result is the concrete tool class, so a
        # rename is caught here rather than as a runtime miss on a string key.
        registry.register_group("file_search", [
            registry.get(GlobTool),
            registry.get(GrepTool),
            registry.get(ReadFileTool),
        ])

        # The agent's toolset is the group view and nothing else. It is a
        # ToolRegistry, so it powers the executor and planner like any registry,
        # but it cannot reach the calculator or list_dir outside the group.
        toolset = registry.get_group("file_search")
        print(f"Agent toolset (file_search group): {list(toolset.get_all_tools())}")
        print(f"Search root: {root}\n")

        executor = ToolExecutor(toolset)
        planner = SimpleReActPlanner(llm, toolset)
        planner.prompt_builder.role_definition = RoleDefinition(
            "You are a codebase navigator. You answer questions about a project "
            "by searching its files. Work one step at a time: glob to find "
            "candidates, grep to locate a symbol, and read the few files that "
            "matter before answering. Do not guess paths you have not seen. Pass "
            "each tool a single bare value as tool_input, never a 'field: value' "
            "pair and never wrapped in quotes: to grep for compute_total, write "
            "tool_input: compute_total (not tool_input: pattern: compute_total, and "
            "not tool_input with the value in quotes)."
        )

        agent = SimpleAgent(
            llm=llm,
            planner=planner,
            tool_executor=executor,
            memory=WorkingMemory(),
            max_steps=12,
        )

        print("=" * 60)
        print(f"Question:\n  {QUESTION}\n")
        print("Running the scoped agent...\n")
        answer = await agent.arun(QUESTION)
        print("Agent answer:")
        print(answer)
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
