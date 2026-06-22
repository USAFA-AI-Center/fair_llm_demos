# demo_agentic_search.py

"""
Agentic search over a file tree with the read-only standard tools.

This demo gives a single agent the four standard file tools - list_dir, glob,
grep, and read_file - all confined to one directory, and asks it a series of
questions - one per file in the tree - each answerable only by navigating: find
where something lives, open it, and report what it found. The agent is not told
where any answer lives; it has to scan, narrow, and read, which is exactly the
agentic-search loop these tools exist to support (Anthropic's recommended
default before reaching for embeddings). The questions form one contiguous
conversation on a single agent, so it can build on what it already found for
earlier questions rather than rediscovering the tree every time.
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
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
)

# The local model that drives the search
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

# A small self-contained project the agent will navigate. Every QUESTION below
# is answerable from exactly one of these files
FIXTURE_FILES = {
    "README.md": (
        "# Storefront\n\n"
        "A toy order-processing package. Pricing lives under billing/.\n"
    ),
    "billing/__init__.py": "from .invoice import compute_total\n",
    "billing/invoice.py": (
        "TAX_RATE = 0.08\n\n"
        "def compute_total(subtotal, shipping):\n"
        '    """Return the grand total: subtotal plus tax plus shipping."""\n'
        "    tax = subtotal * TAX_RATE\n"
        "    return subtotal + tax + shipping\n"
    ),
    "billing/discounts.py": (
        "def apply_coupon(total, percent):\n"
        "    return total * (1 - percent / 100)\n"
    ),
    "catalog/products.py": (
        "PRODUCTS = {\n"
        '    "widget": 9.99,\n'
        '    "gadget": 14.50,\n'
        "}\n"
    ),
    "notes.txt": "Remember to revisit the tax rate before launch.\n",
}

# One question per file in FIXTURE_FILES, each phrased by content rather than by
# path so the agent has to search for the file that answers it. Order roughly
# follows the tree, but the agent receives only the question text.
QUESTIONS = [
    "What is this project, according to its README, and where does the README "
    "say the pricing code lives? Give the file path you found this in.",
    "What single name does the billing package make importable directly from "
    "the package itself? Name the file that decides this.",
    "Which file defines the function compute_total, and what does that function "
    "return? Give the file path and a one-sentence description of the return value.",
    "Does this project have any coupon or discount logic? Name the function, the "
    "file it lives in, and explain what it computes.",
    "What products does this project sell, and at what price each? Give the file "
    "that lists them.",
    "Is there any outstanding note or to-do recorded in the project before "
    "launch? Quote it and give the file it is in.",
]


def build_fixture(root: Path) -> None:
    """Write the sample project tree under root."""
    for relative, content in FIXTURE_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


async def main():
    # The local model. max_new_tokens gives each step room for a thought and a
    # single action without inviting the model to spill the whole loop at once.
    print(f"Loading {MODEL_NAME} via the HuggingFaceAdapter (first run downloads weights)...")
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=512)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_fixture(root)

        # Every file tool is confined to the fixture root granted here. The
        # agent can navigate freely inside it and cannot reach anything above it.
        registry = ToolRegistry()
        for tool in (ListDirTool(root), GlobTool(root), GrepTool(root), ReadFileTool(root)):
            registry.register_tool(tool)
        print(f"Tools: {[name for name in registry.get_all_tools()]}")
        print(f"Search root: {root}\n")

        executor = ToolExecutor(registry)
        
        planner = SimpleReActPlanner(llm, registry)
        planner.prompt_builder.role_definition = RoleDefinition(
            "You are a codebase navigator. You answer questions about a project "
            "by searching its files. Work one step at a time: list or glob to "
            "find candidates, grep to locate a symbol, and read the few files "
            "that matter before answering. Do not guess paths you have not seen. "
            "Pass each tool a single bare value as tool_input, never a 'field: "
            "value' pair: to grep for compute_total, write 'tool_input: "
            "compute_total', not 'tool_input: pattern: compute_total'."
        )

        agent = SimpleAgent(
            llm=llm,
            planner=planner,
            tool_executor=executor,
            memory=WorkingMemory(),
            max_steps=12,
        )

        for index, question in enumerate(QUESTIONS, start=1):
            print("=" * 60)
            print(f"Question {index}/{len(QUESTIONS)}:\n  {question}\n")
            print("Running agentic search...\n")
            answer = await agent.arun(question)
            print("Agent answer:")
            print(answer)
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
