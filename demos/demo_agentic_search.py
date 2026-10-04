# demo_agentic_search.py

"""
Agentic search over a file tree with the read-only standard tools.

This demo gives a single agent the four standard file tools - list_dir, glob,
grep, and read_file - all confined to one directory, and asks it a series of
questions - one per file in the tree - each answerable only by navigating: find
where something lives, open it, and report what it found. The agent is not told
where any answer lives; it has to scan, narrow, and read, which is exactly the
agentic-search loop these tools exist to support (Anthropic's recommended
default before reaching for embeddings). Each question starts from a fresh
working memory, so the agent navigates from nothing every time instead of
answering from what an earlier question turned up.

A named format instruction, survey_first, carries the one search-strategy
rule the loop needs beyond the tools themselves: survey every path in the
tree before narrowing, because a guessed name that matches nothing is not
proof that the thing is absent. The model decodes greedily, so a run is
repeatable.

The demo subscribes to the agent's event bus and prints every tool call as it
happens: the tool, the input it received, and the first line of what it
returned.
"""

import asyncio
import tempfile
from pathlib import Path

from fairlib import (
    AgentEventBus,
    GlobTool,
    GrepTool,
    HuggingFaceAdapter,
    ListDirTool,
    MaxStepsExceeded,
    ReActPlanner,
    ReadFileTool,
    RoleDefinition,
    SimpleAgent,
    ToolCallPostEvent,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
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
        "def apply_coupon(total, percent):\n    return total * (1 - percent / 100)\n"
    ),
    "catalog/products.py": (
        'PRODUCTS = {\n    "widget": 9.99,\n    "gadget": 14.50,\n}\n'
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


def on_tool_call(event: ToolCallPostEvent) -> None:
    """Print one tool call: the tool, its input, and what it returned, whole."""
    status = "ok" if event.succeeded else "failed"
    print(
        f"  [{event.tool_name}] {event.tool_input!r} -> {status}:\n{event.observation}"
    )


def build_fixture(root: Path) -> None:
    """Write the sample project tree under root."""
    for relative, content in FIXTURE_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


async def main():
    # The local model. max_new_tokens gives each step room for a thought and a
    # single action without inviting the model to spill the whole loop at once.
    print(
        f"Loading {MODEL_NAME} via the HuggingFaceAdapter (first run downloads weights)..."
    )
    # Greedy decoding (do_sample=False): a navigation loop needs the model's
    # most likely next step every time, not a sampled one; sampling lets a
    # run skip the survey and conclude early on some runs and not others.
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=512, do_sample=False)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_fixture(root)

        # Every file tool is confined to the fixture root granted here. The
        # agent can navigate freely inside it and cannot reach anything above it.
        registry = ToolRegistry()
        for tool in (
            ListDirTool(root),
            GlobTool(root),
            GrepTool(root),
            ReadFileTool(root),
        ):
            registry.register_tool(tool)
        print(f"Tools: {[name for name in registry.get_all_tools()]}")
        print(f"Search root: {root}\n")

        executor = ToolExecutor(registry)

        # The JSON ReActPlanner: the 14B follows the survey rule and reads its
        # observations more reliably in JSON turns than in key-value ones.
        planner = ReActPlanner(llm, registry)
        planner.prompt_builder.role_definition = RoleDefinition(
            "You are a codebase navigator. You answer questions about a project "
            "by searching its files. Work one step at a time: list or glob to "
            "find candidates, grep to locate a symbol, and read the few files "
            "that matter before answering. Do not guess paths you have not seen."
        )
        # The search strategy, named so it can be read or overridden on its
        # own: survey the tree before narrowing. A glob or grep that guesses a
        # name and finds nothing proves only that the guess was wrong, and a
        # small model reads that as proof the thing is absent.
        planner.prompt_builder.set_format_instruction(
            "survey_first",
            "Begin every question by calling glob with the pattern **/* to see "
            "every file path in the project. Then choose every file whose path "
            "or name could fit the question, open each with read_file, and "
            "answer only from what those files say. Never conclude that "
            "something is absent until you have read every file whose path or "
            "name could hold it.",
        )
        # What an answer carries: the path of every file it draws on, so a
        # reader can check it, even when the question names the file.
        planner.prompt_builder.set_format_instruction(
            "answer_names_paths",
            "Your final answer names the path of every file it draws on, even "
            "when the question already names that file.",
        )

        # One bus for every question's agent: the executor is shared, and an
        # executor serves one bus, so the ticker subscribes once.
        bus = AgentEventBus()
        bus.subscribe(ToolCallPostEvent, on_tool_call)

        for index, question in enumerate(QUESTIONS, start=1):
            print("=" * 60)
            print(f"Question {index}/{len(QUESTIONS)}:\n  {question}\n")
            print("Running agentic search...\n")
            # A fresh agent and working memory per question: nothing found for
            # an earlier question is in context, so each answer is navigated.
            agent = SimpleAgent(
                llm=llm,
                planner=planner,
                tool_executor=executor,
                memory=WorkingMemory(),
                max_steps=12,
                events=bus,
            )
            try:
                answer = await agent.arun(question)
            except MaxStepsExceeded as exc:
                print(f"Agent could not finish: {type(exc).__name__}: {exc}")
                continue
            print("Agent answer:")
            print(answer)
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
