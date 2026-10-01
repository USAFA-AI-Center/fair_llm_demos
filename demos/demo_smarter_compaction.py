"""
Demo: smarter SummarizingMemory compaction, live.

A real model drives a SimpleAgent that reads files, and its memory is a
SummarizingMemory configured with every smarter-compaction feature:

  (a) Token-budget trigger: max_context_tokens bounds the history, and
      overhead_token_count adds the system prompt and tool catalog that
      sit outside memory (measured from the planner's own prompt below).
  (b) Cheap deterministic snip: when one huge tool observation blows the
      budget, the memory first cuts the middle out of long observations
      (head and tail kept) and skips the LLM when that alone is enough.
      The event reason is cheap_pass_sufficient.
  (c) LLM summarization with role-faithful reinjection: when the message
      limit is exceeded, the real model summarizes the middle of the
      conversation and the summary comes back as a user/assistant
      exchange, so role alternation holds.
  (d) Re-grounding: after every compaction a PathArtifactReGrounder
      re-reads a ground-truth file (config.py here) and pins its fresh
      text into history. The agent never reads config.py itself, yet the
      last question about it is answerable, because the re-grounded copy
      is in context.

Every MemorySummarizedEvent is printed as it arrives on the agent's bus,
with the history layout after each turn, so each feature is visible as the
live agent trips it. Set FAIR_LLM_DEMO_MODEL to override the default model.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from fairlib import (
    AgentEventBus,
    HuggingFaceAdapter,
    MaxStepsExceeded,
    MemorySummarizedEvent,
    PathArtifactReGrounder,
    PlannerParseError,
    ReadFileTool,
    RoleDefinition,
    SimpleAgent,
    SimpleReActPlanner,
    SummarizingMemory,
    ToolCallPostEvent,
    ToolExecutor,
    ToolRegistry,
)
from fairlib.core.message import RE_GROUND_MARKER_KEY, SUMMARY_MARKER_KEY

# qwen25-14b by default: after compaction the 7B model sometimes answers the
# summarized earlier question instead of the one just asked.
MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen25-14b")

# The build log is long on purpose: one read of it is far over the token
# budget. The line the user asks about is the last one, so the snip, which
# keeps the head and the tail of an observation, keeps it.
LOG_LINES = (
    ["BUILD 2026-09-28 started on runner ci-07"]
    + [f"INFO step {i:03d}: compiled module_{i:03d}.py ok" for i in range(1, 301)]
    + ["FINAL STATUS: FAILED - 3 tests failed in test_checkout.py"]
)

FIXTURE_FILES = {
    "build.log": "\n".join(LOG_LINES) + "\n",
    "config.py": "MAX_RETRIES = 5\nTIMEOUT_SECONDS = 30\n",
    "notes.txt": "Release owner: Dana Reyes. Freeze starts Friday.\n",
    "team.txt": "On call this week: Priya Shah.\n",
}

TURNS = (
    "Read build.log and tell me whether the build passed or failed, and why.",
    "Read notes.txt and tell me who owns the release.",
    "Read team.txt and tell me who is on call this week.",
    "Without reading any file, answer from what is already in this "
    "conversation: what is MAX_RETRIES in config.py, and did the build pass?",
)


def on_compaction(event: MemorySummarizedEvent) -> None:
    """Print one compaction: why it fired, what it did, and the summary."""
    print(
        f"  [memory] compacted reason={event.reason.value} "
        f"dropped={len(event.dropped)} kept={len(event.kept)}"
    )
    for message in event.dropped[:1]:
        print(
            f"           dropped/snipped message was {len(message.content)} chars "
            f"({message.role})"
        )
    for message in event.kept:
        if "[snipped" in message.content:
            print(f"           snipped observation now:\n{message.content}")
    print(f"           summary:\n{event.summary.content}")
    for message in event.kept:
        if message.metadata.get(RE_GROUND_MARKER_KEY):
            print(
                f"           re-grounded ({message.role}, pinned):\n{message.content}"
            )


def on_tool_call(event: ToolCallPostEvent) -> None:
    """Print one tool call and how long its observation is."""
    status = "ok" if event.succeeded else "failed"
    print(
        f"  [{event.tool_name}] {event.tool_input!r} -> {status}, "
        f"{len(event.observation)} chars"
    )


async def show_layout(memory: SummarizingMemory, overhead: int) -> None:
    """Print the history the next planner step would see, one line per message."""
    history = await memory.aget_history()
    tokens = overhead + sum(
        memory.llm.estimate_token_count(message.content) for message in history
    )
    print(
        f"  history: {len(history)} messages, about {tokens} tokens with overhead "
        f"(budget {memory.max_context_tokens})"
    )
    for message in history:
        tags = []
        if message.importance == "pinned":
            tags.append("pinned")
        if message.metadata.get(RE_GROUND_MARKER_KEY):
            tags.append("re-ground")
        if message.metadata.get(SUMMARY_MARKER_KEY):
            tags.append("summary")
        label = f" [{', '.join(tags)}]" if tags else ""
        print(f"    {message.role:9s}{label}:\n{message.content}")


async def main() -> None:
    print("=== Smarter compaction on a live agent ===\n")
    print(f"Loading {MODEL_NAME} via HuggingFaceAdapter...")
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=256)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, content in FIXTURE_FILES.items():
            (root / name).write_text(content, encoding="utf-8")

        registry = ToolRegistry()
        registry.register_tool(ReadFileTool(root))
        planner = SimpleReActPlanner(llm, registry)
        planner.prompt_builder.role_definition = RoleDefinition(
            "You are a build assistant. Use read_file to read the files the user "
            "names. Then answer in one short plain sentence. "
            "When the user asks you to answer from memory, do not call any tool."
        )

        # The system prompt and the tool catalog are sent on every step but
        # live outside memory, so the budget counts them as overhead.
        overhead = llm.estimate_token_count(planner.render_system_prompt())
        bus = AgentEventBus()
        memory = SummarizingMemory(
            llm,
            max_history_length=10,
            messages_to_keep_at_end=3,
            max_context_tokens=overhead + 1200,
            overhead_token_count=overhead,
            observation_snip_chars=600,
            events=bus,
            artifact_re_grounder=PathArtifactReGrounder(
                [root / "config.py"], root=root
            ),
        )
        print(
            f"Memory: max_history_length=10, max_context_tokens={overhead + 1200} "
            f"(overhead {overhead} measured from the system prompt), "
            "observation_snip_chars=600, re-grounding config.py\n"
        )

        # The memory and the agent share one bus, so compaction events arrive
        # on the same subscription surface as the tool calls.
        agent = SimpleAgent(
            llm=llm,
            planner=planner,
            tool_executor=ToolExecutor(registry),
            memory=memory,
            max_steps=6,
            events=bus,
        )
        bus.subscribe(MemorySummarizedEvent, on_compaction)
        bus.subscribe(ToolCallPostEvent, on_tool_call)

        for turn in TURNS:
            print(f"You: {turn}")
            try:
                print(f"Agent: {await agent.arun(turn)}")
            except (PlannerParseError, MaxStepsExceeded) as exc:
                print(f"Agent: (typed signal from the loop: {type(exc).__name__})")
            await show_layout(memory, overhead)
            print()


if __name__ == "__main__":
    asyncio.run(main())
