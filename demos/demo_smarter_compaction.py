"""
Demo: smarter SummarizingMemory compaction.

Part 1 exercises scripted compaction (no GPU):

  (a) Cheap deterministic snip of a huge tool observation under a token budget
  (b) Token-budget trigger with overhead_token_count for system/tools
  (c) User/assistant summary reinjection after an LLM pass
  (d) PathArtifactReGrounder refreshing a ground-truth file after compaction

Part 2 drives a SimpleAgent whose memory is a SummarizingMemory over a live
HuggingFaceAdapter. Set FAIR_LLM_DEMO_MODEL to override the default model.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from fairlib import (
    AgentEventBus,
    MemorySummarizedEvent,
    Message,
    PathArtifactReGrounder,
    SummarizingMemory,
)
from fairlib.core.message import OBSERVATION_MARKER_KEY

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "dolphin3-qwen25-3b")


class _ScriptedLLM:
    """Deterministic stand-in used by Part 1 (no model weights)."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        return Message(
            role="assistant", content="Scripted compact summary of earlier turns."
        )

    def estimate_token_count(self, text: str) -> int:
        return max(1, len(text) // 4)


def _announce(event: MemorySummarizedEvent) -> None:
    preview = event.summary.content[:80].replace("\n", " ")
    print(
        f"  [memory] compacted reason={event.reason.value} "
        f"dropped={len(event.dropped)} kept={len(event.kept)} "
        f"summary={preview!r}"
    )


def _obs(body: str) -> Message:
    return Message(
        role="system",
        content=f"Observation: {body}",
        metadata={OBSERVATION_MARKER_KEY: True},
    )


async def part1_scripted() -> None:
    print("=== Part 1: scripted smarter compaction (no GPU) ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "source.py"
        artifact.write_text("def answer():\n    return 42\n", encoding="utf-8")

        # --- (a)+(b) cheap pass under a tight token budget -----------------
        bus = AgentEventBus()
        bus.subscribe(MemorySummarizedEvent, _announce)
        llm = _ScriptedLLM()
        mem = SummarizingMemory(
            llm,
            max_history_length=20,
            messages_to_keep_at_end=2,
            max_context_tokens=100,
            overhead_token_count=10,
            observation_snip_chars=48,
            events=bus,
            artifact_re_grounder=PathArtifactReGrounder([artifact]),
        )
        mem.add_message(Message(role="system", content="You are a coding assistant."))
        mem.add_message(Message(role="user", content="read the file"))
        mem.add_message(Message(role="assistant", content="Action: read_file"))
        mem.add_message(_obs("PAYLOAD_" + ("Z" * 5000)))
        mem.add_message(Message(role="user", content="ok"))
        mem.add_message(Message(role="assistant", content="done"))

        print("Before cheap pass: over_token_budget=", mem._over_token_budget())
        hist = await mem.aget_history()
        print(f"After cheap pass: llm_calls={llm.calls} messages={len(hist)}")
        snipped = [m.content[:60] for m in hist if "[snipped" in m.content]
        print(f"Snipped observations: {snipped}")

        # --- (c)+(d) LLM summarize with role-faithful reinjection ----------
        bus2 = AgentEventBus()
        bus2.subscribe(MemorySummarizedEvent, _announce)
        llm2 = _ScriptedLLM()
        mem2 = SummarizingMemory(
            llm2,
            max_history_length=6,
            messages_to_keep_at_end=2,
            events=bus2,
            artifact_re_grounder=PathArtifactReGrounder([artifact]),
        )
        mem2.add_message(Message(role="system", content="first"))
        for i in range(8):
            role = "user" if i % 2 == 0 else "assistant"
            mem2.add_message(Message(role=role, content=f"turn-{i}"))

        hist2 = await mem2.aget_history()
        print(
            f"After LLM compaction: llm_calls={llm2.calls} "
            f"roles={[m.role for m in hist2]}"
        )
        reground = [
            m.content[:60]
            for m in hist2
            if m.importance == "pinned"
            and m.role == "user"
            and "return 42" in m.content
        ]
        print(
            f"Pinned user-role ground truth re-injected from the artifact: {reground}"
        )
        print("Part 1 done.\n")


async def part2_live() -> None:
    from fairlib import (
        HuggingFaceAdapter,
        MaxStepsExceeded,
        PlannerParseError,
        RoleDefinition,
        SimpleAgent,
        SimpleReActPlanner,
        ToolExecutor,
        ToolRegistry,
    )

    print("=== Part 2: live agent over SummarizingMemory ===\n")
    llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=200)
    bus = AgentEventBus()
    bus.subscribe(MemorySummarizedEvent, _announce)
    # The memory and the agent share one event bus, so the compaction event
    # is observable from the same place as the agent's own events.
    memory = SummarizingMemory(
        llm,
        max_history_length=6,
        messages_to_keep_at_end=2,
        events=bus,
    )
    tool_registry = ToolRegistry()
    planner = SimpleReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a note-taking assistant. Acknowledge each note in one short sentence."
    )
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=memory,
        max_steps=4,
        events=bus,
    )
    for i in range(5):
        note = f"Note {i}: the meeting is on day {i + 1}."
        print(f"You: {note}")
        try:
            print(f"Agent: {await agent.arun(note)}")
        except (PlannerParseError, MaxStepsExceeded) as exc:
            # A small model sometimes breaks the planner format or never
            # reaches a final answer; the typed error is the framework's
            # signal, and the note is still in memory either way.
            print(f"Agent: (typed signal from the loop: {exc.__class__.__name__})")
    hist = await memory.aget_history()
    print(
        f"\nAfter 5 turns the history holds {len(hist)} messages; "
        "the compaction event above fired from the agent's bus."
    )
    print("Part 2 OK.\n")


async def main() -> None:
    await part1_scripted()
    await part2_live()


if __name__ == "__main__":
    asyncio.run(main())
