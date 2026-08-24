"""
Demo: smarter SummarizingMemory compaction.

Part 1 exercises scripted compaction (no GPU):

  (a) Cheap deterministic snip of a huge tool observation under a token budget
  (b) Token-budget trigger with overhead_token_count for system/tools
  (c) User/assistant summary reinjection after an LLM pass
  (d) PathArtifactReGrounder refreshing a ground-truth file after compaction

Part 2 loads a live HuggingFaceAdapter (default). Set FAIR_LLM_DEMO_MODEL to
override the default model name.
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

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen25-7b")


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
        assert llm.calls == 0, "cheap pass should skip the LLM"
        assert any("[snipped" in m.content for m in hist)

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
        assert hist2[1].role == "user" and hist2[2].role == "assistant"
        assert any(
            m.importance == "pinned" and m.role == "user" and "return 42" in m.content
            for m in hist2
        ), "artifact re-ground must inject pinned user-role ground truth"
        print("Part 1 OK.\n")


async def part2_live() -> None:
    from fairlib import HuggingFaceAdapter

    print("=== Part 2: live model compaction ===\n")
    llm = HuggingFaceAdapter(MODEL_NAME)
    bus = AgentEventBus()
    bus.subscribe(MemorySummarizedEvent, _announce)
    memory = SummarizingMemory(
        llm,
        max_history_length=8,
        messages_to_keep_at_end=3,
        max_context_tokens=2048,
        overhead_token_count=256,
        events=bus,
    )
    for i in range(12):
        memory.add_message(Message(role="user", content=f"User note {i}"))
        memory.add_message(Message(role="assistant", content=f"Ack {i}"))
    hist = await memory.aget_history()
    print(f"Live compaction left {len(hist)} messages; reason path exercised.")
    print("Part 2 OK.\n")


async def main() -> None:
    await part1_scripted()
    if os.environ.get("FAIR_DEMO_SCRIPTED_ONLY") == "1":
        print("Skipping Part 2 (FAIR_DEMO_SCRIPTED_ONLY=1).")
        return
    await part2_live()


if __name__ == "__main__":
    asyncio.run(main())
