# demos/demo_streaming_agent.py
"""
Stream an agent's model calls end-to-end as typed events.

A real local model answers one question with streaming enabled, twice:
once on the constrained path (the model declares response_schema, so the
planner sends its action schema and the reply is one JSON action object)
and once with constrained_decoding=False (SimpleReActPlanner's key-value
text path). Two consumers subscribe to each run:

  1. A raw-feed consumer prints every ModelStreamChunkEvent delta as it
     arrives - the whole completion, thought and action included, the
     way a trace or debugging surface would watch it.
  2. A chat-surface consumer builds one filter per planner stream from
     the reply_format the planner states on ModelStreamStartEvent
     (final_answer_stream_filter) and prints only the final-answer text -
     the way a chat UI or an avatar app would speak it. It never asks
     which planner or which path produced the stream.

The run's return value is identical to a non-streaming run; streaming
changes how the model call is consumed and observed, not the result.

Requirements: a local HuggingFace model (transformers; a GPU is
recommended) and the grammar extra (pip install "fair-llm[grammar]"), so
the adapter declares response_schema. The first run downloads the
weights. Set FAIR_LLM_DEMO_MODEL to override the default.

Run:
    python demos/demo_streaming_agent.py
"""

import asyncio
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fairlib import (
    AbstractStreamFilter,
    AgentEventBus,
    HuggingFaceAdapter,
    ModelStreamChunkEvent,
    ModelStreamEndEvent,
    ModelStreamStartEvent,
    RoleDefinition,
    SafeCalculatorTool,
    SimpleAgent,
    SimpleReActPlanner,
    StreamSource,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
    final_answer_stream_filter,
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

QUESTION = "What is 6 times 7? Use the calculator tool, then answer in one sentence."


def build_agent(llm, bus: AgentEventBus, constrained: bool) -> SimpleAgent:
    tools = ToolRegistry()
    tools.register_tool(SafeCalculatorTool())
    planner = SimpleReActPlanner(
        llm, tools, stream=True, constrained_decoding=constrained
    )
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a concise assistant. Use your tools when a request needs "
        "them and follow the strict formatting rules that follow."
    )
    return SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tools),
        memory=WorkingMemory(),
        events=bus,
        stream=True,
    )


async def run_once(llm, label: str, constrained: bool) -> None:
    print(f"\n=== Run: {label} (constrained_decoding={constrained}) ===")
    bus = AgentEventBus()
    agent = build_agent(llm, bus, constrained)

    # Consumer 1: the raw feed. Every delta of every stream, tagged by
    # stream so overlapping streams (a retry, the validator rewrite)
    # stay distinguishable.
    def on_start(event: ModelStreamStartEvent) -> None:
        mode = "simulated" if event.simulated else "live"
        stated = event.reply_format.value if event.reply_format else "none"
        print(
            f"\n--- stream {event.stream_id} started "
            f"({event.source.value}, step={event.step}, {mode}) ---"
        )
        print(f"reply_format: {stated}")
        print("raw feed: ", end="", flush=True)

    def on_chunk(event: ModelStreamChunkEvent) -> None:
        print(event.text, end="", flush=True)

    def on_end(event: ModelStreamEndEvent) -> None:
        print(
            f"\n--- stream {event.stream_id} ended: {event.finish_reason.value}, "
            f"{event.chunk_count} chunks, {event.total_chars} chars ---"
        )

    bus.subscribe(ModelStreamStartEvent, on_start)
    bus.subscribe(ModelStreamChunkEvent, on_chunk)
    bus.subscribe(ModelStreamEndEvent, on_end)

    # Consumer 2: the chat surface. One filter per PLANNER stream, chosen
    # from the format the planner states for that stream. A stream that
    # states no format (a VALIDATOR_REWRITE stream is plain answer text)
    # gets no filter; a surface prints the returned answer for it.
    filters: Dict[int, AbstractStreamFilter] = {}
    segments: Dict[int, List[str]] = {}

    def chat_on_start(event: ModelStreamStartEvent) -> None:
        if event.source is not StreamSource.PLANNER:
            return
        stream_filter = final_answer_stream_filter(event.reply_format)
        if stream_filter is not None:
            filters[event.stream_id] = stream_filter
            segments[event.stream_id] = []

    def chat_on_chunk(event: ModelStreamChunkEvent) -> None:
        if event.stream_id in filters:
            text = filters[event.stream_id].feed(event.text)
            if text:
                segments[event.stream_id].append(text)

    def chat_on_end(event: ModelStreamEndEvent) -> None:
        if event.stream_id in filters:
            tail = filters.pop(event.stream_id).finish()
            if tail:
                segments[event.stream_id].append(tail)

    bus.subscribe(ModelStreamStartEvent, chat_on_start)
    bus.subscribe(ModelStreamChunkEvent, chat_on_chunk)
    bus.subscribe(ModelStreamEndEvent, chat_on_end)

    print(f"Question: {QUESTION}")
    answer = await agent.arun(QUESTION)

    print()
    for stream_id, parts in segments.items():
        print(f"streamed segment of stream {stream_id}: {''.join(parts)!r}")
    streamed = "".join("".join(parts) for parts in segments.values())
    print(f"Filtered final-answer feed: {streamed!r}")
    print(f"Returned final answer:      {answer!r}")
    print(f"streamed text equals the returned answer: {streamed == answer}")


async def main() -> None:
    print(
        f"Loading {MODEL_NAME} via the HuggingFaceAdapter (first run downloads weights)..."
    )
    llm = HuggingFaceAdapter(MODEL_NAME, stream=True, max_new_tokens=512)
    declared = "response_schema" in llm.get_model_capabilities().generation_options
    print(f"The model declares response_schema: {declared}")

    await run_once(llm, "constrained JSON action", constrained=True)
    await run_once(llm, "key-value text path", constrained=False)
    print(
        "\nIn each run the returned answer and the streamed turn came from the "
        "SAME model call - streaming is a consumption mode, not a second run."
    )


if __name__ == "__main__":
    asyncio.run(main())
