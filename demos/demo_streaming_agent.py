# demos/demo_streaming_agent.py
"""
Stream an agent's model calls end-to-end as typed events.

A real local model runs one ReAct turn with streaming enabled. Two
consumers subscribe to the same run:

  1. A raw-feed consumer prints every ModelStreamChunkEvent delta as it
     arrives - the whole completion, Thought and Action included, the
     way a trace or debugging surface would watch it.
  2. A chat-surface consumer pipes the same deltas through
     KVFinalAnswerStreamFilter and prints only the final-answer text -
     the way a chat UI or an avatar app would speak it.

The run's return value is identical to a non-streaming run; streaming
changes how the model call is consumed and observed, not the result.

Requirements: a local HuggingFace model (transformers; a GPU is
recommended). The first run downloads the weights. Set
FAIR_LLM_DEMO_MODEL to override the default.

Run:
    python demos/demo_streaming_agent.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fairlib import (
    AgentEventBus,
    HuggingFaceAdapter,
    KVFinalAnswerStreamFilter,
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
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

QUESTION = "What is 6 times 7? Use the calculator tool, then answer in one sentence."


def build_agent(llm, bus: AgentEventBus) -> SimpleAgent:
    tools = ToolRegistry()
    tools.register_tool(SafeCalculatorTool())
    planner = SimpleReActPlanner(llm, tools, stream=True)
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


async def main() -> None:
    print(f"Loading {MODEL_NAME} via the HuggingFaceAdapter (first run downloads weights)...")
    llm = HuggingFaceAdapter(MODEL_NAME, stream=True, max_new_tokens=512)

    bus = AgentEventBus()
    agent = build_agent(llm, bus)

    # Consumer 1: the raw feed. Every delta of every stream, tagged by
    # stream so overlapping streams (a retry, the validator rewrite)
    # stay distinguishable.
    def on_start(event: ModelStreamStartEvent) -> None:
        mode = "simulated" if event.simulated else "live"
        print(f"\n--- stream {event.stream_id} started "
              f"({event.source.value}, step={event.step}, {mode}) ---")
        print("raw feed: ", end="", flush=True)

    def on_chunk(event: ModelStreamChunkEvent) -> None:
        print(event.text, end="", flush=True)

    def on_end(event: ModelStreamEndEvent) -> None:
        print(f"\n--- stream {event.stream_id} ended: {event.finish_reason.value}, "
              f"{event.chunk_count} chunks, {event.total_chars} chars ---")

    bus.subscribe(ModelStreamStartEvent, on_start)
    bus.subscribe(ModelStreamChunkEvent, on_chunk)
    bus.subscribe(ModelStreamEndEvent, on_end)

    # Consumer 2: the chat surface. Streams are ROUTED BY SOURCE: a
    # PLANNER stream carries the KV wire format and goes through one
    # filter per stream; a VALIDATOR_REWRITE stream is already plain
    # answer text and passes through raw - KV-filtering it would yield
    # nothing. (A KV-instructed model may also fall back to the parser's
    # secondary JSON shape, which this filter cannot see; a real chat
    # surface treats "run returned an answer but the filter emitted
    # nothing" as fall back to the returned text.)
    filters: dict[int, KVFinalAnswerStreamFilter] = {}
    answer_parts: list[str] = []

    def chat_on_start(event: ModelStreamStartEvent) -> None:
        if event.source is StreamSource.PLANNER:
            filters[event.stream_id] = KVFinalAnswerStreamFilter()

    def chat_on_chunk(event: ModelStreamChunkEvent) -> None:
        if event.source is StreamSource.PLANNER:
            text = filters[event.stream_id].feed(event.text)
        else:
            text = event.text
        if text:
            answer_parts.append(text)

    def chat_on_end(event: ModelStreamEndEvent) -> None:
        if event.source is StreamSource.PLANNER:
            tail = filters.pop(event.stream_id).finish()
            if tail:
                answer_parts.append(tail)

    bus.subscribe(ModelStreamStartEvent, chat_on_start)
    bus.subscribe(ModelStreamChunkEvent, chat_on_chunk)
    bus.subscribe(ModelStreamEndEvent, chat_on_end)

    print(f"\nQuestion: {QUESTION}")
    answer = await agent.arun(QUESTION)

    print(f"\nFiltered final-answer feed: {''.join(answer_parts)!r}")
    print(f"Returned final answer:      {answer!r}")
    print("\nThe returned answer and the streamed turn came from the SAME "
          "model call - streaming is a consumption mode, not a second run.")


if __name__ == "__main__":
    asyncio.run(main())
