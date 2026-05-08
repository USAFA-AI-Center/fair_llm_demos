"""
This script exercises every public surface added in v0.3.2. The new surfaces, 
all imported from the top-level fairlib namespace:

  Message(importance="pinned")
      Caller-controlled signal that this message must survive summarization
      verbatim. SummarizingMemory excludes pinned messages from the
      summarization input and reinserts them into the returned history in
      their original relative order. WorkingMemory ignores the field.

  MemorySummarizedEvent
      Frozen dataclass payload fired exactly once per summarization, carrying
      dropped, kept, summary, and reason. reason is one of
      "max_history_length_exceeded" or "adapter_failure_fallback".

  SummarizingMemory(on_summarize=callback)
      Optional sync callback registered at construction. Fires after the
      history list has been mutated to its new state. Subscriber exceptions
      are caught and logged so a misbehaving consumer cannot break the agent
      loop.

  SimpleAgent.arun(user_input: str | Message)
      The user-turn entry point now accepts a pre-built Message so callers
      can pin the user turn directly. Passing a Message whose role is not
      "user" raises InvalidAgentInputError.

  Edge-triggered 80% pinning warning
      SummarizingMemory emits a logger.warning when the pinned message count
      reaches 80% of max_history_length. Caller still owns the
      context-window budget; the framework does not silently drop pinned
      messages.

The demo defaults to HuggingFaceAdapter("qwen25-14b")
"""

import asyncio
import logging
from typing import List

from fairlib import (
    HuggingFaceAdapter,
    MemorySummarizedEvent,
    Message,
    RoleDefinition,
    SafeCalculatorTool,
    SimpleAgent,
    SimpleReActPlanner,
    SummarizingMemory,
    ToolExecutor,
    ToolRegistry,
)


def _print_section(title: str) -> None:
    bar = "=" * max(len(title), 60)
    print(f"\n{bar}\n{title}\n{bar}")


def _format_event(evt: MemorySummarizedEvent) -> str:
    summary_preview = evt.summary.content[:180].replace("\n", " ")
    pinned_in_kept = sum(1 for m in evt.kept if m.importance == "pinned")
    return (
        f"  reason   = {evt.reason}\n"
        f"  dropped  = {len(evt.dropped)} message(s)\n"
        f"  kept     = {len(evt.kept)} message(s) (pinned among them: {pinned_in_kept})\n"
        f"  summary  = {summary_preview!r}"
    )

# TODO:: update to use permanent event bus class when implemented
class _EventRecorder:
    """Demo class, track events"""
    def __init__(self) -> None:
        self.events: List[MemorySummarizedEvent] = []

    def __call__(self, evt: MemorySummarizedEvent) -> None:
        self.events.append(evt)
        # A short inline notice helps the reader correlate the event with
        # surrounding agent log lines when stage 2 runs through a ReAct loop.
        print(
            f"\n[on_summarize fired] reason={evt.reason} "
            f"dropped={len(evt.dropped)} kept={len(evt.kept)}"
        )

# TODO:: update to use permanent event bus class when implemented
class _CapturingHandler(logging.Handler):
    """demo class, pretty prints"""
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


async def stage_one_direct_memory(llm) -> None:
    """Drives SummarizingMemory directly so the pinning contract is visible
    without any planner non-determinism between threshold and event.

    The transcript places two pinned messages inside the summarizable middle
    slice (not at index 0 where the first-anchor rule would preserve them
    regardless). After aget_history() runs, both pinned messages must appear
    in the returned history and the on_summarize callback must have fired
    exactly once.
    """
    _print_section("Stage 1: SummarizingMemory honors importance='pinned'")

    recorder = _EventRecorder()
    memory = SummarizingMemory(
        llm=llm,
        max_history_length=8,
        messages_to_keep_at_end=3,
        on_summarize=recorder,
    )

    pinned_budget = Message(
        role="user",
        content="HARD CONSTRAINT: total budget is 1500 USD.",
        importance="pinned",
    )
    pinned_access = Message(
        role="user",
        content="HARD CONSTRAINT: every stop must be wheelchair accessible.",
        importance="pinned",
    )

    transcript: List[Message] = [
        Message(role="user",      content="Hi! Help me plan a 3-stop weekend trip."),
        Message(role="assistant", content="Of course. What constraints should I respect?"),
        pinned_budget,
        Message(role="assistant", content="Got it: 1500 USD ceiling. Anything else?"),
        pinned_access,
        Message(role="assistant", content="Wheelchair accessibility noted. Continuing."),
        Message(role="user",      content="Stop 1: an aquarium that fits both."),
        Message(role="assistant", content="Aquarium of the Bay is accessible; admission is about 30 USD."),
        Message(role="user",      content="Stop 2: a coffee shop nearby."),
        Message(role="assistant", content="Sightglass on 7th Street, accessible entrance, about 12 USD per drink."),
        Message(role="user",      content="Stop 3: a museum."),
        Message(role="assistant", content="SFMOMA is accessible; admission is about 25 USD."),
    ]

    for m in transcript:
        memory.add_message(m)

    print(
        f"\nLoaded {len(memory.history)} messages with max_history_length=8. "
        "About to call aget_history(); summarization should fire."
    )

    history_after = await memory.aget_history()

    print(f"\nReturned history length: {len(history_after)}")
    for i, m in enumerate(history_after):
        marker = "[pinned]" if m.importance == "pinned" else "        "
        preview = m.content[:90].replace("\n", " ")
        print(f"  {i:2d}. {marker} {m.role:>9}: {preview}")

    assert recorder.events, "Expected on_summarize to fire at least once."
    print("\nMemorySummarizedEvent captured by the on_summarize callback:")
    print(_format_event(recorder.events[-1]))

    pinned_in_returned = [m for m in history_after if m.importance == "pinned"]
    assert len(pinned_in_returned) == 2, (
        f"Expected both pinned constraints to survive; got {len(pinned_in_returned)}."
    )
    print(
        f"\nBoth pinned constraints survived verbatim "
        f"(found {len(pinned_in_returned)} pinned messages in returned history)."
    )


async def stage_two_message_as_input(llm) -> None:
    """Drives SimpleAgent.arun with a Message argument so the new
    Message-as-input signature is exercised.

    Turn 1 is a plain string so the pinned constraint at turn 2 ends up at
    a non-zero history index and pinning, not first-anchor preservation, is
    what saves it across summarization.
    """
    _print_section("Stage 2: SimpleAgent.arun accepts Message(importance='pinned')")

    recorder = _EventRecorder()
    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())
    planner = SimpleReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a budget-aware trip planner. Use the calculator for any "
        "arithmetic. Respect every HARD CONSTRAINT you have been given. "
        "Always finish your turn with a clear final answer."
    )

    memory = SummarizingMemory(
        llm=llm,
        max_history_length=8,
        messages_to_keep_at_end=3,
        on_summarize=recorder,
    )
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=memory,
        max_steps=4,
    )

    # Turn 1 keeps the first-anchor slot occupied by a non-pinned message.
    print("\nturn 1 (string input): greeting ...")
    response = await agent.arun("Hi. I would like help planning a 3-stop trip.")
    print(f"  agent: {str(response)[:200]}")

    # Turn 2 is a Message with importance='pinned'. Before P1.1.1, agent.arun
    # only accepted a string and the pinning signal had no place to attach.
    constraint_msg = Message(
        role="user",
        content="HARD CONSTRAINT: my total budget is 1500 USD. Track this for the rest of our session.",
        importance="pinned",
    )
    print("\nturn 2 (Message input, importance='pinned'): budget constraint ...")
    response = await agent.arun(constraint_msg)
    print(f"  agent: {str(response)[:200]}")

    follow_ups = [
        "Stop 1 admission is 30 USD per person and we are 2 people.",
        "Stop 2 is 12 USD per drink and each of us has 2 drinks.",
        "Stop 3 admission is 25 USD per person for 2 people. Sum every stop's "
        "total cost and tell me whether it stays under the budget.",
    ]
    for i, txt in enumerate(follow_ups, start=3):
        print(f"\nturn {i}: {txt[:80]} ...")
        response = await agent.arun(txt)
        print(f"  agent: {str(response)[:200]}")

    print(
        f"\nMemory.history is now {len(memory.history)} messages. "
        f"Summarization fired {len(recorder.events)} time(s) during this run."
    )

    pinned_remaining = [m for m in memory.history if m.importance == "pinned"]
    print(f"Pinned messages still present in memory.history: {len(pinned_remaining)}")
    for m in pinned_remaining:
        print(f"  * {m.role}: {m.content[:120]}")

    if recorder.events:
        print("\nLast MemorySummarizedEvent emitted during the agent loop:")
        print(_format_event(recorder.events[-1]))


def stage_three_overpinning_warning(llm) -> None:
    """Triggers the edge-triggered 80% over-pinning warning by adding more
    pinned messages than 80% of max_history_length permits.

    No LLM calls are made in this stage; the threshold check fires inside
    add_message before any summarization is attempted.
    """
    _print_section("Stage 3: edge-triggered 80% over-pinning warning")

    handler = _CapturingHandler()
    summarization_logger = logging.getLogger("fairlib.modules.memory.summarization")
    summarization_logger.addHandler(handler)
    # Ensure the warning level is reachable even if the root logger is set
    # higher elsewhere. We restore the prior level in finally below.
    prior_level = summarization_logger.level
    summarization_logger.setLevel(logging.WARNING)

    try:
        memory = SummarizingMemory(
            llm=llm,
            max_history_length=10,
            messages_to_keep_at_end=2,
        )
        for i in range(8):
            memory.add_message(
                Message(role="user", content=f"Pin {i}", importance="pinned")
            )

        warnings = [r for r in handler.records if "Pinned message count" in r.getMessage()]
        print(f"\nWarning lines captured: {len(warnings)}")
        for r in warnings:
            print(f"  WARNING {r.name}: {r.getMessage()}")
        print(
            "\nThe warning is edge-triggered: re-adding more pinned messages "
            "without the count first dropping back below 80% does NOT emit a "
            "second warning. Caller still owns the context-window budget; the "
            "framework does not silently drop pinned messages."
        )
    finally:
        summarization_logger.removeHandler(handler)
        summarization_logger.setLevel(prior_level)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    llm = HuggingFaceAdapter("qwen25-14b")

    await stage_one_direct_memory(llm)
    await stage_two_message_as_input(llm)
    stage_three_overpinning_warning(llm)

    _print_section("Demo complete")
    print(
        "Exercised: importance='pinned' on Message, MemorySummarizedEvent via "
        "the on_summarize hook, agent.arun(Message(...)), and the 80% pinning "
        "warning."
    )


if __name__ == "__main__":
    asyncio.run(main())
