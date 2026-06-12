"""
This demo builds a miniature application on the v0.3.2 memory surfaces: a
trip-planning assistant that never forgets your non-negotiables.

The problem this solves for an implementer: a long conversation eventually
overflows the context window, so SummarizingMemory compresses old turns
into a summary. Compression is lossy - and some messages must never be
lossy. A budget ceiling. An accessibility requirement. The user's name.
If those land in the summary blob, the agent starts violating them and the
user notices immediately.

The fix is two coordinated primitives, both flexed here:

    Message(importance="pinned")
        The caller's "this must survive verbatim" signal. Pinned messages
        are excluded from summarization and reinserted in their original
        positions. agent.arun() accepts a full Message, so the user turn
        itself can be pinned at the callsite.

    MemorySummarizedEvent via SummarizingMemory(events=bus)
        Every compaction announces itself on the event bus: what was
        dropped, what was kept, the summary text. The assistant below uses
        it to tell the user their constraints were carried forward -
        no guessing, no inspecting memory internals.

The session starts with a short scripted itinerary-planning conversation
(so compaction actually happens), then hands the keyboard to you.

Requires a local model; defaults to HuggingFaceAdapter("qwen25-14b").
"""

import asyncio

from fairlib import (
    AgentEventBus,
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


def announce_compaction(event: MemorySummarizedEvent) -> None:
    """Turn the compaction event into a user-facing reassurance.

    This is the consumer pattern the event exists for: the application
    decides what a summarization means to its user. Here we surface it;
    a quieter application might log it; a web UI might badge it.
    """
    pinned_kept = sum(1 for m in event.kept if m.importance == "pinned")
    preview = event.summary.content[:100].replace("\n", " ")
    # reason is a SummarizationReason enum; .value is the friendly string.
    print(
        f"\n  [memory] Conversation compacted ({event.reason.value}): "
        f"{len(event.dropped)} older messages folded into a summary, "
        f"{len(event.kept)} kept verbatim - including your "
        f"{pinned_kept} pinned constraint(s).\n"
        f"  [memory] Summary now reads: {preview!r}\n"
    )


def show_memory(memory: SummarizingMemory) -> None:
    """Print the live history with pinned messages marked."""
    print(f"\nMemory currently holds {len(memory.history)} messages:")
    for i, m in enumerate(memory.history):
        marker = "[pinned]" if m.importance == "pinned" else "        "
        preview = m.content[:84].replace("\n", " ")
        print(f"  {i:2d}. {marker} {m.role:>9}: {preview}")
    print()


async def main() -> None:
    print("Assembling a trip-planning agent with pinned-constraint memory...\n")

    llm = HuggingFaceAdapter("qwen25-14b")

    bus = AgentEventBus()
    bus.subscribe(MemorySummarizedEvent, announce_compaction)

    # max_history_length is deliberately small so you can watch
    # summarization happen within one short planning session.
    memory = SummarizingMemory(
        llm=llm,
        max_history_length=8,
        messages_to_keep_at_end=3,
        events=bus,
    )

    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())
    planner = SimpleReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a budget-aware trip planner. Use the calculator for any "
        "arithmetic. Respect every HARD CONSTRAINT you have been given. "
        "Always finish your turn with a clear final answer."
    )

    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=memory,
        max_steps=4,
    )

    # --- Scripted opening: a realistic planning session -------------------
    # The two constraints are sent as pinned Messages. Everything else is a
    # plain string turn, free to be summarized away. By the final scripted
    # turn the history exceeds max_history_length, compaction fires, and
    # the constraints are still in memory verbatim - which is why the
    # agent can still do the budget math correctly afterward.

    opening_turns = [
        "Hi! Help me plan a 3-stop weekend trip in San Francisco.",
        Message(
            role="user",
            content="HARD CONSTRAINT: my total budget is 1500 USD for two people.",
            importance="pinned",
        ),
        Message(
            role="user",
            content="HARD CONSTRAINT: every stop must be wheelchair accessible.",
            importance="pinned",
        ),
        "Stop 1: suggest an aquarium. Admission is 30 USD per person for the two of us.",
        "Stop 2: a coffee shop nearby. Two drinks at 12 USD each.",
        "Stop 3: a museum at 25 USD per person. Now total all three stops "
        "for both of us and tell me how much of the budget remains.",
    ]

    for turn in opening_turns:
        text = turn.content if isinstance(turn, Message) else turn
        pinned = isinstance(turn, Message) and turn.importance == "pinned"
        tag = "  (pinned)" if pinned else ""
        print(f"You{tag}: {text}")
        try:
            response = await agent.arun(turn)
            print(f"Agent: {str(response)[:300]}\n")
        except Exception as exc:
            print(f"Agent could not finish: {type(exc).__name__}: {exc}\n")

    show_memory(memory)
    print(
        "Note what survived: the conversation has been compacted at least "
        "once, yet both HARD CONSTRAINT messages are still there verbatim, "
        "at full fidelity, while ordinary turns became a summary."
    )

    # --- Your turn ---------------------------------------------------------
    print("\nThe session is now yours. Keep planning, or stress the memory.")
    print("Commands:  pin: <text>  - send a turn as a pinned constraint")
    print("           memory       - show history with pinned markers")
    print("           exit         - quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if user_input.lower() == "memory":
            show_memory(memory)
            continue

        if user_input.lower().startswith("pin:"):
            turn = Message(
                role="user",
                content=user_input[4:].strip(),
                importance="pinned",
            )
            print("  (this turn is pinned - it will survive every compaction)")
        else:
            turn = user_input

        try:
            response = await agent.arun(turn)
            print(f"Agent: {response}\n")
        except Exception as exc:
            print(f"Agent could not finish: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())
