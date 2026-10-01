"""
This demo builds a miniature application on the agent event bus: a live
console monitor for a running agent, in the style of a flight recorder.

The application is a calculator agent you chat with, exactly like
demo_single_agent_calculator.py - except this one shows you what the agent
is doing WHILE it works. A ticker line appears for every step the agent
takes, every tool call (with timing), every parse stumble, and every memory
compaction. After any turn you can type 'trace' to replay the whole turn as
a structured timeline.

Every run opens with a short scripted exchange (three questions and a
'trace' command) so the ticker and the flight record are on screen even
when nobody is typing; after that the prompt is yours. Memory compacts
once the history passes eight messages, usually between the steps of a
turn, and each question is still the one answered: the request of the turn
in progress is never summarized away.

The point for an implementer: everything the monitor knows arrives through
bus subscriptions on typed event objects. It never reads framework
internals, never parses log text, never diffs memory lengths. This is the
pattern to copy for building telemetry, debugging UIs, audit logs, or a
web dashboard on top of fairlib.

Components flexed:
    agent.events / AgentEventBus  - one bus shared by agent AND memory
    AgentStepEvent                - one per reasoning step
    ToolCallPreEvent/PostEvent    - paired around every tool dispatch
    PlannerParseErrorEvent        - the model emitted something unusable
    LoopGuardTrippedEvent         - the agent looks stuck
    MemorySummarizedEvent         - SummarizingMemory compacted history

Requires a local model; defaults to HuggingFaceAdapter("qwen25-14b").
"""

import asyncio
import time

from fairlib import (
    AgentEventBus,
    AgentStepEvent,
    HuggingFaceAdapter,
    LoopGuardTrippedEvent,
    MemorySummarizedEvent,
    PlannerParseErrorEvent,
    RoleDefinition,
    SafeCalculatorTool,
    SimpleAgent,
    SimpleReActPlanner,
    SummarizingMemory,
    ToolCallPostEvent,
    ToolCallPreEvent,
    ToolExecutor,
    ToolRegistry,
)

# The opening every run plays before handing the prompt to you.
SCRIPTED_OPENING = (
    "What is 12.5 * 8?",
    "Take 144 divided by 12, then multiply that result by 7.",
    "What is 7 * 13 - 4?",
    "trace",
)


class AgentMonitor:
    """A live activity ticker built purely from bus subscriptions.

    Subscribes one small callback per event type and prints a compact
    line as each event arrives. Keeps the handles it gets back from
    subscribe() so the ticker can be muted (unsubscribed) and unmuted
    (resubscribed) while the agent keeps running - the monitor is a
    bolt-on, never a dependency.
    """

    def __init__(self, bus: AgentEventBus) -> None:
        self.bus = bus
        self._handles = []
        self._tool_started_at = {}
        self.attach()

    def attach(self) -> None:
        if self._handles:
            return
        self._handles = [
            self.bus.subscribe(AgentStepEvent, self._on_step),
            self.bus.subscribe(ToolCallPreEvent, self._on_tool_pre),
            self.bus.subscribe(ToolCallPostEvent, self._on_tool_post),
            self.bus.subscribe(PlannerParseErrorEvent, self._on_parse_error),
            self.bus.subscribe(LoopGuardTrippedEvent, self._on_loop_guard),
            self.bus.subscribe(MemorySummarizedEvent, self._on_summarized),
        ]

    def detach(self) -> None:
        for handle in self._handles:
            self.bus.unsubscribe(handle)
        self._handles = []

    def _on_step(self, event: AgentStepEvent) -> None:
        print(
            f"    | step {event.step + 1}/{event.max_steps} "
            f"(history: {event.history_length} messages)"
        )

    def _on_tool_pre(self, event: ToolCallPreEvent) -> None:
        self._tool_started_at[(event.step, event.call_index)] = time.monotonic()
        print(f"    |   -> {event.tool_name}({str(event.tool_input)[:50]})")

    def _on_tool_post(self, event: ToolCallPostEvent) -> None:
        started = self._tool_started_at.pop((event.step, event.call_index), None)
        elapsed = f" in {time.monotonic() - started:.2f}s" if started else ""
        outcome = "ok" if event.succeeded else f"FAILED: {event.error}"
        print(f"    |   <- {event.tool_name} {outcome}{elapsed}")

    def _on_parse_error(self, event: PlannerParseErrorEvent) -> None:
        retrying = "retrying with guidance" if event.will_retry else "giving up"
        print(f"    |   model output was unparseable; {retrying}")

    def _on_loop_guard(self, event: LoopGuardTrippedEvent) -> None:
        # guard_type is a GuardType enum; .value is the friendly string.
        print(
            f"    |   loop guard '{event.guard_type.value}' tripped "
            f"({event.count} >= {event.threshold}) - agent may be stuck"
        )

    def _on_summarized(self, event: MemorySummarizedEvent) -> None:
        print(
            f"    |   memory compacted: {len(event.dropped)} messages "
            f"summarized, {len(event.kept)} kept"
        )


class FlightRecorder:
    """A per-turn structured record built from the same bus.

    Where AgentMonitor prints and forgets, the recorder keeps every event
    object. After the turn you can render the timeline - which is exactly
    how a production consumer would persist a trace for later audit. The
    events are frozen dataclasses, so what you store is what happened.
    """

    def __init__(self, bus: AgentEventBus) -> None:
        self.turn_events = []
        for event_type in (
            AgentStepEvent,
            ToolCallPreEvent,
            ToolCallPostEvent,
            PlannerParseErrorEvent,
            LoopGuardTrippedEvent,
            MemorySummarizedEvent,
        ):
            bus.subscribe(event_type, self.turn_events.append)

    def new_turn(self) -> None:
        self.turn_events.clear()

    def render(self) -> None:
        if not self.turn_events:
            print("No events recorded yet - ask the agent something first.")
            return
        print(f"\nFlight record of the last turn ({len(self.turn_events)} events):")
        for i, event in enumerate(self.turn_events, start=1):
            print(f"  {i:2d}. {type(event).__name__:24s} {self._describe(event)}")

    @staticmethod
    def _describe(event) -> str:
        """The fields of one stored event worth reading in a timeline."""
        if isinstance(event, AgentStepEvent):
            return f"step={event.step} history_length={event.history_length}"
        if isinstance(event, ToolCallPreEvent):
            return f"step={event.step} {event.tool_name} input={event.tool_input!r}"
        if isinstance(event, ToolCallPostEvent):
            return (
                f"step={event.step} {event.tool_name} succeeded={event.succeeded} "
                f"observation={event.observation!r}"
            )
        if isinstance(event, MemorySummarizedEvent):
            summary = event.summary.content.replace("\n", " ")
            return (
                f"reason={event.reason.value} dropped={len(event.dropped)} "
                f"kept={len(event.kept)} summary={summary!r}"
            )
        if isinstance(event, PlannerParseErrorEvent):
            return (
                f"step={event.step} attempt={event.attempt} "
                f"will_retry={event.will_retry} raw_output={event.raw_output!r}"
            )
        if isinstance(event, LoopGuardTrippedEvent):
            return (
                f"step={event.step} guard={event.guard_type.value} "
                f"count={event.count} threshold={event.threshold}"
            )
        return repr(event)


async def main() -> None:
    print("Assembling a calculator agent with a live event monitor...\n")

    # One bus, shared by the agent AND its memory, so every event in the
    # system arrives on a single subscription surface. This is the
    # recommended wiring: pass the same bus to both constructors.
    bus = AgentEventBus()
    monitor = AgentMonitor(bus)
    recorder = FlightRecorder(bus)

    llm = HuggingFaceAdapter("qwen25-14b")

    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())
    planner = SimpleReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a precise assistant. Use the calculator for any arithmetic."
    )

    # A small max_history_length means a normal conversation will trigger
    # compaction within a few turns, so you can watch the
    # MemorySummarizedEvent arrive on the same ticker as the loop events.
    # Compaction may fire between the steps of a turn; the question being
    # answered is kept verbatim, so the agent still answers the right one.
    memory = SummarizingMemory(
        llm=llm,
        max_history_length=8,
        messages_to_keep_at_end=3,
        events=bus,
    )

    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=memory,
        max_steps=6,
        events=bus,
    )

    async def handle(user_input: str) -> bool:
        """Run one line of input; return False when the session should end."""
        command = user_input.lower()
        if command in ("exit", "quit"):
            print("Goodbye!")
            return False
        if command == "trace":
            recorder.render()
            return True
        if command == "mute":
            monitor.detach()
            print("(ticker muted - the agent still emits, nobody is listening)")
            return True
        if command == "unmute":
            monitor.attach()
            print("(ticker reattached)")
            return True

        recorder.new_turn()
        try:
            response = await agent.arun(user_input)
            print(f"Agent: {response}\n")
        except Exception as exc:
            # Framework errors are typed; a real application would branch
            # on the specific FairlibError subclass here.
            print(f"Agent could not finish: {type(exc).__name__}: {exc}\n")
        return True

    print("Agent ready. Ask it math questions and watch the ticker.")
    print("Commands:  trace  - replay the last turn as a timeline")
    print("           mute   - detach the ticker (agent runs silently)")
    print("           unmute - reattach the ticker")
    print("           exit   - quit\n")

    # The scripted opening: three turns the ticker narrates live, then the
    # flight record of the third turn.
    for scripted in SCRIPTED_OPENING:
        print(f"You (scripted): {scripted}")
        await handle(scripted)

    print("\nYour turn.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if not await handle(user_input):
            break


if __name__ == "__main__":
    asyncio.run(main())
