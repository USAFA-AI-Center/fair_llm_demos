"""Demo: degenerate-output recovery on planner parse failure."""

import json

from fairlib.core.event_bus import AgentEventBus
from fairlib.core.events import DegenerateOutputTrimmedEvent
from fairlib.modules.planning.multi_action_planner import MultiActionReActPlanner


def main() -> None:
    action = {
        "thought": "compute",
        "actions": [{"tool_name": "calc", "tool_input": "1+1"}],
    }
    duplicated = json.dumps(action) + json.dumps(action)

    bus = AgentEventBus()
    events: list[DegenerateOutputTrimmedEvent] = []
    bus.subscribe(DegenerateOutputTrimmedEvent, events.append)

    planner = MultiActionReActPlanner.__new__(MultiActionReActPlanner)
    planner._max_actions_per_turn = 16
    planner._sanitizer_enabled = True
    planner._provider = "ollama"
    planner._event_bus = bus

    result = planner._parse_response(duplicated)
    thought, parsed = result
    print("Recovered thought:", thought.text)
    print("Recovered action:", parsed.tool_name, parsed.tool_input)
    print("Trim event emitted:", len(events) == 1)
    if events:
        print("Original length:", events[0].original_length)
        print("Retained length:", events[0].retained_length)
    print("Degenerate output sanitizer demo complete.")


if __name__ == "__main__":
    main()
