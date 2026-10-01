# demo_request_shape_across_routes.py
"""
One agent, three providers, one request shape.

fairlib sends every model the same portable conversation: one leading system
message, then user and assistant turns in order, with a tool's observation
reaching the model as a user turn in position and joined with the
continuation prompt into one turn. No adapter moves, folds or drops a turn,
and a system message after the conversation begins is refused rather than
sent. This demo shows that from the outside.

It builds the same calculator agent (the same planner class, the same one
tool, the same question) on three routes: a Hugging Face model loaded with
transformers, a model served by Ollama, and Google Gemini. It subscribes to
each agent's event bus and prints, for every model call, the role sequence
the provider received (ModelRequestEvent). It then prints the three
second-call sequences side by side with a word saying whether they are
identical, and the exact prompt text the two local models read for their
second call (the adapters are built with capture_rendered_prompt=True): the
observation sits in a user turn after the assistant turn, the continuation
follows it in the same turn, and the only system turn is the first.

Set FAIR_LLM_DEMO_MODEL for the Hugging Face model (default qwen25-7b),
FAIR_LLM_DEMO_OLLAMA for the Ollama model (default qwen2.5:14b, served at
localhost:11434) and FAIR_LLM_DEMO_GEMINI for the Gemini model (default
gemini-3.6-flash). GEMINI_API_KEY must be set; without it the demo prints
the adapter's typed refusal and exits non-zero.

Run: PYTHONPATH=. python demos/demo_request_shape_across_routes.py
"""

import asyncio
import os
import sys
from typing import Dict, List, Tuple

from fairlib import (
    AbstractChatModel,
    FairlibError,
    GeminiAdapter,
    HuggingFaceAdapter,
    ModelRequestEvent,
    OllamaAdapter,
    RoleDefinition,
    SafeCalculatorTool,
    SimpleAgent,
    SimpleReActPlanner,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
)
from fairlib.core.events import ToolBatchScheduledEvent

HF_MODEL = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen25-7b")
OLLAMA_MODEL = os.environ.get("FAIR_LLM_DEMO_OLLAMA", "qwen2.5:14b")
GEMINI_MODEL = os.environ.get("FAIR_LLM_DEMO_GEMINI", "gemini-3.6-flash")

QUESTION = "What is 17 * 23? Use the calculator."


def build_routes() -> List[Tuple[str, AbstractChatModel]]:
    """The only place a provider is named; everything below is blind to it."""
    gemini = GeminiAdapter(
        model_name=GEMINI_MODEL, timeout=60, options={"temperature": 0.0}
    )
    ollama = OllamaAdapter(
        model_name=OLLAMA_MODEL,
        timeout=300,
        options={"temperature": 0.0},
        capture_rendered_prompt=True,
    )
    print(f"Loading {HF_MODEL} via HuggingFaceAdapter...")
    hf = HuggingFaceAdapter(
        HF_MODEL, max_new_tokens=256, do_sample=False, capture_rendered_prompt=True
    )
    return [("huggingface", hf), ("ollama", ollama), ("gemini", gemini)]


def build_agent(llm: AbstractChatModel) -> SimpleAgent:
    registry = ToolRegistry()
    registry.register_tool(SafeCalculatorTool())
    planner = SimpleReActPlanner(llm, registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a careful calculator assistant. Use the safe_calculator tool "
        "for arithmetic, then give the result as your final answer."
    )
    return SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(registry),
        memory=WorkingMemory(),
        max_steps=4,
    )


async def run_route(
    name: str, llm: AbstractChatModel
) -> Tuple[List[ModelRequestEvent], List[ToolBatchScheduledEvent], str]:
    agent = build_agent(llm)
    requests: List[ModelRequestEvent] = []
    batches: List[ToolBatchScheduledEvent] = []
    agent.events.subscribe(ModelRequestEvent, requests.append)
    agent.events.subscribe(ToolBatchScheduledEvent, batches.append)
    print(f"\n=== {name} ===")
    answer = await agent.arun(QUESTION)
    for index, request in enumerate(requests, start=1):
        print(f"  call {index}: {', '.join(request.roles)}")
    for batch in batches:
        groups = "; ".join(
            f"{'parallel' if g.parallel else 'sequential'} {', '.join(g.tool_names)}"
            for g in batch.groups
        )
        print(f"  tool batch at step {batch.step}: {groups}")
    print(f"  final answer: {answer}")
    return requests, batches, answer


async def main() -> int:
    try:
        routes = build_routes()
    except FairlibError as exc:
        print(f"A route could not be built: {type(exc).__name__}: {exc}")
        return 1
    print(f"\nQuestion on every route: {QUESTION}")
    seen: Dict[str, List[ModelRequestEvent]] = {}
    for name, llm in routes:
        requests, _, _ = await run_route(name, llm)
        seen[name] = requests

    print("\n=== The second call on every route ===")
    second = {
        name: requests[1].roles for name, requests in seen.items() if len(requests) > 1
    }
    for name, roles in second.items():
        print(f"  {name:<12} {', '.join(roles)}")
    verdict = "identical" if len(set(second.values())) == 1 else "different"
    print(f"  the sequences are {verdict} across {len(second)} routes")

    for name in ("huggingface", "ollama"):
        requests = seen.get(name, [])
        if len(requests) > 1:
            print(f"\n=== What the {name} model read for its second call ===")
            print(requests[1].rendered)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
