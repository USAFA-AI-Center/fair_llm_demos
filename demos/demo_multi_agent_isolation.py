# demo_multi_agent_isolation.py
"""
Multi-agent isolation with capability attenuation.

A real local model (HuggingFaceAdapter) drives a manager and an isolated
worker under a parent CapabilityBag. Shows:

  - Fresh memory each delegation (prior history does not leak)
  - Provenance-tagged WorkerSummaryResult (not a transcript)
  - Opt-in communicator handoff of the summary only
  - Child bag attenuates tools; ungranted tools are denied
  - Model axis: parent must grant the worker model id for spawn to succeed

Requirements: a local HuggingFace model (transformers; GPU recommended).
The first run downloads the weights. Set FAIR_LLM_DEMO_MODEL to override.

Run:
    python demos/demo_multi_agent_isolation.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from fairlib import (
    CapabilityBag,
    HuggingFaceAdapter,
    ReActPlanner,
    SafeCalculatorTool,
    SimpleAgent,
    ToolExecutor,
    ToolRegistry,
    WorkerAgentTool,
    WorkingMemory,
    build_worker_manager,
)
from fairlib.core.errors import CapabilityAttenuationError, ToolInvocationError
from fairlib.core.interfaces.llm import AbstractChatModel
from fairlib.core.interfaces.tools import (
    AbstractTool,
    SideEffect,
    StringInput,
    TextResult,
    ToolOutput,
)
from fairlib.core.message import Message
from fairlib.modules.agent.worker_tool import (
    WORKER_SUMMARY_MARKER_KEY,
    WorkerSubtaskInput,
)
from fairlib.modules.communication.in_memory_communicator import InMemoryCommunicator
from fairlib.modules.security.basic_security_manager import BasicSecurityManager

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen25-7b")


class _ShellTool(AbstractTool):
    """Ungranted tool used to demonstrate capability denial."""

    name = "shell"
    description = "Run a shell command (should be denied under the bag)."
    input_schema = StringInput
    output_schema = TextResult
    side_effect = SideEffect.EXTERNAL
    required_capability = "shell.exec"

    async def acall(self, tool_input: StringInput) -> ToolOutput:
        return TextResult(result="should-not-run")


def create_worker(llm: AbstractChatModel) -> SimpleAgent:
    registry = ToolRegistry()
    registry.register_tool(SafeCalculatorTool())
    registry.register_tool(_ShellTool())
    return SimpleAgent(
        llm,
        ReActPlanner(llm, registry),
        ToolExecutor(registry, BasicSecurityManager(capability_bag=None)),
        WorkingMemory(),
        stateless=True,
    )


class _ShellProbeWorker(SimpleAgent):
    """Demo-local worker that only probes shell denial under the child bag."""

    async def arun(self, user_input, **kwargs):  # type: ignore[override]
        try:
            await self.tool_executor.aexecute("shell", "echo pwn")
            return "shell-leaked"
        except ToolInvocationError as exc:
            return f"denied:{exc.kind.value}"


def _shell_probe_worker(llm: AbstractChatModel) -> _ShellProbeWorker:
    registry = ToolRegistry()
    registry.register_tool(SafeCalculatorTool())
    registry.register_tool(_ShellTool())
    return _ShellProbeWorker(
        llm,
        ReActPlanner(llm, registry),
        ToolExecutor(registry, BasicSecurityManager(capability_bag=None)),
        WorkingMemory(),
        stateless=True,
    )


async def main() -> int:
    print(f"Loading model {MODEL_NAME!r} ...")
    llm = HuggingFaceAdapter(model_name=MODEL_NAME)
    model_id = llm.describe_config().model_name

    parent_bag = CapabilityBag.from_grants(
        tools=["safe_calculator", "researcher"],
        models=[model_id],
    )

    worker = create_worker(llm)
    # Plant prior history that isolation must erase on the first delegation.
    worker.memory.add_message(Message(role="user", content="SECRET_PRIOR"))

    researcher = WorkerAgentTool(
        worker,
        name="researcher",
        description=(
            "Isolated research worker with a calculator. "
            "Use for arithmetic subtasks only."
        ),
        side_effect=SideEffect.READ_ONLY,
    )

    communicator = InMemoryCommunicator()
    manager = build_worker_manager(
        llm,
        [researcher],
        capability_bag=parent_bag,
        communicator=communicator,
        max_steps=6,
    )

    print("\n--- Delegation under capability bag ---")
    answer = await manager.arun(
        "Use the researcher worker to compute 17+25. Reply with only the number."
    )
    print("manager answer:", answer)

    print("\n--- Communicator summary handoff ---")
    msg = communicator.receive_message("manager")
    delegated = msg is not None
    if msg is None:
        print("communicator: no summary (manager may not have delegated)")
    else:
        print("communicator content:", msg.content)
        print(
            "communicator marker:",
            msg.metadata.get(WORKER_SUMMARY_MARKER_KEY),
        )
        print(
            "provenance:",
            msg.metadata.get("worker_name"),
            msg.metadata.get("delegation_id"),
            msg.metadata.get("ok"),
        )

    history = worker.memory.get_history()
    leaked = any("SECRET_PRIOR" in (m.content or "") for m in history)
    print("prior history leaked into worker:", leaked)
    if not delegated:
        print("isolation: SKIP (no WorkerAgentTool delegation observed)")
    elif leaked:
        print("isolation: FAIL")
        return 1
    else:
        print("isolation: prior history cleared on delegation")

    print("\n--- Shell denial under attenuated bag ---")
    probe = WorkerAgentTool(
        _shell_probe_worker(llm),
        name="probe",
        description="Probe worker.",
        parent_bag=parent_bag,
    )
    result = await probe.acall(WorkerSubtaskInput(subtask="probe"))
    print("shell probe:", result.summary)
    print(
        "provenance:",
        result.worker_name,
        result.delegation_id,
        result.ok,
    )
    if "denied" not in result.summary:
        print("shell denial: FAIL")
        return 1

    print("\n--- Widen attempt must fail at spawn ---")
    widen = CapabilityBag.from_grants(
        tools=["safe_calculator", "shell"],
        models=[model_id],
    )
    bad = WorkerAgentTool(
        create_worker(llm),
        name="bad_worker",
        description="Tries to widen.",
        parent_bag=parent_bag,
        worker_bag=widen,
    )
    try:
        await bad.acall(WorkerSubtaskInput(subtask="nope"))
        print("widen: UNEXPECTED SUCCESS")
        return 1
    except CapabilityAttenuationError as exc:
        print("widen blocked:", type(exc).__name__)

    print("\ndemo_multi_agent_isolation: OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
