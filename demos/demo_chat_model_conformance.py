# demo_chat_model_conformance.py
"""
Run the chat-model conformance suite over a real local adapter, then over
a broken twin, so the failure a report shows is a real one.

An adapter is the unit that makes fairlib provider-agnostic, and the
conformance suite is the green/red proof that it honors the
AbstractChatModel contract: the declared surface, a typed config
description, an assistant Message from every call, the system turn
reaching the provider payload, statelessness across identical calls,
streaming that agrees with the one-shot call, and failures on the typed
channel. The suite never raises; it returns a report you read.

The demo loads one local Hugging Face model, hooks its tokenizer so the
suite can see the payload each call sends, and prints the report. It then
wraps the same model in a twin that silently drops system turns - the
classic quiet divergence the suite exists to catch - and prints that
report. Skipped lines are honest: a live model's responses are not
deterministic, so the content-comparison checks report skipped rather
than passing on evidence they do not have.

Set FAIR_LLM_DEMO_MODEL to pick the local model (a settings.yml alias or
a Hugging Face model id).
"""

import contextlib
import os
from typing import Any, AsyncIterator, Dict, Iterator, List

from fairlib import (
    AbstractChatModel,
    ChatModelConformanceCase,
    HuggingFaceAdapter,
    Message,
    ModelDescription,
    check_chat_model_conformance,
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "dolphin3-qwen25-3b")


class SystemDroppingTwin(AbstractChatModel):
    """The same model behind a wrapper that quietly drops system turns.

    One adapter diverging like this is why the suite checks delivery: the
    agent's role definition simply stops arriving, and nothing raises.
    """

    def __init__(self, inner: AbstractChatModel) -> None:
        self._inner = inner

    def _filtered(self, messages: List[Message]) -> List[Message]:
        return [m for m in messages if m.role != "system"]

    def invoke(self, messages: List[Message], **kwargs: Any) -> Message:
        return self._inner.invoke(self._filtered(messages), **kwargs)

    async def ainvoke(self, messages: List[Message], **kwargs: Any) -> Message:
        return await self._inner.ainvoke(self._filtered(messages), **kwargs)

    def stream(self, messages: List[Message], **kwargs: Any) -> Iterator[Message]:
        return self._inner.stream(self._filtered(messages), **kwargs)

    def astream(self, messages: List[Message], **kwargs: Any) -> AsyncIterator[Message]:
        return self._inner.astream(self._filtered(messages), **kwargs)

    def get_model_capabilities(self) -> Dict[str, Any]:
        return self._inner.get_model_capabilities()

    def describe_config(self) -> ModelDescription:
        return self._inner.describe_config()

    def estimate_token_count(self, text: str) -> int:
        return self._inner.estimate_token_count(text)


def build_case(
    model: AbstractChatModel, adapter: HuggingFaceAdapter, label: str
) -> ChatModelConformanceCase:
    """A full case over a live model: every hook set, determinism declared off.

    The recorded-request hook reads the last conversation the adapter
    templated, and the failure hook swaps the generation pipeline for one
    that fails the way a real out-of-memory does.
    """
    recorded: list = []
    templated = adapter.tokenizer.apply_chat_template

    def recording_template(conversation, *args: Any, **kwargs: Any):
        recorded.append(conversation)
        return templated(conversation, *args, **kwargs)

    adapter.tokenizer.apply_chat_template = recording_template

    def out_of_memory(*args: Any, **kwargs: Any):
        # The exception type a real exhausted GPU raises, so the adapter's
        # classification is exercised against the genuine failure shape.
        import torch

        raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated for the demo)")

    @contextlib.contextmanager
    def failure_mode():
        healthy = adapter.generator
        adapter.generator = out_of_memory
        try:
            yield
        finally:
            adapter.generator = healthy

    return ChatModelConformanceCase(
        model=model,
        recorded_request=lambda: recorded[-1] if recorded else None,
        failure_mode=failure_mode,
        deterministic=False,
        label=label,
    )


def run_suite(case: ChatModelConformanceCase) -> None:
    report = check_chat_model_conformance(case)
    print(report.render())
    if report.skipped:
        print(
            "(skipped means the suite could not observe that property on this "
            "transport; a skipped line is never a pass)"
        )
    print()


def main() -> None:
    print("=" * 70)
    print("FAIR-LLM: Chat-Model Conformance Suite Demo")
    print("=" * 70)
    print(f"\nLoading {MODEL_NAME}...")
    adapter = HuggingFaceAdapter(MODEL_NAME, stream=True, max_new_tokens=32)
    print("\n--- The real adapter ---")
    run_suite(build_case(adapter, adapter, label=f"{MODEL_NAME} (real adapter)"))
    print("--- The same model behind a twin that drops system turns ---")
    run_suite(
        build_case(
            SystemDroppingTwin(adapter),
            adapter,
            label=f"{MODEL_NAME} (system-dropping twin)",
        )
    )


if __name__ == "__main__":
    main()
