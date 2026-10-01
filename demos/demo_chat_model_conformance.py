# demo_chat_model_conformance.py
"""
Run the chat-model conformance suite over a real local adapter, then over
a broken twin, so the failure a report shows is a real one.

An adapter is the unit that makes fairlib provider-agnostic, and the
conformance suite is the green/red proof that it honors the
AbstractChatModel contract, one check per clause of that class's
docstring: the declared surface and construction, a typed config
description, an assistant Message from every call, the system turn and
every other turn reaching the provider payload in order, a late system
message refused on every entry, tool turns carried or refused as declared,
options carried or refused, usage on every reply and on the last stream
chunk, one request and one invocation event per call on every bound bus,
and failures on the typed channel. The suite never raises; it returns a
report you read.

The demo loads one local Hugging Face model, hooks its tokenizer and its
generation pipeline so the suite can see the whole payload each call
sends (the messages and the generation options), and prints the report. It then
wraps the same model in a twin that silently drops system turns - the
classic quiet divergence the suite exists to catch - and prints that
report. The twin fails six checks, all real: system_message_delivery and
message_order_delivery (the dropped system turn); non_leading_system_refusal
(a late system message is dropped instead of refused); construction (the
wrapper takes the model it wraps, not model_name, timeout and options, so
settings cannot build it); and model_invocation_event and request_event,
because a wrapper that does not forward the event bus it is bound to onto
the model it wraps hides every model call, and every request it sends,
from the agent's observers (so bus_fan_out has nothing to observe and
reports skipped). Skipped lines are honest: a live model's responses are not
deterministic, so the content-comparison checks report skipped rather
than passing on evidence they do not have. construction_stop_merge reports
skipped as well: proving that a call's stop adds to a stop the model was
built with takes a second model built with one (the case's
construction_stop_case), and the demo loads the weights once.

Set FAIR_LLM_DEMO_MODEL to pick the local model (a settings.yml alias or
a Hugging Face model id).
"""

import contextlib
import os
from typing import Any, AsyncIterator, Iterator, List

from fairlib import (
    AbstractChatModel,
    ChatModelConformanceCase,
    HuggingFaceAdapter,
    Message,
    ModelCapabilities,
    ModelDescription,
    check_chat_model_conformance,
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen25-7b")


class SystemDroppingTwin(AbstractChatModel):
    """The same model behind a wrapper that quietly drops system turns.

    One adapter diverging like this is why the suite checks delivery: the
    agent's role definition simply stops arriving, and nothing raises. The
    twin also leaves bind_event_bus at the base behavior, so the bus lands
    on the twin while the wrapped model keeps emitting on its own; the
    suite reports that on both event checks.
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

    def get_model_capabilities(self) -> ModelCapabilities:
        return self._inner.get_model_capabilities()

    def describe_config(self) -> ModelDescription:
        return self._inner.describe_config()

    def estimate_token_count(self, text: str) -> int:
        return self._inner.estimate_token_count(text)


def build_case(
    model: AbstractChatModel, adapter: HuggingFaceAdapter, label: str
) -> ChatModelConformanceCase:
    """A full case over a live model: every hook set, determinism declared off.

    On this adapter one request has two halves: the chat template call
    carries the messages, and the generation pipeline call carries the
    output budget and the sampling options. The recorded-request hook
    returns both, so the suite sees everything the adapter sends. The
    failure hook swaps the generation pipeline for one that fails the way
    a real out-of-memory does.
    """
    prompts: list = []
    generations: list = []
    templated = adapter.tokenizer.apply_chat_template
    pipeline = adapter.generator

    def recording_template(conversation, *args: Any, **kwargs: Any):
        prompts.append(conversation)
        return templated(conversation, *args, **kwargs)

    def recording_generator(*args: Any, **kwargs: Any):
        generations.append(kwargs)
        return pipeline(*args, **kwargs)

    adapter.tokenizer.apply_chat_template = recording_template
    adapter.generator = recording_generator

    def recorded_request() -> object:
        if not prompts:
            return None
        return {
            "prompt": prompts[-1],
            "generation": generations[-1] if generations else None,
        }

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
        recorded_request=recorded_request,
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
