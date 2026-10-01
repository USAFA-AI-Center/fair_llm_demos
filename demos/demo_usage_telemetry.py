# demo_usage_telemetry.py
"""
Read the per-call telemetry a fairlib adapter attaches to every reply.

An adapter surfaces the provider's usage block on every reply. The Message
it returns carries a Usage record: prompt and completion token counts,
the call's duration, the model, and why generation stopped - a normalized
DoneReason plus the raw provider string. The caller reads it off the
reply, the same object it already holds; nothing touches a provider
payload shape.

The demo makes one ordinary call and prints what it cost, then forces a
truncation (a tiny output budget) so the model stops mid-answer. That
second call is the point: a length cutoff arrives as done_reason LENGTH,
an observable typed signal, instead of a 200 response with quietly
truncated content that a pipeline would mistake for a complete answer.

Set FAIR_LLM_DEMO_MODEL to pick the local model served by Ollama.
"""

import os
from typing import Optional

from fairlib import DoneReason, Message, OllamaAdapter, Usage

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen3-vl-instruct-16k")


def show(usage: Optional[Usage]) -> None:
    if usage is None:
        print("  (this backend reported no usage)")
        return
    print(f"  prompt tokens:     {usage.prompt_tokens}")
    print(f"  completion tokens: {usage.completion_tokens}")
    print(f"  total duration:    {usage.total_duration_ms:.0f} ms")
    print(f"  model:             {usage.model}")
    done_reason = usage.done_reason.value if usage.done_reason is not None else None
    print(f"  done_reason:       {done_reason} (raw: {usage.raw_done_reason!r})")


def main() -> None:
    print(f"Model: {MODEL_NAME} via Ollama\n")

    adapter = OllamaAdapter(model_name=MODEL_NAME)

    print("A normal call - the reply carries what it cost and why it stopped:")
    reply = adapter.invoke([Message(role="user", content="Name three primary colors.")])
    print(f'  reply: "{reply.content}"')
    show(reply.usage)

    print(
        "\nThe same question with a two-token output budget - the model is cut"
        "\noff mid-answer, and the truncation is not silent:"
    )
    truncated = adapter.invoke(
        [Message(role="user", content="Name three primary colors.")],
        num_predict=2,
    )
    print(f'  reply: "{truncated.content}"')
    show(truncated.usage)

    if truncated.usage is not None and truncated.usage.done_reason is DoneReason.LENGTH:
        print(
            "\ndone_reason LENGTH is the overflow signal a caller branches on"
            "\nwithout matching any provider's wording - the difference between"
            "\ndiagnosing a context overflow and shipping a half-written answer."
        )


if __name__ == "__main__":
    main()
