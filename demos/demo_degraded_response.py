"""
Demo of the typed degraded-response signal (DegradedResponse).

When a provider call fails, adapters raise DegradedResponse instead of a
prose error string. The exception carries a classification (its Kind: rate
limit, timeout, auth, context overflow, ...) plus machine-readable recovery
policy fields, so your code branches on typed values instead of parsing
message text:

    if exc.should_compress:   shrink the context, then retry
    elif exc.retryable:       wait exc.retry_after seconds, then retry
    else:                     escalate - retrying cannot help

Adapters build the signal one of two ways, both shown below:
  - DegradedResponse.classify(raw_exception)  - infer the Kind from a raw
    provider error (status code, class name, message text)
  - DegradedResponse.for_kind(kind, message)  - when the adapter already
    knows what went wrong (e.g. it caught a GPU out-of-memory error)

This demo needs no API keys or network. FakeProvider stands in for a real
adapter and raises scripted failures, so you can watch each policy branch
run: a rate limit gets retried, a context overflow gets compressed, an auth
failure escalates, and a GPU OOM drives the same compress-and-retry path.
"""

import time
from typing import List

from fairlib import DegradedResponse, Message


# Simulated provider errors. These mimic what real SDKs raise: an exception
# with a status code, a recognizable class name, and a message.

class RateLimitError(Exception):
    status_code = 429
    retry_after = 0.1


class ContextLengthError(Exception):
    status_code = 400


class AuthenticationError(Exception):
    status_code = 401


class FakeProvider:
    """Stands in for a real adapter. Raises each scripted failure once
    (typed via DegradedResponse.classify, exactly like the real adapters),
    then answers normally."""

    def __init__(self, failures: List[Exception]) -> None:
        self.failures = list(failures)

    def chat(self, messages: List[Message]) -> Message:
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, DegradedResponse):
                raise failure  # adapter already built the typed signal
            raise DegradedResponse.classify(failure, provider="demo") from failure
        question = messages[-1].content
        return Message(role="assistant", content=f"[answer to: {question}]")


def compress(messages: List[Message]) -> List[Message]:
    """Stand-in for real context compression: keep only the latest turn."""
    return messages[-1:]


def call_with_recovery(provider: FakeProvider, messages: List[Message]) -> Message:
    """Call the provider, recovering from DegradedResponse using only its
    typed policy fields - never by parsing the error message."""
    for attempt in range(1, 4):
        try:
            answer = provider.chat(messages)
            print(f"  attempt {attempt}: success -> {answer.content}")
            return answer
        except DegradedResponse as exc:
            print(f"  attempt {attempt}: degraded (kind={exc.kind.value}, "
                  f"retryable={exc.retryable}, should_compress={exc.should_compress})")
            if exc.should_compress:
                messages = compress(messages)
                print("    -> compressing context and retrying")
            elif exc.retryable:
                wait = exc.retry_after or 0.1
                print(f"    -> waiting {wait}s and retrying")
                time.sleep(wait)
            else:
                print("    -> not recoverable, escalating")
                raise
    raise DegradedResponse.for_kind(
        DegradedResponse.Kind.UNKNOWN, "Recovery attempts exhausted.", provider="demo"
    )


def main() -> None:
    question = [Message(role="user", content="What is 2 + 2?")]

    print("Scenario 1: rate limit -> wait and retry")
    call_with_recovery(FakeProvider([RateLimitError("slow down")]), question)

    print("\nScenario 2: context overflow -> compress and retry")
    call_with_recovery(
        FakeProvider([ContextLengthError("maximum context length is 8192 tokens")]),
        question,
    )

    print("\nScenario 3: auth failure -> escalate (retrying cannot help)")
    try:
        call_with_recovery(FakeProvider([AuthenticationError("invalid api key")]), question)
    except DegradedResponse as exc:
        print(f"  escalated to caller: kind={exc.kind.value}")

    print("\nScenario 4: GPU out of memory -> same compress-and-retry policy")
    # Here the adapter knows the kind itself (no raw error to classify), so it
    # builds the signal directly - this is what HuggingFaceAdapter does on OOM.
    oom = DegradedResponse.for_kind(
        DegradedResponse.Kind.RESOURCE_EXHAUSTED,
        "GPU out of memory; shrink the context and retry.",
        provider="demo",
    )
    call_with_recovery(FakeProvider([oom]), question)

    print("\nEvery branch used the typed fields - no message parsing anywhere.")


if __name__ == "__main__":
    main()
