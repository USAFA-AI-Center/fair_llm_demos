"""Demo: a hard wall-clock timeout bounds a hung provider call.

Drives the real OllamaAdapter with a fake transport that never answers,
and shows the caller receives a typed DegradedResponse(TIMEOUT) in
roughly the configured deadline - not after the hang duration.
"""

import asyncio
import time

from fairlib.core.errors import DegradedResponse
from fairlib.core.message import Message
from fairlib.modules.mal.local_llama_adapter import OllamaAdapter

HANG_SECONDS = 30.0
TIMEOUT_SECONDS = 0.5


class HungTransport:
    """Stands in for httpx.AsyncClient; the server never answers."""

    async def post(self, *args, **kwargs):
        await asyncio.sleep(HANG_SECONDS)


def main() -> None:
    adapter = OllamaAdapter(model_name="demo-model", timeout=TIMEOUT_SECONDS)
    adapter.client = HungTransport()

    caught = None
    start = time.monotonic()
    try:
        asyncio.run(adapter.ainvoke([Message(role="user", content="hello?")]))
    except DegradedResponse as exc:
        caught = exc
    elapsed = time.monotonic() - start

    if caught is None:
        print("No DegradedResponse was raised - the call completed unexpectedly.")
    else:
        print(f"Typed signal: kind={caught.kind.value}, retryable={caught.retryable}")
    print(f"Elapsed: {elapsed:.2f}s (timeout={TIMEOUT_SECONDS}s, hang={HANG_SECONDS}s)")
    print("Hard timeout demo complete.")


if __name__ == "__main__":
    main()
