"""Demo: per-provider circuit breaker fail-fast and recovery (async path)."""

import asyncio

from fairlib.core.errors import DegradedResponse
from fairlib.modules.mal.circuit_breaker import CircuitBreakerRegistry


async def main() -> None:
    registry = CircuitBreakerRegistry(enabled=True, failure_threshold=2, cooldown_seconds=0.2)
    async def fail():
        raise DegradedResponse.for_kind(
            DegradedResponse.Kind.SERVER_ERROR, "provider down", provider="demo"
        )

    for attempt in range(1, 4):
        try:
            await registry.acall("demo", fail)
        except DegradedResponse as exc:
            print(f"attempt {attempt}: {exc.kind.value}")

    print("waiting for cooldown...")
    await asyncio.sleep(0.25)

    async def ok():
        return "recovered"

    result = await registry.acall("demo", ok)
    print(f"half-open probe succeeded: {result!r}, state={registry.state_of('demo').value}")
    print("Circuit breaker demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
