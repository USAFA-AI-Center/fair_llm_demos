# demo_gemini_adapter.py
"""
Talk to Google Gemini through fairlib's Model Abstraction Layer.

The GeminiAdapter is one more implementation of AbstractChatModel: the
same Message list in, the same Message out, the same Usage record and
ModelInvocationEvent every other adapter produces. Nothing in this demo
names a Gemini payload shape; it sets messages, asks, and reads what the
framework hands back.

The demo asks the model one question and prints the reply with the
provider's token counts and stop reason, then repeats the question under
a tiny generation budget so the model stops on LENGTH and the typed stop
reason (not the prose) says so. It streams a third answer chunk by
chunk, hands the model a small generated JPEG and asks what it sees (the
adapter types the image from its own bytes), and finally runs the shared
conformance suite on the live wire so the adapter's contract reads as a
tested one rather than a claim.

Set GEMINI_API_KEY (or GOOGLE_API_KEY) to your key. FAIR_LLM_DEMO_GEMINI
picks the model; it defaults to gemini-3.6-flash. The conformance section
sends twenty-one requests to the API in about a minute (the adapter refuses
thirty-six more before any request: the role, tool-turn, option and gating
refusals the suite checks on every entry), which a free-tier key (five
requests per minute, twenty per day, per model) cannot carry; run the demo
with a key on a paid tier.
"""

import io
import os

from PIL import Image, ImageDraw

from fairlib import (
    ChatModelConformanceCase,
    GeminiAdapter,
    Message,
    check_chat_model_conformance,
)
from fairlib.core.event_bus import AgentEventBus
from fairlib.core.events import ModelInvocationEvent

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_GEMINI", "gemini-3.6-flash")


def sample_jpeg() -> bytes:
    """A 96x96 JPEG: a red square with a blue circle, on white."""
    image = Image.new("RGB", (96, 96), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 84, 84), fill="red")
    draw.ellipse((30, 30, 66, 66), fill="blue")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def show_usage(label: str, reply: Message) -> None:
    usage = reply.usage
    if usage is None:
        print(f"[{label}] no usage record on the reply")
        return
    print(
        f"[{label}] prompt {usage.prompt_tokens} / completion "
        f"{usage.completion_tokens} tokens; stop reason "
        f"{usage.done_reason.value if usage.done_reason else None} "
        f"(provider said {usage.raw_done_reason!r}); model {usage.model}"
    )


def main() -> None:
    # Construction-time generation defaults ride with every call; a call's
    # own options override them key by key (the picture question below
    # sets temperature to 0.0 over this default).
    llm = GeminiAdapter(model_name=MODEL_NAME, timeout=60, options={"temperature": 0.2})
    bus = AgentEventBus()
    events = []
    bus.subscribe(ModelInvocationEvent, events.append)
    llm.bind_event_bus(bus)

    question = [
        Message(role="system", content="Answer in one short sentence."),
        Message(role="user", content="Why does the sky look blue in daytime?"),
    ]

    print("=== 1. One question, one reply ===")
    reply = llm.invoke(question)
    print(reply.content)
    show_usage("invoke", reply)

    print("\n=== 2. The same question under a budget of 8 output tokens ===")
    short = llm.invoke(question, max_tokens=8)
    print(
        f"content: {short.content!r} (a thinking model may spend the whole budget on thoughts)"
    )
    show_usage("budgeted", short)

    print("\n=== 3. Streaming ===")
    pieces = []
    for chunk in llm.stream(
        [Message(role="user", content="Count from one to five in words.")]
    ):
        pieces.append(chunk.content)
        print(chunk.content, end="", flush=True)
    print()
    print(f"[stream] {len(pieces)} chunks")

    print("\n=== 4. A picture, typed from its own bytes ===")
    picture = Message(
        role="user",
        content="Describe the shapes and colors in this image in one sentence.",
        images=(sample_jpeg(),),
    )
    seen = llm.invoke([picture], temperature=0.0)
    print(seen.content)
    show_usage("vision", seen)

    print("\n=== 5. What the bus saw ===")
    for event in events:
        print(
            f"{event.provider}/{event.model_name}: {event.outcome.value} in "
            f"{event.duration_ms:.0f} ms, {event.message_count} message(s), "
            f"usage {event.usage.prompt_tokens if event.usage else None}/"
            f"{event.usage.completion_tokens if event.usage else None}"
        )

    print("\n=== 6. The conformance suite, on the live wire ===")
    report = check_chat_model_conformance(
        ChatModelConformanceCase(
            model=GeminiAdapter(model_name=MODEL_NAME, timeout=60),
            vision_sample=picture,
            deterministic=False,
            timeout_seconds=90,
        )
    )
    print(report.render())


if __name__ == "__main__":
    main()
