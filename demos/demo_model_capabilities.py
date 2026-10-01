# demo_model_capabilities.py
"""
Branch on what a model can do without knowing which provider it is.

Every chat model answers get_model_capabilities() with a ModelCapabilities
record: streaming, tool_calling, vision, max_context_window, the
provider-neutral generation options it carries, tool_results, whether it
carries a native tool-result turn, and max_stop_sequences, the most stop
sequences it takes in one request. The same seven fields come back from
every adapter, so the code below builds a list of models and asks each one
the same questions - it never names an adapter class.

A record may settle while the model is in use: the Ollama model reads
tool_results from its server's /api/show reply on the first request that
carries a tool turn, and a plain chat never reads it. So the demo prints
tool_results before any call, after the plain calls, and after one request
carrying a paired tool turn (sent or refused typed). Read the record for
each decision; never cache it.

Two local models take part: a vision model served by Ollama, and a text
model loaded through Hugging Face transformers. For each, the demo prints
the record, sends an image if vision is declared and shows the typed
refusal if it is not, and passes a seed only when the model carries it,
showing the refusal a model gives for an option it does not carry. The
model's own replies carry the result.

Set FAIR_LLM_DEMO_VLM for the Ollama vision model and FAIR_LLM_DEMO_MODEL
for the Hugging Face text model.
"""

import base64
import os
from dataclasses import fields
from typing import List, Tuple

from fairlib import (
    AbstractChatModel,
    ConfigurationError,
    HuggingFaceAdapter,
    Message,
    OllamaAdapter,
)

VISION_MODEL = os.environ.get("FAIR_LLM_DEMO_VLM", "qwen3-vl-instruct-16k")
TEXT_MODEL = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen25-7b")

# A 96x96 JPEG: a red square with a blue circle, on white.
SAMPLE_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAoHBwgHBgoICAgLCgoLDhgQDg0NDh0VFhEYIx8l"
    "JCIfIiEmKzcvJik0KSEiMEExNDk7Pj4+JS5ESUM8SDc9Pjv/2wBDAQoLCw4NDhwQEBw7KCIo"
    "Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozv/wAAR"
    "CABgAGADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAA"
    "AgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkK"
    "FhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWG"
    "h4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl"
    "5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREA"
    "AgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYk"
    "NOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOE"
    "hYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk"
    "5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD2aiiigAooooAKKKKACiiigAooooAKKKKA"
    "CiiigD5zoooryD9ECiiigAooooAKKKKAPevDv/Is6X/15w/+gCtKs3w7/wAizpf/AF5w/wDo"
    "ArSr1Y7I/P6v8SXqwoooqjM+c6KKK8g/RAqtLfRocIN5/IUX0pSIIOr/AMqz6+nyjKKeIp+3"
    "r6roj53NM0qUKnsaO/Vl+O/Rjh1Ke/WrQIIyDkGsarthKTuiPbkVpm2TUqNJ1qGlt1v/AMEz"
    "yzNqlWqqVbW+zLlFFFfKH0x714d/5FnS/wDrzh/9AFaVZvh3/kWdL/684f8A0AVpV6sdkfn9"
    "X+JL1YUUUVRmfOdFFFeQfohVv4y0auP4Tz+NUK2SARgjINU5bDJzE2PZq+rybNqVGl7Cs7W2"
    "fr/wT5nNssq1antqSvfdFKrmnxnc0nbG2iPTzn944x/s1cVVRQqjAHQVrm+b0Z0XRou7e76J"
    "GeV5XVhVVasrJbIWiiivjz6o968O/wDIs6X/ANecP/oArSrN8O/8izpf/XnD/wCgCtKvVjsj"
    "8/q/xJerCiiiqMz5zoooryD9ECiiigAooooAKKKKAPevDv8AyLOl/wDXnD/6AK0qzfDv/Is6"
    "X/15w/8AoArSr1Y7I/P6v8SXqwoooqjMzf8AhHdD/wCgLp//AICp/hR/wjuh/wDQF0//AMBU"
    "/wAK0qKXKuxp7Wp/M/vM3/hHdD/6Aun/APgKn+FH/CO6H/0BdP8A/AVP8K0qKOVdg9rU/mf3"
    "mb/wjuh/9AXT/wDwFT/Cj/hHdD/6Aun/APgKn+FaVFHKuwe1qfzP7zN/4R3Q/wDoC6f/AOAq"
    "f4Uf8I7of/QF0/8A8BU/wrSoo5V2D2tT+Z/eMjjjhiSKJFjjRQqoowFA6ADsKfRRTMz/2Q=="
)


def build_models() -> List[Tuple[str, AbstractChatModel]]:
    """The only place a provider is named; everything below is blind to it."""
    return [
        (
            "vision model via Ollama",
            OllamaAdapter(model_name=VISION_MODEL, vision=True),
        ),
        (
            "text model via transformers",
            HuggingFaceAdapter(model_name=TEXT_MODEL, max_new_tokens=60),
        ),
    ]


def show_record(label: str, model: AbstractChatModel) -> None:
    caps = model.get_model_capabilities()
    print(f"--- {label}")
    for field in fields(caps):
        value = getattr(caps, field.name)
        if isinstance(value, frozenset):
            value = ", ".join(sorted(value))
        print(f"  {field.name}: {value}")


def ask_about_the_image(model: AbstractChatModel) -> None:
    question = "In one short sentence, what shapes and colors are in this image?"
    message = Message(role="user", content=question, images=(SAMPLE_JPEG,))
    if model.get_model_capabilities().vision:
        reply = model.invoke([message], max_tokens=60)
        print(f"  vision declared, so the image was sent: {reply.content.strip()}")
        return
    try:
        model.invoke([message], max_tokens=60)
        print("  no vision declared, yet the image went through (unexpected)")
    except ConfigurationError as exc:
        print(f"  no vision declared, so the image is refused: {exc}")


def ask_with_a_seed(model: AbstractChatModel) -> None:
    prompt = "Name one primary color. Answer with the color only."
    message = Message(role="user", content=prompt)
    if "seed" in model.get_model_capabilities().generation_options:
        reply = model.invoke([message], max_tokens=10, seed=7)
        print(f"  seed carried, reply: {reply.content.strip()}")
        return
    try:
        model.invoke([message], max_tokens=10, seed=7)
        print("  seed not declared, yet it was accepted (unexpected)")
    except ConfigurationError as exc:
        print(f"  seed not declared, so the call is refused: {exc}")
    reply = model.invoke([message], max_tokens=10)
    print(f"  without the seed, reply: {reply.content.strip()}")


def send_a_tool_turn(model: AbstractChatModel) -> None:
    call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "add", "arguments": {"a": 2, "b": 3}},
    }
    history = [
        Message(role="user", content="What is 2 + 3? Use the add tool."),
        Message(role="assistant", content="", tool_calls=[call]),
        Message(role="tool", content="5", tool_call_id="call-1"),
        Message(role="user", content="State the result in one short sentence."),
    ]
    try:
        reply = model.invoke(history, max_tokens=30)
        print(f"  the paired tool turn was carried: {reply.content.strip()}")
    except ConfigurationError as exc:
        print(f"  the paired tool turn is refused: {exc}")


def main() -> None:
    print("One record per model; the loop never asks which provider it holds.\n")
    for label, model in build_models():
        show_record(label, model)
        before = model.get_model_capabilities().tool_results
        ask_about_the_image(model)
        ask_with_a_seed(model)
        plain = model.get_model_capabilities().tool_results
        send_a_tool_turn(model)
        after = model.get_model_capabilities().tool_results
        print(
            f"  tool_results before any call: {before}; after the plain calls: "
            f"{plain}; after the tool-turn request: {after}"
        )
        print()


if __name__ == "__main__":
    main()
