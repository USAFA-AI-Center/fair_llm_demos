# demo_vision_message.py
"""
Ask a local vision model what it sees in an image, through fairlib.

A Message carries images first-class: a tuple of raw bytes riding beside
the content, which the Ollama adapter lifts into the provider payload as
base64. The caller never touches a provider payload shape - it sets
images on the message, asks a question, and reads the reply. The vision
flag in get_model_capabilities is how code branches on what a model can
do, never on which adapter class it holds.

The demo hands a small embedded JPEG (a red square with a blue circle on
white) to a real local VL model and asks it two questions about the
picture, so the model's own answers carry the result. It then routes the
same image to an adapter built without the vision flag and shows the
typed ConfigurationError - images are never silently dropped, so a
pipeline misrouted onto a text-only model fails loudly instead of
captioning thin air. Finally it runs the conformance suite on the live
wire, with vision declared and undeclared, so the carry-or-refuse
behavior reads as a tested contract every adapter meets, not a one-off.

Set FAIR_LLM_DEMO_VLM to pick the local vision model served by Ollama.
"""

import base64
import os

from fairlib import (
    ChatModelConformanceCase,
    ConfigurationError,
    Message,
    OllamaAdapter,
    check_chat_model_conformance,
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_VLM", "qwen3-vl-instruct-16k")

# A 96x96 JPEG: a red square with a blue circle, on white. Embedded so the
# demo is self-contained; any JPEG or PNG bytes work the same way.
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


def build_recording_adapter(vision: bool) -> ChatModelConformanceCase:
    """One live adapter, hooked to record the payload each call builds."""
    adapter = OllamaAdapter(model_name=MODEL_NAME, vision=vision)
    recorded = {}
    original_body = adapter._payload_body

    def recording_body(messages, stream, options):
        payload = original_body(messages, stream, options)
        recorded["json"] = payload
        return payload

    adapter._payload_body = recording_body

    return ChatModelConformanceCase(
        model=adapter,
        recorded_request=lambda: recorded.get("json"),
        vision_sample=Message(
            role="user",
            content="What shapes and colors are in this image?",
            images=(SAMPLE_JPEG,),
        ),
        deterministic=False,
        label=f"{MODEL_NAME} (vision={vision})",
    )


def ask(adapter: OllamaAdapter, question: str) -> str:
    """Send the sample image with a question and return the reply text."""
    reply = adapter.invoke(
        [Message(role="user", content=question, images=(SAMPLE_JPEG,))]
    )
    return reply.content


def main() -> None:
    print(f"Model: {MODEL_NAME} via Ollama\n")

    vision = OllamaAdapter(model_name=MODEL_NAME, vision=True)
    print("We hand the model a small image as raw bytes on the Message and")
    print("just ask about it. The calling code never builds a provider")
    print("payload, and get_model_capabilities() reports vision =", end=" ")
    print(f"{vision.get_model_capabilities().vision}.\n")

    first = "Describe this image in one short sentence."
    print(f"  you   > {first}")
    print(f"  model > {ask(vision, first)}\n")

    second = (
        "Answer in one short phrase: what single shape is in the center, "
        "and what color is it?"
    )
    print(f"  you   > {second}")
    print(f"  model > {ask(vision, second)}\n")

    print("Those answers came from the pixels we sent, not from the prompt text.\n")

    print("Route the same image to a text-only model and fairlib stops you.")
    print("The vision flag, not the adapter class, is what code branches on:\n")
    text_only = OllamaAdapter(model_name=MODEL_NAME, vision=False)
    try:
        ask(text_only, "Describe this image.")
        print("  no refusal (this line should never print)")
    except ConfigurationError as exc:
        print(f"  typed refusal: {exc}")
    print("  (a misrouted pipeline fails loudly instead of captioning thin air)\n")

    print("And carry-or-refuse is a tested contract every adapter meets,")
    print("checked here on the live wire by the conformance suite:\n")
    for declared in (True, False):
        report = check_chat_model_conformance(build_recording_adapter(declared))
        print(report.render())
        print()


if __name__ == "__main__":
    main()
