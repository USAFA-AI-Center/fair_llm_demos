# demo_huggingface.py
"""
This script demonstrates the HuggingFace adapter.

On transformers v5 (the pinned version), the adapter automatically:
  1. Detects the version and sets the TRANSFORMERS_V5 flag
  2. Passes attn_implementation="sdpa" for faster inference
  3. Uses AsyncTextIteratorStreamer for true async streaming
  4. Handles BatchEncoding returns from apply_chat_template

This demo walks through all adapter methods, then drops you into an
interactive chat loop. It works on transformers v4 too - you will just
see the v4 fallback behavior instead.
"""

# --- Step 1: Import the necessary components ---
import asyncio
import os
import sys
from dataclasses import fields

from fairlib import (
    DegradedResponse,
    FairlibError,
    HuggingFaceAdapter,
    Message,
    RoleDefinition,
    SafeCalculatorTool,
    SimpleAgent,
    SimpleReActPlanner,
    ToolExecutor,
    ToolRegistry,
    WorkingMemory,
)
from fairlib.modules.mal.huggingface_adapter import TRANSFORMERS_V5

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "qwen25-7b")


async def main():
    """
    The main function to demonstrate the HuggingFace adapter.
    """

    # --- Step 2: Show version detection ---
    print("=== HuggingFace Adapter Demo ===\n")

    try:
        import transformers

        print(f"  transformers version: {transformers.__version__}")
    except Exception:
        print("  transformers: not available")
    print(f"  TRANSFORMERS_V5 flag: {TRANSFORMERS_V5}")

    if TRANSFORMERS_V5:
        print("  -> v5 codepath ACTIVE")
        print("     - SDPA attention auto-enabled")
        print("     - AsyncTextIteratorStreamer for async streaming")
        print("     - BatchEncoding handling in _format_prompt()")
    else:
        print(
            "  -> v4 codepath ACTIVE (upgrade to transformers>=5.0.0 for v5 features)"
        )
    print()

    # --- Step 3: Load the model with v5-relevant constructor arguments ---
    # On v5, the adapter automatically passes attn_implementation="sdpa" to
    # AutoModelForCausalLM.from_pretrained(). You can override this:
    #   attn_implementation="flash_attention_2"  (requires flash-attn package)
    #   attn_implementation="eager"              (disable optimized attention)

    print(f"Loading model: {MODEL_NAME}")
    print("  (v5 will auto-set attn_implementation='sdpa' for faster inference)")
    llm = HuggingFaceAdapter(
        model_name=MODEL_NAME,
        quantized=False,
        stream=True,
        auth_token=None,
        verbose=True,
        max_new_tokens=256,
        temperature=0.3,
        top_p=0.9,
        # v5-specific: override attention implementation if desired
        # attn_implementation="flash_attention_2",
    )
    print("Model loaded successfully.\n")

    # --- Step 4: _prepare_messages() - clean dict output ---
    # This helper strips metadata and None fields so v5 chat templates
    # do not choke on unexpected keys.
    print("=== _prepare_messages() - Clean Message Dicts ===")
    raw_messages = [
        Message(role="system", content="Be concise.", metadata={"source": "demo"}),
        Message(role="user", content="Hello!", name=None, tool_calls=None),
    ]
    prepared = llm._prepare_messages(raw_messages)
    for i, d in enumerate(prepared):
        print(f"  Message {i}: {d}")
    print("  -> No 'metadata', 'tool_calls=None', or 'name=None' in output\n")

    # --- Step 5: get_model_capabilities() ---
    print("=== get_model_capabilities() ===")
    caps = llm.get_model_capabilities()
    for field in fields(caps):
        print(f"  {field.name}: {getattr(caps, field.name)}")
    print()

    # --- Step 6: invoke() - synchronous generation ---
    print("=== invoke() ===")
    response = llm.invoke(
        [
            Message(role="system", content="You are a helpful assistant. Be concise."),
            Message(role="user", content="What are three planets in our solar system?"),
        ],
        max_new_tokens=128,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
    )
    print(f"  Assistant: {response.content}\n")

    # --- Step 7: ainvoke() - async generation ---
    print("=== ainvoke() ===")
    async_response = await llm.ainvoke(
        [Message(role="user", content="Name three programming languages.")],
        max_new_tokens=64,
    )
    print(f"  Assistant: {async_response.content}\n")

    # --- Step 8: stream() - synchronous streaming ---
    # Uses TextIteratorStreamer + Thread (works on both v4 and v5).
    # Streams raise DegradedResponse on provider failure instead of
    # yielding an error-text chunk, so wrap iteration when rendering.
    print("=== stream() - Synchronous Streaming ===")
    print("  Assistant: ", end="", flush=True)
    try:
        for chunk in llm.stream(
            [Message(role="user", content="Count from 1 to 5.")],
            max_new_tokens=64,
        ):
            print(chunk.content, end="", flush=True)
    except DegradedResponse as exc:
        print(f"\n  [stream degraded: {exc.kind.value}]")
    print("\n")

    # --- Step 9: astream() - async streaming ---
    # On v5: uses AsyncTextIteratorStreamer for true non-blocking iteration.
    # On v4: runs the checked generation off the loop and yields one Message.
    print("=== astream() - Async Streaming ===")
    if TRANSFORMERS_V5:
        print("  (v5: using AsyncTextIteratorStreamer)")
    else:
        print("  (v4: one complete Message from a generation run off the loop)")
    print("  Assistant: ", end="", flush=True)
    try:
        async for chunk in llm.astream(
            [Message(role="user", content="Write a short greeting.")],
            max_new_tokens=64,
        ):
            print(chunk.content, end="", flush=True)
    except DegradedResponse as exc:
        print(f"\n  [stream degraded: {exc.kind.value}]")
    print("\n")

    # --- Step 10: chat() - convenience method ---
    print("=== chat() - Convenience Method ===")
    chat_response = llm.chat(
        [Message(role="user", content="What is 2 + 2?")],
        temperature=0.3,
    )
    print(f"  Assistant: {chat_response}\n")

    # --- Step 11: The same adapter as the brain of an agent ---
    # Everything above talked to the model directly. An agent wraps the
    # adapter with a planner, tools, and memory, so the model can reason
    # about when to call a tool instead of guessing at arithmetic.
    print("=" * 60)
    print("Interactive agent - type 'exit' or 'quit' to stop.")
    print("=" * 60)

    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())
    planner = SimpleReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are a friendly assistant. Use the calculator for any arithmetic "
        "and keep answers short."
    )
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=ToolExecutor(tool_registry),
        memory=WorkingMemory(),
        max_steps=6,
    )

    print("\nYou: What is 123 * 45?")
    try:
        print(f"Agent: {await agent.arun('What is 123 * 45?')}")
    except FairlibError as exc:
        # A small sampled model can wander; the framework reports that as a
        # typed error rather than a made-up answer.
        print(f"Agent run ended with {type(exc).__name__}: {exc}")

    if not sys.stdin.isatty():
        return
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Agent: Goodbye!")
                break
            print(f"Agent: {await agent.arun(user_input)}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    asyncio.run(main())
