# demo_huggingface_v5.py
import asyncio

"""
This script demonstrates the v5-specific features of the HuggingFace adapter.

When transformers v5 is installed, the adapter automatically:
  1. Detects the version and sets the TRANSFORMERS_V5 flag
  2. Passes attn_implementation="sdpa" for faster inference
  3. Uses AsyncTextIteratorStreamer for true async streaming
  4. Handles BatchEncoding returns from apply_chat_template

This demo walks through all adapter methods with v5-aware commentary, then
drops you into an interactive chat loop. It works on v4 too — you will just
see the v4 fallback behavior instead.

For the v4-focused walkthrough, see demo_huggingface_v4.py.
"""

# --- Step 1: Import the necessary components ---
from fairlib.modules.mal.huggingface_adapter import (
    HuggingFaceAdapter,
    MODEL_REGISTRY,
    TRANSFORMERS_V5,
)
from fairlib.core.message import Message


async def main():
    """
    The main function to demonstrate v5-specific HuggingFace adapter features.
    """

    # --- Step 2: Show version detection ---
    print("=== HuggingFace Adapter Demo (v5 Features) ===\n")

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
        print("  -> v4 codepath ACTIVE (upgrade to transformers>=5.0.0 for v5 features)")
    print()

    # --- Step 3: Load the model with v5-relevant constructor arguments ---
    # On v5, the adapter automatically passes attn_implementation="sdpa" to
    # AutoModelForCausalLM.from_pretrained(). You can override this:
    #   attn_implementation="flash_attention_2"  (requires flash-attn package)
    #   attn_implementation="eager"              (disable optimized attention)

    print("Loading model: tinyllama")
    print("  (v5 will auto-set attn_implementation='sdpa' for faster inference)")
    llm = HuggingFaceAdapter(
        model_name="tinyllama",
        quantized=False,
        stream=True,
        auth_token=None,
        verbose=True,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        # v5-specific: override attention implementation if desired
        # attn_implementation="flash_attention_2",
    )
    print("Model loaded successfully.\n")

    # --- Step 4: _prepare_messages() — clean dict output ---
    # This helper strips metadata and None fields so v5 chat templates
    # do not choke on unexpected keys.
    print("=== _prepare_messages() — Clean Message Dicts ===")
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
    for key, val in caps.items():
        print(f"  {key}: {val}")
    print()

    # --- Step 6: invoke() — synchronous generation ---
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

    # --- Step 7: ainvoke() — async generation ---
    print("=== ainvoke() ===")
    async_response = await llm.ainvoke(
        [Message(role="user", content="Name three programming languages.")],
        max_new_tokens=64,
    )
    print(f"  Assistant: {async_response.content}\n")

    # --- Step 8: stream() — synchronous streaming ---
    # Uses TextIteratorStreamer + Thread (works on both v4 and v5).
    print("=== stream() — Synchronous Streaming ===")
    print("  Assistant: ", end="", flush=True)
    for chunk in llm.stream(
        [Message(role="user", content="Count from 1 to 5.")],
        max_new_tokens=64,
    ):
        print(chunk.content, end="", flush=True)
    print("\n")

    # --- Step 9: astream() — async streaming ---
    # On v5: uses AsyncTextIteratorStreamer for true non-blocking iteration.
    # On v4: falls back to ainvoke() and yields a single Message.
    print("=== astream() — Async Streaming ===")
    if TRANSFORMERS_V5:
        print("  (v5: using AsyncTextIteratorStreamer)")
    else:
        print("  (v4: falling back to ainvoke)")
    print("  Assistant: ", end="", flush=True)
    async for chunk in llm.astream(
        [Message(role="user", content="Write a short greeting.")],
        max_new_tokens=64,
    ):
        print(chunk.content, end="", flush=True)
    print("\n")

    # --- Step 10: chat() — convenience method ---
    print("=== chat() — Convenience Method ===")
    chat_response = llm.chat(
        [Message(role="user", content="What is 2 + 2?")],
        temperature=0.3,
    )
    print(f"  Assistant: {chat_response}\n")

    # --- Step 11: Interactive chat loop ---
    print("=" * 60)
    print("Interactive chat — type 'exit' or 'quit' to stop.")
    print("=" * 60)

    history = [
        Message(role="system", content="You are a friendly assistant. Keep answers short."),
    ]

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Assistant: Goodbye!")
                break

            history.append(Message(role="user", content=user_input))
            agent_response = llm.invoke(history)
            print(f"Assistant: {agent_response.content}")
            history.append(agent_response)

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    asyncio.run(main())
