# demo_huggingface_v4.py
import asyncio

"""
This script demonstrates how to use the HuggingFace adapter with transformers v4.

It walks through:
  1. Loading a model with all available constructor arguments
  2. Invoking the model (sync and async)
  3. Streaming output (sync)
  4. Using the convenience chat() method
  5. Inspecting model capabilities
  6. An interactive chat loop so you can talk to the model

This demo works on both v4 and v5, but focuses on the v4-compatible code paths
and arguments. For v5-specific features, see demo_huggingface_v5.py.
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
    The main function to demonstrate the HuggingFace adapter (v4 style).
    """

    # --- Step 2: Show version info and available models ---
    print("=== HuggingFace Adapter Demo (v4 Style) ===\n")

    try:
        import transformers
        print(f"  transformers version: {transformers.__version__}")
    except Exception:
        print("  transformers: not available")
    print(f"  TRANSFORMERS_V5 flag: {TRANSFORMERS_V5}\n")

    print("Available model aliases:")
    for alias, full_name in sorted(MODEL_REGISTRY.items()):
        print(f"  {alias:28s} -> {full_name}")
    print(f"\n  Total: {len(MODEL_REGISTRY)} aliases (or pass any full HuggingFace model path)\n")

    # --- Step 3: Load the model with all constructor arguments ---
    # The constructor accepts these parameters:
    #   model_name   - alias from MODEL_REGISTRY or a full HuggingFace model path
    #   quantized    - load in 4-bit quantization via bitsandbytes (requires GPU)
    #   stream       - enable streaming support (TextIteratorStreamer)
    #   auth_token   - HuggingFace Hub token (or set HF_TOKEN env var)
    #   verbose      - print loading info
    #   torch_dtype  - torch dtype for model weights (default: torch.float16)
    #   device_map   - device placement strategy (default: "auto")
    #   max_new_tokens - default max tokens to generate (default: 256)
    #   temperature  - default sampling temperature (default: 0.7)
    #   top_p        - default nucleus sampling cutoff (default: 0.9)

    print("Loading model: tinyllama (all default constructor args shown)...")
    llm = HuggingFaceAdapter(
        model_name="tinyllama",       # alias -> TinyLlama/TinyLlama-1.1B-Chat-v1.0
        quantized=False,              # set True for 4-bit quantization (needs GPU + bitsandbytes)
        stream=True,                  # enable streaming support
        auth_token=None,              # or set HF_TOKEN env var
        verbose=True,                 # print loading info
        # torch_dtype=torch.float16,  # default; pass torch.bfloat16 for newer GPUs
        # device_map="auto",          # default; maps layers across available devices
        max_new_tokens=256,           # max tokens per generation
        temperature=0.7,              # sampling temperature
        top_p=0.9,                    # nucleus sampling
    )
    print("Model loaded successfully.\n")

    # --- Step 4: Inspect model capabilities ---
    print("=== get_model_capabilities() ===")
    caps = llm.get_model_capabilities()
    for key, val in caps.items():
        print(f"  {key}: {val}")
    print()

    # --- Step 5: invoke() — synchronous single-turn generation ---
    print("=== invoke() — Synchronous Generation ===")
    messages = [
        Message(role="system", content="You are a helpful assistant. Be concise."),
        Message(role="user", content="What is the capital of France?"),
    ]
    response = llm.invoke(messages)
    print(f"  Assistant: {response.content}\n")

    # Override generation kwargs per-call:
    print("=== invoke() — Custom generation kwargs ===")
    response_creative = llm.invoke(
        [Message(role="user", content="Tell me a one-sentence joke.")],
        temperature=1.2,
        max_new_tokens=64,
        top_p=0.95,
        do_sample=True,
    )
    print(f"  (temp=1.2, top_p=0.95): {response_creative.content}\n")

    # --- Step 6: ainvoke() — async generation ---
    print("=== ainvoke() — Async Generation ===")
    async_response = await llm.ainvoke(
        [Message(role="user", content="Name three colors.")],
        max_new_tokens=64,
    )
    print(f"  Assistant: {async_response.content}\n")

    # --- Step 7: stream() — synchronous streaming ---
    print("=== stream() — Synchronous Streaming ===")
    print("  Assistant: ", end="", flush=True)
    for chunk in llm.stream([Message(role="user", content="Count from 1 to 5.")]):
        print(chunk.content, end="", flush=True)
    print("\n")

    # --- Step 8: chat() — convenience method (returns string) ---
    print("=== chat() — Convenience Method ===")
    chat_response = llm.chat(
        [Message(role="user", content="What is 2 + 2?")],
        temperature=0.3,
    )
    print(f"  Assistant: {chat_response}\n")

    # --- Step 9: Interactive chat loop ---
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
