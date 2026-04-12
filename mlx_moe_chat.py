#!/usr/bin/env python3
"""Fast Gemma 4 26B-A4B MoE chat — 53-59 tok/s on M1 Max.

The fastest Gemma 4 config: MoE (4B active params) + int4 KV cache.
Requires Python 3.12: /opt/homebrew/bin/python3.12

Usage:
    python3.12 mlx_moe_chat.py "What is the capital of France?"
    python3.12 mlx_moe_chat.py   # interactive mode
"""
import sys
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import QuantizedKVCache

MODEL_DIR = (
    "/Users/andromeda/.cache/huggingface/hub/"
    "models--mlx-community--gemma-4-26b-a4b-it-4bit/"
    "snapshots/8bcfa0de037c2b1bfa323a1e8d1f0132243b9e87"
)

print("Loading Gemma 4 26B-A4B MoE (int4 KV)...")
model, tok = load(MODEL_DIR)
n_layers = len(model.make_cache())
model.make_cache = lambda: [QuantizedKVCache(group_size=64, bits=4) for _ in range(n_layers)]
print(f"Ready! {n_layers} layers, 53+ tok/s\n")


def chat(user_text, max_tokens=300):
    messages = [{"role": "user", "content": user_text}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Stream tokens
    in_answer = False
    answer_parts = []
    for response in stream_generate(model, tok, prompt=prompt, max_tokens=max_tokens):
        if response.text:
            # Detect transition from thought to answer
            if "<channel|>" in response.text:
                in_answer = True
                text = response.text.split("<channel|>")[-1]
                sys.stdout.write(text)
                answer_parts.append(text)
            elif in_answer:
                clean = response.text.replace("<turn|>", "").replace("<|channel>", "")
                sys.stdout.write(clean)
                answer_parts.append(clean)
            sys.stdout.flush()

        if hasattr(response, "finish_reason") and response.finish_reason:
            break

    print()  # newline after streaming
    if hasattr(response, "generation_tps"):
        print(f"  [{response.generation_tokens} tokens, {response.generation_tps:.0f} tok/s]")

    return "".join(answer_parts).strip()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(f"You: {text}\n")
        chat(text)
    else:
        print("Gemma 4 MoE — 53 tok/s Interactive Chat")
        print("Type 'quit' to exit\n")
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not text or text.lower() == "quit":
                break
            print()
            chat(text)
            print()
