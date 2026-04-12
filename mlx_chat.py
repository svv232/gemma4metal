#!/usr/bin/env python3
"""Fast Gemma 4 31B chat via mlx-lm with int4 KV cache.

15 tok/s + 6.4x KV compression — best of both worlds.
Requires Python 3.12: /opt/homebrew/bin/python3.12

Usage:
    python3.12 mlx_chat.py "What is the capital of France?"
    python3.12 mlx_chat.py   # interactive mode
"""
import sys
from mlx_lm import load, generate
from mlx_lm.models.cache import QuantizedKVCache

MODEL_DIR = (
    "/Users/andromeda/.cache/huggingface/hub/"
    "models--mlx-community--gemma-4-31b-it-4bit/"
    "snapshots/535c5606372deb5d5ab7e29280f111ef2a8e084e"
)

print("Loading Gemma 4 31B (int4 KV cache)...")
model, tok = load(MODEL_DIR)

# Monkey-patch: use int4 quantized KV cache for 6.4x compression
n_layers = len(model.make_cache())
model.make_cache = lambda: [QuantizedKVCache(group_size=64, bits=4) for _ in range(n_layers)]
print(f"Ready! ({n_layers} layers, int4 KV cache)\n")


def chat(user_text, max_tokens=200):
    messages = [{"role": "user", "content": user_text}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    result = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=True)

    # Clean up thought markers
    clean = result
    for marker in ["<|channel>", "<channel|>", "thought", "<turn|>"]:
        clean = clean.replace(marker, "")
    clean = clean.strip().lstrip("-/ \n")
    return clean


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(f"You: {text}\n")
        answer = chat(text)
        print(f"\nAnswer: {answer}")
    else:
        print("Gemma 4 31B — Interactive Chat (15 tok/s, int4 KV)")
        print("Type 'quit' to exit\n")
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not text or text.lower() == "quit":
                break
            answer = chat(text)
            print(f"\nGemma: {answer}\n")
