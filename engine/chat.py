#!/usr/bin/env python3
"""Gemma 4 31B streaming chat with multi-turn memory.

Runs entirely on-device via MLX on Apple Silicon.
Usage: python chat.py
"""
import sys
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import QuantizedKVCache

MODEL_DIR = (
    "/Users/andromeda/.cache/huggingface/hub/"
    "models--mlx-community--gemma-4-31b-it-4bit/"
    "snapshots/535c5606372deb5d5ab7e29280f111ef2a8e084e"
)

SYSTEM_PROMPT = (
    "You are Gemma, a helpful AI assistant running locally on a MacBook Pro "
    "M1 Max with 64GB RAM. You are Google's Gemma 4 31B model running via "
    "TurboQuant on Apple Silicon — fully on-device, no cloud. "
    "Be direct, concise, and helpful. Remember everything in the conversation."
)

print("Loading Gemma 4 31B...")
model, tok = load(MODEL_DIR)
n_layers = len(model.make_cache())
model.make_cache = lambda: [QuantizedKVCache(group_size=64, bits=4) for _ in range(n_layers)]
print(f"Ready! {n_layers} layers, int4 KV cache\n")

history = []


def chat(user_text, max_tokens=500):
    history.append({"role": "user", "content": user_text})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    sys.stdout.write("\n")
    sys.stdout.flush()

    in_answer = False
    answer_parts = []
    think_buf = []
    response = None

    for response in stream_generate(model, tok, prompt=prompt, max_tokens=max_tokens):
        if not response.text:
            continue

        # Model outputs: <|channel>thought\n...thinking...\n<channel|>\nACTUAL ANSWER
        # Show dots while thinking, then stream the answer.
        if not in_answer:
            think_buf.append(response.text)
            joined = "".join(think_buf)
            # Show thinking indicator
            sys.stdout.write(".")
            sys.stdout.flush()
            if "<channel|>" in joined:
                in_answer = True
                # Clear the dots and start answer
                sys.stdout.write("\r\033[K")  # clear line
                sys.stdout.write("Gemma: ")
                after = joined.split("<channel|>", 1)[1].lstrip("\n")
                after = after.replace("<turn|>", "")
                if after:
                    sys.stdout.write(after)
                    sys.stdout.flush()
                answer_parts = [after] if after else []
        else:
            clean = response.text.replace("<turn|>", "")
            sys.stdout.write(clean)
            sys.stdout.flush()
            answer_parts.append(clean)

        if hasattr(response, "finish_reason") and response.finish_reason:
            break

    # If model never emitted <channel|>, show what we got
    if not in_answer and think_buf:
        raw = "".join(think_buf)
        for m in ["<|channel>", "<channel|>", "thought", "<turn|>"]:
            raw = raw.replace(m, "")
        raw = raw.strip()
        if raw:
            sys.stdout.write(raw)
            sys.stdout.flush()
            answer_parts = [raw]

    print()
    if response and hasattr(response, "generation_tps"):
        print(f"  [{response.generation_tokens} tokens, {response.generation_tps:.0f} tok/s]")

    answer = "".join(answer_parts).strip()
    history.append({"role": "assistant", "content": answer})
    return answer


if __name__ == "__main__":
    print("Gemma 4 31B — Interactive Chat")
    print("Type 'quit' to exit, 'clear' to reset\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not text:
            continue
        if text.lower() == "quit":
            break
        if text.lower() == "clear":
            history.clear()
            print("[Conversation cleared]\n")
            continue
        chat(text)
        print()
