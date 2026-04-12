#!/usr/bin/env python3
"""Interactive multi-turn REPL for Gemma 4 31B.

Keeps conversation context via KV cache files between turns.
Type 'quit' to exit, 'clear' to reset conversation.
"""
import os, struct, subprocess

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/"
    "snapshots/535c5606372deb5d5ab7e29280f111ef2a8e084e/"
)
BINARY = os.path.join(ROOT, "build", "gemma4")
PROMPT_FILE = os.path.join(ROOT, "prompt.bin")
CACHE_FILE = os.path.join(ROOT, "kv_cache.safetensors")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)


def tokenize_turn(text, is_first=True):
    if is_first:
        prompt = f"<start_of_turn>user\n{text}\n<end_of_turn>\n<start_of_turn>model\n"
        return tok.encode(prompt, add_special_tokens=True)
    else:
        prompt = f"<end_of_turn>\n<start_of_turn>user\n{text}\n<end_of_turn>\n<start_of_turn>model\n"
        return tok.encode(prompt, add_special_tokens=False)


def write_tokens(tokens):
    with open(PROMPT_FILE, 'wb') as f:
        f.write(struct.pack('<I', len(tokens)))
        for t in tokens:
            f.write(struct.pack('<i', t))


def run_inference():
    result = subprocess.run(
        [BINARY], capture_output=True, text=True, timeout=300, cwd=ROOT
    )
    lines = result.stdout.split('\n')
    response = []
    in_gen = False
    for line in lines:
        if 'Generating:' in line:
            in_gen = True
            continue
        if in_gen:
            if 'tokens in' in line or 'KV cache' in line:
                break
            response.append(line)

    text = '\n'.join(response).strip()
    # Clean thought markers
    for marker in ['<|channel>', '<channel|>', 'thought', '-//-', '<turn|>']:
        text = text.replace(marker, '')
    text = text.strip().lstrip('-\n ')

    # Extract timing
    tok_s = ""
    for line in lines:
        if 'tok/s' in line:
            tok_s = line.strip()
            break

    return text, tok_s


if __name__ == "__main__":
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

    print("Gemma 4 31B — Interactive Chat")
    print("Type 'quit' to exit, 'clear' to reset\n")

    turn = 0
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'clear':
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
            turn = 0
            print("[Conversation cleared]\n")
            continue

        tokens = tokenize_turn(user_input, is_first=(turn == 0))
        write_tokens(tokens)

        response, tok_s = run_inference()
        print(f"\nGemma: {response}")
        if tok_s:
            print(f"  [{tok_s}]")
        print()
        turn += 1
