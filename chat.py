#!/usr/bin/env python3
"""Single-prompt chat with Gemma 4 31B.

Usage:
    python3 chat.py "What is the capital of France?"
    python3 chat.py  # interactive prompt
"""
import sys, os, struct

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/"
    "snapshots/535c5606372deb5d5ab7e29280f111ef2a8e084e/"
)
BINARY = os.path.join(ROOT, "build", "gemma4")
PROMPT_FILE = os.path.join(ROOT, "prompt.bin")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)

def tokenize(user_text):
    prompt = f"<start_of_turn>user\n{user_text}\n<end_of_turn>\n<start_of_turn>model\n"
    return tok.encode(prompt, add_special_tokens=True)

def write_tokens(tokens, path):
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(tokens)))
        for t in tokens:
            f.write(struct.pack('<i', t))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_text = " ".join(sys.argv[1:])
    else:
        user_text = input("You: ")

    tokens = tokenize(user_text)
    print(f"[{len(tokens)} tokens]")
    write_tokens(tokens, PROMPT_FILE)

    os.chdir(ROOT)
    os.execv(BINARY, [BINARY])
