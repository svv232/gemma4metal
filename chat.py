#!/usr/bin/env python3
"""Interactive chat with Gemma 4 31B via TurboQuant C++ inference engine."""
import sys, os, struct, subprocess, tempfile

# Load tokenizer
from transformers import AutoTokenizer
MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/"
    "snapshots/535c5606372deb5d5ab7e29280f111ef2a8e084e/"
)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
BINARY = os.path.join(os.path.dirname(__file__), "build", "gemma4_multilayer")

def tokenize_prompt(user_text):
    prompt = f"<start_of_turn>user\n{user_text}\n<end_of_turn>\n<start_of_turn>model\n"
    return tok.encode(prompt, add_special_tokens=True)

def write_prompt_file(tokens, path):
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(tokens)))
        for t in tokens:
            f.write(struct.pack('<i', t))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_text = " ".join(sys.argv[1:])
    else:
        print("Gemma 4 31B + int4 KV Cache (6.4x compression)")
        print("=" * 50)
        user_text = input("You: ")

    tokens = tokenize_prompt(user_text)
    print(f"[{len(tokens)} tokens]")

    prompt_file = os.path.join(os.path.dirname(__file__), "prompt_491.bin")
    write_prompt_file(tokens, prompt_file)

    os.chdir(os.path.dirname(__file__))
    os.execv(BINARY, [BINARY])
