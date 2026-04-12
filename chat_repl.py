#!/usr/bin/env python3
"""Interactive multi-turn chat with Gemma 4 31B.
Keeps the C++ inference process running between turns to avoid restart overhead.
Writes tokenized prompts as binary files and launches the C++ binary per turn.
The KV cache persists via safetensors between invocations."""
import sys, os, struct, subprocess, time

from transformers import AutoTokenizer

MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/"
    "snapshots/535c5606372deb5d5ab7e29280f111ef2a8e084e/"
)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "gemma4_multilayer")
PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_491.bin")
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kv_cache.safetensors")

def tokenize_turn(text, is_first=True):
    if is_first:
        prompt = f"<start_of_turn>user\n{text}\n<end_of_turn>\n<start_of_turn>model\n"
        return tok.encode(prompt, add_special_tokens=True)
    else:
        prompt = f"<end_of_turn>\n<start_of_turn>user\n{text}\n<end_of_turn>\n<start_of_turn>model\n"
        return tok.encode(prompt, add_special_tokens=False)

def write_prompt(tokens):
    with open(PROMPT_FILE, 'wb') as f:
        f.write(struct.pack('<I', len(tokens)))
        for t in tokens:
            f.write(struct.pack('<i', t))

def run_inference():
    result = subprocess.run(
        [BINARY],
        capture_output=True, text=True, timeout=300,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    # Extract the model response from output
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
    # Clean up thought tokens
    for marker in ['<|channel>', '<channel|>', 'thought', '-//-', '<turn|>']:
        text = text.replace(marker, '')
    text = text.strip()
    # Remove leading newlines/dashes
    while text and text[0] in '-\n ':
        text = text[1:]
    return text, result.stdout

if __name__ == "__main__":
    # Clear cache for fresh conversation
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

    print("Gemma 4 31B — Interactive Chat (int4 KV, 6.0x compression)")
    print("Type 'quit' to exit, 'clear' to reset conversation\n")

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
        write_prompt(tokens)

        t0 = time.time()
        response, raw = run_inference()
        elapsed = time.time() - t0

        # Extract tok/s from raw output
        tok_s = ""
        for line in raw.split('\n'):
            if 'tok/s' in line:
                tok_s = line.strip()
                break

        print(f"\nGemma: {response}")
        print(f"  [{tok_s}]\n")

        turn += 1
