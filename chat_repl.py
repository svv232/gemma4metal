#!/usr/bin/env python3
"""Interactive multi-turn REPL for Gemma 4 31B.

Streams tokens as they're generated.
Type 'quit' to exit, 'clear' to reset conversation.
"""
import os, sys, struct, subprocess, select

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


def clean_thought_markers(text):
    for marker in ['<|channel>', '<channel|>', 'thought', '-//-', '<turn|>']:
        text = text.replace(marker, '')
    return text


def run_inference_streaming():
    """Run inference and stream generated tokens to stdout in real time."""
    proc = subprocess.Popen(
        [BINARY],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT,
        bufsize=0,  # unbuffered
    )

    in_gen = False
    tok_s = ""
    buf = b""

    # Read byte-by-byte from stdout to get tokens as soon as they're flushed
    while True:
        byte = proc.stdout.read(1)
        if not byte:
            break
        buf += byte

        # Try to decode — handles multi-byte UTF-8
        try:
            char = buf.decode("utf-8")
        except UnicodeDecodeError:
            continue  # incomplete multi-byte char, read more
        buf = b""

        if char == '\n':
            # We need to track lines for state transitions
            if not hasattr(run_inference_streaming, '_line'):
                run_inference_streaming._line = ""
            line = run_inference_streaming._line

            if 'Generating:' in line:
                in_gen = True
                sys.stdout.write("\nGemma: ")
                sys.stdout.flush()
            elif in_gen and ('tokens in' in line or '[EOS]' in line):
                # End of generation
                if '[EOS]' in line:
                    pass  # don't print EOS marker
                in_gen = False
            elif in_gen and 'tok/s' in line:
                tok_s = line.strip()
                in_gen = False
            elif 'tok/s' in line:
                tok_s = line.strip()

            run_inference_streaming._line = ""
        else:
            if not hasattr(run_inference_streaming, '_line'):
                run_inference_streaming._line = ""
            run_inference_streaming._line += char

            if in_gen:
                # Stream the character immediately
                cleaned = clean_thought_markers(char)
                if cleaned:
                    sys.stdout.write(cleaned)
                    sys.stdout.flush()

    run_inference_streaming._line = ""
    proc.wait()
    return tok_s


if __name__ == "__main__":
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

    print("Gemma 4 31B — Interactive Chat (streaming)")
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

        tok_s = run_inference_streaming()
        print()  # newline after streamed output
        if tok_s:
            print(f"  [{tok_s}]")
        print()
        turn += 1
