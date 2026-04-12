#!/usr/bin/env python3
"""Interactive multi-turn chat with Gemma 4 31B.

Streams tokens in real-time. Shows the model's thinking process (dimmed)
followed by the final response. Full conversation history is sent each turn.

Usage:
    python3.12 engine/chat_repl.py

Commands:
    quit  — exit
    clear — reset conversation memory
"""
import os, sys, struct, subprocess, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/"
    "snapshots/535c5606372deb5d5ab7e29280f111ef2a8e084e/"
)
BINARY = os.path.join(ROOT, "build", "gemma4")
PROMPT_FILE = os.path.join(ROOT, "prompt.bin")
CACHE_FILE = os.path.join(ROOT, "kv_cache.safetensors")

# Load tokenizer
try:
    from mlx_lm import load as _load
    _, tok = _load(MODEL_DIR)
except ImportError:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

SYSTEM_PROMPT = """\
You are Gemma, a helpful AI assistant running locally on Apple Silicon. \
You are Google's Gemma 4 31B model running via a custom Metal inference engine \
with fused int4 SDPA attention kernels.

You are direct, concise, and knowledgeable. You remember everything the user \
has said in this conversation. Keep responses focused and useful."""

DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"


def write_tokens(tokens):
    with open(PROMPT_FILE, 'wb') as f:
        f.write(struct.pack('<I', len(tokens)))
        for t in tokens:
            f.write(struct.pack('<i', t))


def run_streaming():
    """Run the C++ engine and stream tokens, separating thought from response."""
    proc = subprocess.Popen(
        [BINARY], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=ROOT, bufsize=0,
    )

    # States: prefill → thinking → answering → done
    state = "prefill"
    thought_parts = []
    answer_parts = []
    pre_buf = ""
    tag_buf = ""
    byte_buf = b""
    tok_s = ""

    # Spinner during prefill
    spinning = True
    def spinner():
        frames = ["thinking.", "thinking..", "thinking..."]
        i = 0
        while spinning:
            sys.stdout.write(f"\r  {DIM}{frames[i % len(frames)]}{RESET}   ")
            sys.stdout.flush()
            i += 1
            time.sleep(0.4)
        sys.stdout.write("\r" + " " * 30 + "\r")
        sys.stdout.flush()

    spin_thread = threading.Thread(target=spinner, daemon=True)
    spin_thread.start()

    def emit(char):
        """Write a character in the appropriate style for current state."""
        if state == "thinking":
            sys.stdout.write(f"{DIM}{char}{RESET}")
            thought_parts.append(char)
        elif state == "answering":
            sys.stdout.write(char)
            answer_parts.append(char)
        sys.stdout.flush()

    while True:
        byte = proc.stdout.read(1)
        if not byte:
            break
        byte_buf += byte
        try:
            char = byte_buf.decode("utf-8")
        except UnicodeDecodeError:
            continue
        byte_buf = b""

        # Wait for "Generating:" marker from the C++ engine
        if state == "prefill":
            pre_buf += char
            if 'Generating:\n' in pre_buf:
                spinning = False
                spin_thread.join(timeout=1)
                state = "thinking"
                sys.stdout.write(f"  {DIM}thought: ")
                sys.stdout.flush()
            continue

        # Buffer potential tags
        if tag_buf:
            tag_buf += char
            if '>' in tag_buf or (tag_buf.startswith('[') and ']' in tag_buf):
                if 'EOS' in tag_buf or 'eos' in tag_buf:
                    break
                elif 'channel|' in tag_buf:
                    # <channel|> = transition from thinking to answering
                    if state == "thinking":
                        sys.stdout.write(f"{RESET}\n")
                        state = "answering"
                        sys.stdout.write(f"{BOLD}Gemma:{RESET} ")
                        sys.stdout.flush()
                    tag_buf = ""
                elif '|channel' in tag_buf:
                    # <|channel> = start of thinking (already in thinking state)
                    tag_buf = ""
                elif 'turn|' in tag_buf or '|turn' in tag_buf:
                    tag_buf = ""
                else:
                    emit(tag_buf)
                    tag_buf = ""
            elif len(tag_buf) > 30:
                emit(tag_buf)
                tag_buf = ""
        elif char in '<[':
            tag_buf = char
        elif char == '\n':
            if tag_buf:
                emit(tag_buf)
                tag_buf = ""
            if state == "answering":
                emit(char)
        else:
            emit(char)

    spinning = False

    # Extract speed
    try:
        remaining = proc.stdout.read().decode("utf-8", errors="replace")
        for line in remaining.split('\n'):
            if 'tok/s' in line:
                tok_s = line.strip()
                break
    except Exception:
        pass

    proc.wait()
    return "".join(answer_parts).strip(), tok_s


def main():
    if not os.path.exists(BINARY):
        print(f"Error: {BINARY} not found. Run 'cd build && cmake .. && make -j8' first.")
        sys.exit(1)

    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

    print(f"{BOLD}Gemma 4 31B{RESET} — TurboQuant Fused int4 SDPA")
    print(f"Type 'quit' to exit, 'clear' to reset memory\n")

    history = []

    while True:
        try:
            user_input = input(f"{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'clear':
            history = []
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
            print("[Memory cleared]\n")
            continue

        history.append(("user", user_input))
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for role, text in history:
            messages.append({"role": role, "content": text})

        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tok.encode(prompt, add_special_tokens=False)

        if len(tokens) > 900:
            print(f"  {DIM}[context: {len(tokens)} tokens — approaching int4 limit]{RESET}")

        write_tokens(tokens)

        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)

        response, tok_s = run_streaming()
        print()
        if tok_s:
            print(f"  {DIM}{tok_s}{RESET}")
        print()

        if response:
            history.append(("model", response))


if __name__ == "__main__":
    main()
