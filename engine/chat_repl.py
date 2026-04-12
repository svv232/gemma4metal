#!/usr/bin/env python3
"""Interactive multi-turn REPL for Gemma 4 31B.

Streams tokens as they're generated. Sends full conversation
history each turn so the model sees all prior context.
Type 'quit' to exit, 'clear' to reset conversation.
"""
import os, sys, struct, subprocess, re

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

THINK_END = '<channel|>'
TAG_RE = re.compile(r'<[^>]*\|[^>]*>')

SYSTEM_PROMPT = """\
You are Gemma, a helpful AI assistant running locally on a MacBook Pro M1 Max \
with 64GB of unified memory. You are Google's Gemma 4 31B model, quantized to \
4-bit and running via TurboQuant — an open-source Metal inference engine built \
specifically for Apple Silicon.

You are direct, concise, and knowledgeable. You remember everything the user \
has said in this conversation. When asked about yourself, you can mention that \
you are running entirely on-device with no internet connection or cloud API — \
all computation happens locally on the user's machine at ~10 tokens per second.

Keep responses focused and useful. Avoid excessive hedging or disclaimers."""


def build_prompt(history):
    # Gemma 4 uses: <|turn>role\n...<turn|>
    # Model turn starts with: <|turn>model\n<|channel>thought\n<channel|>\n
    parts = [f"<|turn>system\n{SYSTEM_PROMPT}<turn|>"]
    for role, text in history:
        parts.append(f"<|turn>{role}\n{text}<turn|>")
    # Open model turn with thinking block (model expects this)
    parts.append("<|turn>model\n<|channel>thought\n<channel|>\n")
    return "\n".join(parts)


def write_tokens(tokens):
    with open(PROMPT_FILE, 'wb') as f:
        f.write(struct.pack('<I', len(tokens)))
        for t in tokens:
            f.write(struct.pack('<i', t))


def run_inference_streaming():
    """Run inference and stream tokens as they arrive."""
    proc = subprocess.Popen(
        [BINARY], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=ROOT, bufsize=0,
    )

    in_gen = False
    tok_s = ""
    response_parts = []
    pre_buf = ""     # buffer before generation starts
    tag_buf = ""     # holds partial <...> tags
    byte_buf = b""

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

        if not in_gen:
            pre_buf += char
            if 'Generating:\n' in pre_buf:
                in_gen = True
                sys.stdout.write("\nGemma: ")
                sys.stdout.flush()
            continue

        # Detect end of generation
        if char == '\n':
            # Flush any held tag buffer as text (wasn't a real tag)
            if tag_buf:
                sys.stdout.write(tag_buf)
                sys.stdout.flush()
                response_parts.append(tag_buf)
                tag_buf = ""
            continue

        # Hold back characters that might be part of a tag
        if tag_buf:
            tag_buf += char
            if '>' in tag_buf:
                # Complete tag — check if it's a marker to suppress
                if any(m in tag_buf for m in ['channel', 'turn', 'EOS']):
                    if '[EOS]' in tag_buf:
                        break
                    # Discard the tag
                    tag_buf = ""
                else:
                    # Not a known marker, print it
                    sys.stdout.write(tag_buf)
                    sys.stdout.flush()
                    response_parts.append(tag_buf)
                    tag_buf = ""
            elif len(tag_buf) > 20:
                # Too long to be a tag, flush it
                sys.stdout.write(tag_buf)
                sys.stdout.flush()
                response_parts.append(tag_buf)
                tag_buf = ""
        elif char == '<':
            tag_buf = char
        elif char == '[':
            tag_buf = char
        else:
            sys.stdout.write(char)
            sys.stdout.flush()
            response_parts.append(char)

    # Grab stats from remaining output
    try:
        remaining = proc.stdout.read().decode("utf-8", errors="replace")
        for line in remaining.split('\n'):
            if 'tok/s' in line:
                tok_s = line.strip()
                break
    except Exception:
        pass

    proc.wait()
    return "".join(response_parts).strip(), tok_s


if __name__ == "__main__":
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

    print("Gemma 4 31B — Interactive Chat (streaming)")
    print("Type 'quit' to exit, 'clear' to reset\n")

    history = []

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
            history = []
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
            print("[Conversation cleared]\n")
            continue

        history.append(("user", user_input))

        # Use the tokenizer's own chat template for correct formatting
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for role, text in history:
            messages.append({"role": role, "content": text})
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tok.encode(prompt, add_special_tokens=False)
        write_tokens(tokens)

        # Fresh prefill with full conversation history each turn
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)

        response, tok_s = run_inference_streaming()
        print()
        if tok_s:
            print(f"  [{tok_s}]")
        print()

        if response:
            history.append(("model", response))
