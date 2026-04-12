#!/bin/bash
# TurboQuant Regression Test Suite
# Tests Gemma 4 31B inference across diverse task types
set -e
cd "$(dirname "$0")"

PASS=0
FAIL=0
TOTAL=0

run_test() {
    local name="$1"
    local prompt="$2"
    local expected="$3"
    TOTAL=$((TOTAL + 1))

    python3 -c "
from transformers import AutoTokenizer
import struct, os
w = os.path.expanduser('~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/535c5606372deb5d5ab7e29280f111ef2a8e084e/')
tok = AutoTokenizer.from_pretrained(w)
prompt = '<start_of_turn>user\n${prompt}\n<end_of_turn>\n<start_of_turn>model\n'
ids = tok.encode(prompt, add_special_tokens=True)
with open('prompt.bin', 'wb') as f:
    f.write(struct.pack('<I', len(ids)))
    for t in ids: f.write(struct.pack('<i', t))
" 2>/dev/null
    rm -f kv_cache.safetensors
    output=$(./build/gemma4 2>&1)

    if echo "$output" | grep -qi "$expected"; then
        echo "  PASS: $name (found '$expected')"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $name (expected '$expected')"
        FAIL=$((FAIL + 1))
    fi
}

echo "TurboQuant Regression Tests"
echo "=========================="

run_test "Math: 11*12" "What is 11 times 12?" "132"
run_test "Capital" "What is the capital of Germany?" "Berlin"
run_test "Science" "What gas do plants absorb?" "carbon dioxide"
run_test "History" "Who painted the Sistine Chapel ceiling?" "Michelangelo"
run_test "Logic" "If all roses are flowers and all flowers need water, do roses need water?" "yes"

echo ""
echo "Results: $PASS/$TOTAL passed, $FAIL failed"
exit $FAIL
