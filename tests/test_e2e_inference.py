"""
End-to-end inference test for TurboQuant with real MLX-LM models.

Tests generation quality and memory compression across models and presets.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "python"))

import mlx.core as mx
import mlx_lm
from mlx_lm import generate
from mlx_adapter import TurboQuantCache
from turboquant import TurboQuantConfig
import time


def test_model(model_name, head_dim, n_kv_heads, n_layers, prompts, presets=None):
    """Test TurboQuant on a specific model."""
    if presets is None:
        presets = ["balanced", "hifi"]

    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Architecture: {n_layers} layers, head_dim={head_dim}, n_kv_heads={n_kv_heads}")
    print(f"{'=' * 60}")

    model, tokenizer = mlx_lm.load(model_name)

    for prompt in prompts:
        print(f"\nPrompt: {prompt[:60]}...")

        # Standard generation
        t0 = time.perf_counter()
        out_std = generate(model, tokenizer, prompt=prompt, max_tokens=60)
        t_std = time.perf_counter() - t0
        print(f"\n  Standard ({t_std:.1f}s):")
        print(f"    {out_std[:150]}")

        # TurboQuant presets
        for preset in presets:
            config = TurboQuantConfig.preset(preset, embed_dim=head_dim, num_heads=n_kv_heads)
            tq_cache = [TurboQuantCache(config) for _ in range(n_layers)]

            t0 = time.perf_counter()
            out_tq = generate(model, tokenizer, prompt=prompt, max_tokens=60,
                              prompt_cache=tq_cache)
            t_tq = time.perf_counter() - t0

            mem = sum(c.memory_bytes() for c in tq_cache)
            fp16 = sum(c.fp16_bytes() for c in tq_cache)
            compression = fp16 / max(mem, 1)

            print(f"\n  TurboQuant [{preset}] ({t_tq:.1f}s, {compression:.2f}x):")
            print(f"    {out_tq[:150]}")


if __name__ == "__main__":
    prompts = [
        "The capital of France is",
        "Explain how quantum computing differs from classical computing in 3 sentences.",
        "Write a Python function to compute the Fibonacci sequence:",
    ]

    test_model(
        "HuggingFaceTB/SmolLM-135M-Instruct",
        head_dim=64, n_kv_heads=3, n_layers=30,
        prompts=prompts,
    )

    test_model(
        "Qwen/Qwen2.5-3B-Instruct",
        head_dim=128, n_kv_heads=2, n_layers=36,
        prompts=prompts,
    )

    print("\n" + "=" * 60)
    print("All end-to-end tests complete.")
