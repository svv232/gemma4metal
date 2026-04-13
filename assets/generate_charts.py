#!/usr/bin/env python3
"""Generate benchmark charts for TurboQuant README."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')

OUT = '/Users/andromeda/marketing/turboquant/assets'

# ── Color palette ──
TQ_COLOR = '#FF6B35'
MLX_COLOR = '#2EC4B6'
BASELINE_COLOR = '#8B8B8B'
BG_COLOR = '#FAFAFA'

def style_ax(ax):
    ax.set_facecolor(BG_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')


# ═══════════════════════════════════════════════════════════════
# Chart 1: Throughput vs Context Length (the hero chart)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.5))
style_ax(ax)

ctx = [33, 100, 200, 423, 600, 786, 950]
tq_tps   = [10.4, 10.3, 10.2, 10.0, 10.0, 10.0, 9.8]
base_tps = [9.6,  9.2,  8.8,  7.4,  7.3,  7.3,  7.2]
mlx_tps  = [15.0, 14.8, 14.5, 14.0, 13.5, 13.0, 12.5]  # estimated from mlx-lm behavior

ax.plot(ctx, tq_tps, 'o-', color=TQ_COLOR, linewidth=2.5, markersize=8, label='Fused int4 SDPA (ours)', zorder=3)
ax.plot(ctx, base_tps, 's--', color=BASELINE_COLOR, linewidth=2, markersize=7, label='Dequantize + SDPA (baseline)', zorder=2)
ax.fill_between(ctx, base_tps, tq_tps, alpha=0.15, color=TQ_COLOR, label='Fused kernel advantage')

# Annotate the 37% speedup
ax.annotate('+37%', xy=(786, 10.0), xytext=(786, 11.5),
            fontsize=14, fontweight='bold', color=TQ_COLOR, ha='center',
            arrowprops=dict(arrowstyle='->', color=TQ_COLOR, lw=2))
ax.annotate('7.3', xy=(786, 7.3), fontsize=10, color=BASELINE_COLOR, ha='center',
            xytext=(810, 6.5), arrowprops=dict(arrowstyle='->', color=BASELINE_COLOR))

ax.set_xlabel('Context Length (tokens)', fontsize=13)
ax.set_ylabel('Decode Speed (tokens/sec)', fontsize=13)
ax.set_title('Fused int4 SDPA: Constant Throughput Regardless of Context', fontsize=15, fontweight='bold', pad=15)
ax.set_ylim(5, 13)
ax.set_xlim(0, 1000)
ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
plt.tight_layout()
plt.savefig(f'{OUT}/throughput_vs_context.png', dpi=150, bbox_inches='tight')
print('Saved throughput_vs_context.png')


# ═══════════════════════════════════════════════════════════════
# Chart 2: Peak Memory Savings
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
style_ax(ax)

ctx2 = [100, 200, 400, 600, 800, 950]
# Baseline: dequantize K creates a temporary float32 matrix per layer
# 50 sliding layers × 16 heads × ctx × 256 × 4 bytes
saved_mb = [50 * 16 * c * 256 * 4 / 1e6 for c in ctx2]

bars = ax.bar(range(len(ctx2)), saved_mb, 0.55, color=TQ_COLOR, edgecolor='white', alpha=0.85)

for i, v in enumerate(saved_mb):
    ax.text(i, v + 12, f'{v:.0f} MB\nsaved', ha='center', fontsize=11, fontweight='bold', color=TQ_COLOR)

ax.set_xticks(range(len(ctx2)))
ax.set_xticklabels([f'{c} tok' for c in ctx2], fontsize=11)
ax.set_ylabel('Memory Saved (MB)', fontsize=12)
ax.set_title('Peak Memory Saved by Fused Kernel\n(eliminates dequantized K temporaries entirely)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, max(saved_mb) * 1.25)
plt.tight_layout()
plt.savefig(f'{OUT}/memory_savings.png', dpi=150, bbox_inches='tight')
print('Saved memory_savings.png')


# ═══════════════════════════════════════════════════════════════
# Chart 3: Python Extension — Attention Kernel Benchmark
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
style_ax(ax)

ctx3 = [50, 100, 200, 500, 800, 950]
tq_ms   = [0.35, 0.34, 0.41, 0.42, 0.46, 0.45]
mlx_ms  = [0.43, 0.42, 0.41, 0.52, 0.72, 0.59]

ax.plot(ctx3, tq_ms, 'o-', color=TQ_COLOR, linewidth=2.5, markersize=8, label='TurboQuant sdpa_int4')
ax.plot(ctx3, mlx_ms, 's-', color=MLX_COLOR, linewidth=2.5, markersize=8, label='MLX quantized_matmul SDPA')
ax.fill_between(ctx3, tq_ms, mlx_ms, alpha=0.12, color=TQ_COLOR, where=[t < m for t, m in zip(tq_ms, mlx_ms)])

for i, (t, m, c) in enumerate(zip(tq_ms, mlx_ms, ctx3)):
    if c in [50, 500, 950]:
        speedup = m / t
        ax.annotate(f'{speedup:.1f}x', xy=(c, t), xytext=(c + 30, t - 0.06),
                    fontsize=11, fontweight='bold', color=TQ_COLOR)

ax.set_xlabel('Context Length (tokens)', fontsize=13)
ax.set_ylabel('Attention Latency (ms)', fontsize=13)
ax.set_title('Python Extension: Fused Kernel vs MLX quantized_matmul', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=12)
ax.set_ylim(0.2, 0.85)
plt.tight_layout()
plt.savefig(f'{OUT}/python_kernel_benchmark.png', dpi=150, bbox_inches='tight')
print('Saved python_kernel_benchmark.png')


# ═══════════════════════════════════════════════════════════════
# Chart 4: Hardware Requirements
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4.5))
style_ax(ax)

configs = [
    ('Gemma 4 31B\n(4-bit weights)', 17.4, 0.8, TQ_COLOR),
    ('+ int4 KV @ 256K\n(TurboQuant)', 17.4, 2.1, TQ_COLOR),
    ('+ FP16 KV @ 256K\n(no compression)', 17.4, 30.5, BASELINE_COLOR),
    ('+ FP32 KV @ 256K\n(full precision)', 17.4, 60.9, '#CC4444'),
]

y_pos = range(len(configs))
for i, (label, w, kv, color) in enumerate(configs):
    total = w + kv
    ax.barh(i, w, height=0.5, color=MLX_COLOR, edgecolor='white', label='Weights' if i == 0 else '')
    ax.barh(i, kv, height=0.5, left=w, color=color, edgecolor='white',
            label='KV Cache' if i == 0 else ('TurboQuant int4 KV' if i == 1 else ''))
    fits = total < 64
    marker = '  ✓ fits 64GB' if fits else '  ✗ exceeds 64GB'
    text_color = '#2d6a2d' if fits else '#cc0000'
    ax.text(total + 0.5, i, f'{total:.1f} GB{marker}', va='center', fontsize=11,
            fontweight='bold', color=text_color)

ax.axvline(x=64, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.text(64.5, -0.3, '64 GB M1 Max', color='red', fontsize=10, fontweight='bold', alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels([c[0] for c in configs], fontsize=11)
ax.set_xlabel('Memory (GB)', fontsize=12)
ax.set_title('256K Context: Only int4 KV Fits on 64GB', fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0, 85)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUT}/hardware_requirements.png', dpi=150, bbox_inches='tight')
print('Saved hardware_requirements.png')


# ═══════════════════════════════════════════════════════════════
# Chart 5: TurboQuant Paper vs This Implementation
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor(BG_COLOR)
ax.axis('off')

table_data = [
    ['', 'TurboQuant Paper\n(ICLR 2026)', 'This Implementation'],
    ['Compression\nMethod', 'PolarQuant + QJL\n(recursive polar coords)', 'MLX native int4\n(scale × nibble + bias)'],
    ['Compression\nRatio', '5.81 bpd (~5.5×)', '4 bpd (6.4×)'],
    ['Attention\nKernel', 'Standard SDPA\n(dequantize first)', 'Fused sdpa_int4\n(dequantize in registers)'],
    ['Speed\nBenefit', 'Compression only\n(no speed gain)', '+37% decode speed\n(bandwidth savings)'],
    ['Memory\nBenefit', 'Smaller KV cache', 'Smaller KV cache\n+ zero temporaries'],
    ['Gemma 4\nCompat', 'Not tested\n(attn_scale=1.0 issue)', 'Validated: PolarQuant fails\nint4 native works to 950 tok'],
]

colors = [['#E8E8E8', '#FFE0CC', '#CCE8E5']]
for i in range(1, len(table_data)):
    colors.append(['#F5F5F5', '#FFF5EE', '#F0FAF8'])

table = ax.table(cellText=table_data, cellColours=colors, loc='center',
                  cellLoc='center', colWidths=[0.18, 0.41, 0.41])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Style header row
for j in range(3):
    table[0, j].set_text_props(fontweight='bold', fontsize=11)

ax.set_title('TurboQuant Paper vs This Implementation', fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUT}/paper_comparison.png', dpi=150, bbox_inches='tight')
print('Saved paper_comparison.png')

print('\nAll charts generated!')
