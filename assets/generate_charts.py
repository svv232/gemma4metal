#!/usr/bin/env python3
"""Generate benchmark charts for TurboQuant README and tweet."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# --- Data ---
configs = [
    ("MoE 26B-A4B",     59, 14.5, "#FF6B35"),
    ("MoE + int4 KV",   53, 14.5, "#FF9F1C"),
    ("31B dense",        15, 17.5, "#2EC4B6"),
    ("31B + int4 KV",    14, 17.4, "#2EC4B6"),
    ("31B (C++ Metal)",  10, 17.4, "#5C8A8A"),
    ("E2B 2B",          111,  3.6, "#CCCCCC"),
]

names = [c[0] for c in configs]
speeds = [c[1] for c in configs]
memory = [c[2] for c in configs]
colors = [c[3] for c in configs]

# Sort by speed descending (excluding E2B which is draft-only)
order = sorted(range(len(configs)), key=lambda i: -speeds[i])
names_s = [names[i] for i in order]
speeds_s = [speeds[i] for i in order]
colors_s = [colors[i] for i in order]

# --- Chart 1: Speed Comparison ---
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(names_s)), speeds_s, color=colors_s, height=0.6, edgecolor='white', linewidth=0.5)

# Add value labels
for bar, val, name in zip(bars, speeds_s, names_s):
    ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
            f'{val} tok/s', va='center', fontsize=13, fontweight='bold')

ax.set_yticks(range(len(names_s)))
ax.set_yticklabels(names_s, fontsize=12)
ax.set_xlabel('Tokens per Second', fontsize=13)
ax.set_title('Gemma 4 Decode Speed — Apple M1 Max (64GB)', fontsize=15, fontweight='bold', pad=15)
ax.set_xlim(0, max(speeds_s) * 1.2)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotation
ax.annotate('MoE is 4x faster\nthan 31B dense',
            xy=(59, 0), xytext=(80, 2.5),
            fontsize=11, fontstyle='italic', color='#FF6B35',
            arrowprops=dict(arrowstyle='->', color='#FF6B35', lw=1.5))

plt.tight_layout()
plt.savefig('/Users/andromeda/marketing/turboquant/benchmark_speed.png', dpi=150, bbox_inches='tight')
print("Saved benchmark_speed.png")

# --- Chart 2: Memory Usage ---
fig, ax = plt.subplots(figsize=(10, 5))

# Sort by memory
order_m = sorted(range(len(configs)), key=lambda i: memory[i])
names_m = [names[i] for i in order_m]
mem_m = [memory[i] for i in order_m]
colors_m = [colors[i] for i in order_m]

bars = ax.barh(range(len(names_m)), mem_m, color=colors_m, height=0.6, edgecolor='white', linewidth=0.5)

for bar, val in zip(bars, mem_m):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.1f} GB', va='center', fontsize=13, fontweight='bold')

# 64GB line
ax.axvline(x=64, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax.text(64.5, len(names_m)-0.5, '64 GB limit', color='red', alpha=0.7, fontsize=10)

ax.set_yticks(range(len(names_m)))
ax.set_yticklabels(names_m, fontsize=12)
ax.set_xlabel('Peak Memory (GB)', fontsize=13)
ax.set_title('Gemma 4 Memory Usage — Apple M1 Max', fontsize=15, fontweight='bold', pad=15)
ax.set_xlim(0, 70)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/Users/andromeda/marketing/turboquant/benchmark_memory.png', dpi=150, bbox_inches='tight')
print("Saved benchmark_memory.png")

# --- Chart 3: Combined tweet chart (speed + memory side by side) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Speed (left) - only the 5 main configs, not E2B
main = [(n, s, m, c) for n, s, m, c in configs if n != "E2B 2B"]
main_sorted = sorted(main, key=lambda x: -x[1])
mn = [x[0] for x in main_sorted]
ms = [x[1] for x in main_sorted]
mc = [x[3] for x in main_sorted]
mm = [x[2] for x in main_sorted]

bars1 = ax1.barh(range(len(mn)), ms, color=mc, height=0.6, edgecolor='white')
for bar, val in zip(bars1, ms):
    ax1.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
             f'{val}', va='center', fontsize=14, fontweight='bold')
ax1.set_yticks(range(len(mn)))
ax1.set_yticklabels(mn, fontsize=12)
ax1.set_xlabel('tok/s', fontsize=12)
ax1.set_title('Decode Speed', fontsize=14, fontweight='bold')
ax1.set_xlim(0, max(ms) * 1.25)
ax1.invert_yaxis()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Memory (right)
main_mem = sorted(main, key=lambda x: x[2])
mn2 = [x[0] for x in main_mem]
mm2 = [x[2] for x in main_mem]
mc2 = [x[3] for x in main_mem]

bars2 = ax2.barh(range(len(mn2)), mm2, color=mc2, height=0.6, edgecolor='white')
for bar, val in zip(bars2, mm2):
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.1f} GB', va='center', fontsize=14, fontweight='bold')
ax2.set_yticks(range(len(mn2)))
ax2.set_yticklabels(mn2, fontsize=12)
ax2.set_xlabel('Peak Memory (GB)', fontsize=12)
ax2.set_title('Memory Usage', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 25)
ax2.invert_yaxis()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('Gemma 4 on Apple M1 Max (64GB) — 158 Experiments', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/andromeda/marketing/turboquant/benchmark_combined.png', dpi=150, bbox_inches='tight')
print("Saved benchmark_combined.png")

# --- Chart 4: 256K Context Memory Projection ---
fig, ax = plt.subplots(figsize=(10, 5))

ctx_configs = [
    ("MoE + int4 KV",   2.1, 14.5),
    ("MoE + FP16 KV",  10.5, 14.5),
    ("31B + int4 KV",   6.1, 17.4),
    ("31B + FP16 KV",  30.5, 17.4),
    ("31B + FP32 KV",  60.9, 17.4),
]

cnames = [c[0] for c in ctx_configs]
kv_mem = [c[1] for c in ctx_configs]
w_mem = [c[2] for c in ctx_configs]

y_pos = range(len(cnames))
bars_w = ax.barh(y_pos, w_mem, height=0.5, color='#2EC4B6', label='Weights', edgecolor='white')
bars_kv = ax.barh(y_pos, kv_mem, height=0.5, left=w_mem, color='#FF6B35', label='KV Cache @ 256K', edgecolor='white')

# Total labels
for i, (kv, w) in enumerate(zip(kv_mem, w_mem)):
    total = kv + w
    color = '#2d6a2d' if total < 64 else '#cc0000'
    symbol = '  ✓' if total < 64 else '  ✗'
    ax.text(total + 0.5, i, f'{total:.1f} GB{symbol}', va='center', fontsize=12, fontweight='bold', color=color)

ax.axvline(x=64, color='red', linestyle='--', alpha=0.6, linewidth=2)
ax.text(64.5, -0.4, '64 GB M1 Max', color='red', fontsize=11, fontweight='bold', alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(cnames, fontsize=12)
ax.set_xlabel('Memory (GB)', fontsize=13)
ax.set_title('256K Context: Will It Fit on 64GB?', fontsize=15, fontweight='bold', pad=15)
ax.set_xlim(0, 85)
ax.legend(loc='lower right', fontsize=11)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/Users/andromeda/marketing/turboquant/benchmark_256k.png', dpi=150, bbox_inches='tight')
print("Saved benchmark_256k.png")

print("\nAll charts generated!")
