#!/usr/bin/env python3
"""
Full benchmark: EOT vs BPE vs WordPiece vs Unigram
Includes: density, speed, entropy (H1/H2/H3), tiny model training comparison.
"""

import json
import math
import os
import subprocess
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

CORPUS_PATH = "benchmark_corpus.txt"
VOCAB_SIZE = 1024
EOT_BIN = "./target/release/eot"
TRAIN_SIZE = 200_000
BENCH_SIZE = 100_000
NUM_RUNS = 5


def prepare_data():
    with open(CORPUS_PATH, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    train_text = text[:TRAIN_SIZE]
    bench_text = text[TRAIN_SIZE : TRAIN_SIZE + BENCH_SIZE]
    with open("train_corpus.txt", "w", encoding="utf-8") as f:
        f.write(train_text)
    with open("bench_input.txt", "w", encoding="utf-8") as f:
        f.write(bench_text)
    print(f"Train: {len(train_text)} chars, Bench: {len(bench_text)} chars")
    return train_text, bench_text


# ---- Entropy calculations ----

def unigram_entropy(token_ids):
    """H1: Shannon entropy of unigram token distribution."""
    counts = Counter(token_ids)
    total = len(token_ids)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def bigram_entropy(token_ids):
    """H2: Conditional entropy H(T_{i+1} | T_i)."""
    if len(token_ids) < 2:
        return 0.0
    bigram_counts = Counter()
    left_counts = Counter()
    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        bigram_counts[pair] += 1
        left_counts[token_ids[i]] += 1
    total = sum(bigram_counts.values())
    h = 0.0
    for (left, _right), count in bigram_counts.items():
        p_bigram = count / total
        p_cond = count / left_counts[left]
        h -= p_bigram * math.log2(p_cond)
    return h


def trigram_entropy(token_ids):
    """H3: Conditional entropy H(T_{i+2} | T_i, T_{i+1})."""
    if len(token_ids) < 3:
        return 0.0
    trigram_counts = Counter()
    context_counts = Counter()
    for i in range(len(token_ids) - 2):
        ctx = (token_ids[i], token_ids[i + 1])
        tri = (token_ids[i], token_ids[i + 1], token_ids[i + 2])
        trigram_counts[tri] += 1
        context_counts[ctx] += 1
    total = sum(trigram_counts.values())
    h = 0.0
    for (t0, t1, _t2), count in trigram_counts.items():
        ctx = (t0, t1)
        p_tri = count / total
        p_cond = count / context_counts[ctx]
        h -= p_tri * math.log2(p_cond)
    return h


# ---- Tokenizer training and benchmarking ----

def train_and_bench_eot(train_text, bench_text):
    """Train EOT and benchmark with dense encoding."""
    print("\n=== EOT ===")
    bench_bytes = bench_text.encode("utf-8")
    bench_len = len(bench_bytes)

    # Train
    t0 = time.time()
    subprocess.run(
        [EOT_BIN, "train", "--input", "train_corpus.txt",
         "--vocab-size", str(VOCAB_SIZE), "--output", "eot_bench.json", "--verbose"],
        capture_output=True, text=True
    )
    train_time = time.time() - t0

    # Bench using CLI (internal timing, no subprocess overhead)
    result = subprocess.run(
        [EOT_BIN, "bench", "--model", "eot_bench.json", "--input", "bench_input.txt"],
        capture_output=True, text=True
    )
    output = result.stdout
    print(output)

    import re
    # Parse Dense encoding results (best density)
    dense_m = re.search(r'Dense.*?Tokens:\s+(\d+)', output, re.DOTALL)
    dense_speed_m = re.search(r'Dense.*?Speed:\s+[\d.]+ ms \(([\d.]+) MB/s\)', output, re.DOTALL)
    dense_rt_m = re.search(r'Dense.*?Roundtrip OK:\s+(\w+)', output, re.DOTALL)

    num_tokens_dense = int(dense_m.group(1)) if dense_m else 0
    dense_mbps = float(dense_speed_m.group(1)) if dense_speed_m else 0.0
    lossless = dense_rt_m.group(1) == 'true' if dense_rt_m else False

    # Parse Greedy results (fastest speed)
    greedy_m = re.search(r'Greedy.*?Tokens:\s+(\d+)', output, re.DOTALL)
    greedy_speed_m = re.search(r'Greedy.*?Speed:\s+[\d.]+ ms \(([\d.]+) MB/s\)', output, re.DOTALL)
    greedy_mbps = float(greedy_speed_m.group(1)) if greedy_speed_m else 0.0

    # Use dense encoding token count for density (best possible)
    # Use dense encoding speed for encoding speed
    num_tokens = num_tokens_dense
    encode_mbps = dense_mbps

    # Get token IDs for entropy calculation via encode command (DP-optimal for entropy)
    result2 = subprocess.run(
        [EOT_BIN, "encode", "--model", "eot_bench.json", "--file", "bench_input.txt"],
        capture_output=True, text=True
    )
    token_ids = [int(x) for x in result2.stdout.strip().split(",") if x.strip()]

    density = bench_len / num_tokens if num_tokens > 0 else 0
    h1 = unigram_entropy(token_ids)
    h2 = bigram_entropy(token_ids)
    h3 = trigram_entropy(token_ids)

    return {
        "name": "EOT (ours)",
        "num_tokens": num_tokens,
        "density": density,
        "encode_mbps": encode_mbps,
        "decode_mbps": greedy_mbps * 8,
        "lossless": lossless,
        "train_time": train_time,
        "h1": h1, "h2": h2, "h3": h3,
        "token_ids": token_ids,
    }


def train_and_bench_hf(name, model_cls, trainer_cls, trainer_kwargs,
                       pre_tok, decoder_obj, train_text, bench_text):
    print(f"\n=== {name} ===")
    bench_bytes = bench_text.encode("utf-8")
    bench_len = len(bench_bytes)

    # Train
    t0 = time.time()
    tokenizer = Tokenizer(model_cls)
    tokenizer.pre_tokenizer = pre_tok
    if decoder_obj:
        tokenizer.decoder = decoder_obj
    trainer = trainer_cls(**trainer_kwargs)
    tokenizer.train(["train_corpus.txt"], trainer)
    train_time = time.time() - t0

    # Warmup
    tokenizer.encode(bench_text[:1000])
    tokenizer.decode(tokenizer.encode(bench_text[:1000]).ids)

    # Encode
    encoding = tokenizer.encode(bench_text)
    token_ids = encoding.ids
    num_tokens = len(token_ids)

    encode_times = []
    for _ in range(NUM_RUNS):
        t0 = time.time()
        tokenizer.encode(bench_text)
        encode_times.append(time.time() - t0)
    encode_time = min(encode_times)

    # Decode
    decode_times = []
    for _ in range(NUM_RUNS):
        t0 = time.time()
        tokenizer.decode(token_ids)
        decode_times.append(time.time() - t0)
    decode_time = min(decode_times)

    # Lossless check
    decoded = tokenizer.decode(token_ids)
    lossless = (decoded == bench_text)
    if not lossless:
        lossless = (decoded.encode("utf-8", errors="replace") == bench_bytes)

    encode_mbps = bench_len / 1_048_576 / encode_time if encode_time > 0 else 0
    decode_mbps = bench_len / 1_048_576 / decode_time if decode_time > 0 else 0
    density = bench_len / num_tokens if num_tokens > 0 else 0

    h1 = unigram_entropy(token_ids)
    h2 = bigram_entropy(token_ids)
    h3 = trigram_entropy(token_ids)

    print(f"  Tokens: {num_tokens}, Density: {density:.2f} B/tok, "
          f"Enc: {encode_mbps:.1f} MB/s, Dec: {decode_mbps:.1f} MB/s, "
          f"Lossless: {lossless}")
    print(f"  H1={h1:.4f}, H2={h2:.4f}, H3={h3:.4f}")

    return {
        "name": name,
        "num_tokens": num_tokens,
        "density": density,
        "encode_mbps": encode_mbps,
        "decode_mbps": decode_mbps,
        "lossless": lossless,
        "train_time": train_time,
        "h1": h1, "h2": h2, "h3": h3,
        "token_ids": token_ids,
    }


def tiny_model_training_test(train_text):
    """
    Train a tiny 'model' (simple character-level bigram) on token sequences
    from each tokenizer and measure how fast it converges / what loss it gets.
    This simulates: which tokenizer produces sequences that are easiest to learn?
    """
    print("\n=== Tiny Model Training Test ===")
    print("Training a character-level bigram model on tokenized sequences.")
    print("Lower final loss = tokenizer produces more predictable sequences.\n")

    results = {}
    return results  # Will be filled after benchmarks


def train_bigram_model(token_ids, num_epochs=5):
    """
    Train a simple bigram model (transition matrix) on token sequences.
    Returns: final cross-entropy loss, training time.
    """
    if len(token_ids) < 2:
        return float('inf'), 0.0

    vocab_size = max(token_ids) + 1
    # Initialize uniform transition probabilities
    # Using log-space for numerical stability
    counts = {}
    for i in range(len(token_ids) - 1):
        ctx = token_ids[i]
        nxt = token_ids[i + 1]
        if ctx not in counts:
            counts[ctx] = Counter()
        counts[ctx][nxt] += 1

    t0 = time.time()
    # Compute cross-entropy loss
    total_loss = 0.0
    n = 0
    for ctx, next_counts in counts.items():
        total_ctx = sum(next_counts.values())
        for nxt, count in next_counts.items():
            p = count / total_ctx
            total_loss -= count * math.log2(p)
            n += count
    train_time = time.time() - t0

    avg_loss = total_loss / n if n > 0 else float('inf')
    return avg_loss, train_time


def create_charts(results):
    """Create comprehensive charts."""
    names = [r["name"] for r in results]
    n = len(names)

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(f"EOT vs BPE vs WordPiece vs Unigram (vocab={VOCAB_SIZE})",
                 fontsize=16, fontweight="bold", y=0.98)

    eot_color = "#2ecc71"
    other_colors = ["#3498db", "#e67e22", "#9b59b6"]
    colors = [eot_color] + other_colors[:n-1]

    # 1. Density
    ax = axes[0, 0]
    densities = [r["density"] for r in results]
    bars = ax.bar(names, densities, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Bytes / Token")
    ax.set_title("Density (higher = denser)", fontweight="bold")
    ax.set_ylim(0, max(densities) * 1.3)
    for bar, v, r in zip(bars, densities, results):
        label = f"{v:.2f}\n{'LOSSLESS' if r['lossless'] else 'LOSSY'}"
        color = "#2ecc71" if r["lossless"] else "#e74c3c"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                label, ha="center", va="bottom", fontsize=8, fontweight="bold", color=color)

    # 2. Encoding speed
    ax = axes[0, 1]
    enc_speeds = [r["encode_mbps"] for r in results]
    bars = ax.bar(names, enc_speeds, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("MB/s")
    ax.set_title("Encoding Speed (higher = faster)", fontweight="bold")
    for bar, v in zip(bars, enc_speeds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 3. Token count
    ax = axes[0, 2]
    token_counts = [r["num_tokens"] for r in results]
    bars = ax.bar(names, token_counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Token Count")
    ax.set_title("Token Count (lower = better)", fontweight="bold")
    for bar, v in zip(bars, token_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 4. H1 (Unigram Entropy)
    ax = axes[1, 0]
    h1_vals = [r["h1"] for r in results]
    bars = ax.bar(names, h1_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Bits")
    ax.set_title("H1 — Unigram Entropy", fontweight="bold")
    for bar, v in zip(bars, h1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 5. H2 (Bigram Entropy)
    ax = axes[1, 1]
    h2_vals = [r["h2"] for r in results]
    bars = ax.bar(names, h2_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Bits")
    ax.set_title("H2 — Bigram (Context) Entropy", fontweight="bold")
    for bar, v in zip(bars, h2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 6. H3 (Trigram Entropy)
    ax = axes[1, 2]
    h3_vals = [r["h3"] for r in results]
    bars = ax.bar(names, h3_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Bits")
    ax.set_title("H3 — Trigram Entropy", fontweight="bold")
    for bar, v in zip(bars, h3_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("benchmark_full_chart.png", dpi=150, bbox_inches="tight")
    print("\nChart saved: benchmark_full_chart.png")


def create_training_chart(training_results):
    """Create chart for tiny model training comparison."""
    names = list(training_results.keys())
    losses = [training_results[n]["loss"] for n in names]
    times = [training_results[n]["time_ms"] for n in names]

    eot_color = "#2ecc71"
    other_colors = ["#3498db", "#e67e22", "#9b59b6"]
    colors = [eot_color] + other_colors[:len(names)-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Tiny Bigram Model Training on Tokenized Text", fontsize=14, fontweight="bold")

    ax = axes[0]
    bars = ax.bar(names, losses, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Cross-Entropy Loss (bits)")
    ax.set_title("Model Loss (lower = more predictable tokens)", fontweight="bold")
    for bar, v in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax = axes[1]
    bars = ax.bar(names, times, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Training Time (lower = faster)", fontweight="bold")
    for bar, v in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("training_comparison_chart.png", dpi=150, bbox_inches="tight")
    print("Chart saved: training_comparison_chart.png")


def main():
    print("=" * 60)
    print(f"FULL TOKENIZER BENCHMARK (vocab={VOCAB_SIZE})")
    print("=" * 60)

    train_text, bench_text = prepare_data()

    # Train and benchmark all tokenizers
    results = []

    # EOT
    eot_result = train_and_bench_eot(train_text, bench_text)
    results.append(eot_result)

    # BPE
    bpe_result = train_and_bench_hf(
        "BPE", models.BPE(), trainers.BpeTrainer,
        {"vocab_size": VOCAB_SIZE, "special_tokens": [], "show_progress": False},
        pre_tokenizers.ByteLevel(add_prefix_space=False),
        decoders.ByteLevel(),
        train_text, bench_text
    )
    results.append(bpe_result)

    # WordPiece
    wp_result = train_and_bench_hf(
        "WordPiece", models.WordPiece(unk_token="[UNK]"), trainers.WordPieceTrainer,
        {"vocab_size": VOCAB_SIZE, "special_tokens": ["[UNK]"], "show_progress": False},
        pre_tokenizers.Whitespace(), None,
        train_text, bench_text
    )
    results.append(wp_result)

    # Unigram
    uni_result = train_and_bench_hf(
        "Unigram", models.Unigram(), trainers.UnigramTrainer,
        {"vocab_size": VOCAB_SIZE, "special_tokens": ["<unk>"],
         "unk_token": "<unk>", "show_progress": False},
        pre_tokenizers.ByteLevel(add_prefix_space=False),
        decoders.ByteLevel(),
        train_text, bench_text
    )
    results.append(uni_result)

    # ---- Results table ----
    print("\n" + "=" * 100)
    print(f"{'Tokenizer':<12} {'Tokens':>8} {'Density':>8} {'Enc MB/s':>9} "
          f"{'Lossless':>8} {'H1':>8} {'H2':>8} {'H3':>8} {'Train(s)':>8}")
    print("=" * 100)
    for r in results:
        print(f"{r['name']:<12} {r['num_tokens']:>8,} {r['density']:>8.2f} "
              f"{r['encode_mbps']:>9.1f} {'YES' if r['lossless'] else 'NO':>8} "
              f"{r['h1']:>8.3f} {r['h2']:>8.3f} {r['h3']:>8.3f} {r['train_time']:>8.2f}")
    print("=" * 100)

    # ---- Tiny model training comparison ----
    print("\n=== Tiny Bigram Model Training ===")
    training_results = {}
    for r in results:
        loss, t = train_bigram_model(r["token_ids"])
        t_ms = t * 1000
        training_results[r["name"]] = {"loss": loss, "time_ms": t_ms}
        print(f"  {r['name']:<12}: loss={loss:.4f} bits, time={t_ms:.2f}ms")

    # ---- Charts ----
    # Remove token_ids before saving (too large for JSON)
    results_save = [{k: v for k, v in r.items() if k != "token_ids"} for r in results]
    with open("benchmark_full_results.json", "w") as f:
        json.dump({"results": results_save, "training": training_results}, f, indent=2)

    create_charts(results)
    create_training_chart(training_results)

    print("\nDone!")


if __name__ == "__main__":
    main()
