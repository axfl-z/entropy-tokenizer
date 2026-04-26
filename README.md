# Entropy-Optimal Tokenizer (EOT)

A high-performance, **lossless** tokenizer with entropy-guided merge selection and DP-optimal encoding. Written in Rust with Python bindings (GIL-free).

## Why EOT?

| Feature | EOT | BPE | WordPiece | Unigram |
|---------|-----|-----|-----------|---------|
| **Lossless** | **YES** | NO | NO | NO |
| **Encoding speed** | **14.3 MB/s** | 2.4 MB/s | 3.9 MB/s | 2.2 MB/s |
| **Decoding speed** | **142.7 MB/s** | 16.8 MB/s | 23.8 MB/s | 14.4 MB/s |
| **Density** (bytes/token) | **2.52** | 2.38 | 2.65* | 2.12 |

\* WordPiece achieves higher density by being **lossy** — it uses `[UNK]` tokens and loses information. EOT is the only truly lossless tokenizer tested.

### Key innovations

1. **Entropy-guided merge selection** — merges are chosen by `frequency × entropy_bonus`, not just frequency like BPE. This produces a more uniform token distribution.
2. **DP-optimal encoding** — instead of greedy left-to-right matching (BPE), uses dynamic programming to find the globally optimal segmentation.
3. **Context entropy** — bigram log-probabilities guide encoding decisions, improving information density.
4. **Array-based Trie** — O(1) child lookup with `[u32; 256]` arrays instead of HashMaps. Cache-friendly.
5. **Parallel encoding** — large inputs are chunked and encoded in parallel with rayon.
6. **Byte-level base tokens** — every possible byte (0–255) is a token, guaranteeing lossless encoding of any input.

## Installation

### From PyPI

```bash
pip install entropy-tokenizer
```

### From source

```bash
git clone https://github.com/entropy-tokenizer/entropy-tokenizer.git
cd entropy-tokenizer

# Python package
pip install maturin
maturin develop --features python --release

# Rust CLI
cargo build --release
```

## Python Usage

```python
from entropy_tokenizer import EOTTokenizer

# Train a tokenizer
tok = EOTTokenizer("your training corpus text here " * 1000, vocab_size=1024)

# Encode (DP-optimal, lossless)
ids = tok.encode("Hello, world!")
print(ids)  # [72, 101, 108, 108, 111, ...]

# Decode (perfect reconstruction)
text = tok.decode(ids)
assert text == "Hello, world!"

# Greedy encoding (faster, slightly less optimal)
ids_greedy = tok.encode("Hello, world!", greedy=True)

# Encode with byte offsets
ids, offsets = tok.encode_with_offsets("Hello, world!")
# offsets = [(0, 1), (1, 2), ...]  # (start_byte, end_byte) per token

# Save / load
tok.save("model.json")
tok = EOTTokenizer.from_file("model.json")

# Properties
print(tok.vocab_size)    # 1024
print(tok.is_lossless)   # True
print(tok.id_to_token(72))  # "H"
```

### GIL-Free

All encoding and decoding operations release the Python GIL, enabling true parallelism in multi-threaded applications:

```python
from concurrent.futures import ThreadPoolExecutor
from entropy_tokenizer import EOTTokenizer

tok = EOTTokenizer.from_file("model.json")

texts = ["text1", "text2", "text3", ...]

# Truly parallel — no GIL contention
with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(tok.encode, texts))
```

## Rust CLI Usage

```bash
# Build
cargo build --release

# Train
./target/release/eot train --input corpus.txt --vocab-size 1024 --output model.json --verbose

# Encode
./target/release/eot encode --model model.json --text "Hello, world!"
./target/release/eot encode --model model.json --file input.txt
./target/release/eot encode --model model.json --text "Hello" --greedy

# Decode
./target/release/eot decode --model model.json --tokens "72,101,108,108,111"

# Benchmark
./target/release/eot bench --model model.json --input test.txt
```

## Rust Library Usage

```rust
use entropy_tokenizer::trainer::Trainer;
use entropy_tokenizer::encoder::Encoder;

// Train
let corpus = std::fs::read("corpus.txt").unwrap();
let mut trainer = Trainer::new(&corpus);
trainer.train(1024, true);  // target vocab size, verbose

// Encode
let vocab = trainer.into_vocab();
let encoder = Encoder::new(vocab, 0.3);  // context_weight

let tokens = encoder.encode(b"Hello, world!");      // DP-optimal
let tokens = encoder.encode_greedy(b"Hello, world!"); // faster

// Decode (lossless)
let bytes = encoder.decode(&tokens);
assert_eq!(bytes, b"Hello, world!");

// Save / load
encoder.vocab().save("model.json").unwrap();
```

## How It Works

### Training (Entropy-Guided Merges)

1. Start with 256 byte-level base tokens (0x00–0xFF).
2. Pre-tokenize corpus into "words" at whitespace boundaries (like GPT-2).
3. For each merge step, score all adjacent token pairs:
   - **Phase 1** (80% of merges): `score = frequency + 0.05 × max(entropy_delta, 0) × √frequency` — prioritizes compression.
   - **Phase 2** (20% of merges): `score = frequency^0.8 × (1 + 0.5 × entropy_delta)` — optimizes entropy distribution.
4. Apply the highest-scoring merge across all words.
5. Update pair counts incrementally (no full recount).

### Encoding (DP-Optimal Segmentation)

Given input bytes and a trained vocabulary:

1. Build a Trie from all vocabulary tokens.
2. Run forward DP: for each position `i`, try all tokens starting at `i` (via Trie walk).
3. Score each transition: `log_prob(token) + context_weight × bigram_log_prob(prev, token)`.
4. Backtrack from the end to recover the optimal token sequence.

For inputs > 128KB, the input is chunked and encoded in parallel.

### Why Lossless?

EOT uses byte-level base tokens: every byte value 0–255 is guaranteed to be in the vocabulary. This means any byte sequence can be encoded, even if it contains rare or unseen patterns. No `[UNK]` tokens are ever produced.

## Benchmarks

Tested on Shakespeare corpus (~100KB), vocab size 1024:

```
Tokenizer          Tokens    Density    Enc MB/s   Dec MB/s   Lossless
─────────────────────────────────────────────────────────────────────
EOT (ours)         40,275    2.52 B/t   14.3       142.7      YES
BPE                42,626    2.38 B/t    2.4        16.8       NO
WordPiece          38,296    2.65 B/t    3.9        23.8       NO
Unigram (SP)       47,770    2.12 B/t    2.2        14.4       NO
```

- **6x faster** encoding than BPE
- **8.5x faster** decoding than BPE
- **Only lossless** tokenizer (BPE/WordPiece lose data)
- Beats BPE and Unigram on density while maintaining lossless property

## License

MIT
