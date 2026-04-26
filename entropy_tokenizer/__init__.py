"""
Entropy-Optimal Tokenizer (EOT)
===============================

A high-performance, lossless tokenizer with entropy-guided merge selection
and DP-optimal encoding. Rust backend, GIL-free.

Usage::

    from entropy_tokenizer import EOTTokenizer

    # Train from text
    tok = EOTTokenizer("your training corpus here", vocab_size=1024)
    ids = tok.encode("hello world")
    text = tok.decode(ids)
    assert text == "hello world"  # lossless!

    # Save / load
    tok.save("model.json")
    tok = EOTTokenizer.from_file("model.json")

    # Greedy (faster) encoding
    ids = tok.encode("hello world", greedy=True)
"""

from entropy_tokenizer.entropy_tokenizer_core import EOTTokenizer, __version__
from entropy_tokenizer.compat import HFCompatTokenizer, Encoding

__all__ = ["EOTTokenizer", "HFCompatTokenizer", "Encoding", "__version__"]
