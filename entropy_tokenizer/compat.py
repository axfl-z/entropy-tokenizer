"""
HuggingFace tokenizers compatibility layer.

Provides a wrapper that makes EOTTokenizer compatible with the
HuggingFace tokenizers/transformers API, so you can use it as a
drop-in replacement in training and inference pipelines.

Usage::

    from entropy_tokenizer.compat import HFCompatTokenizer

    tok = HFCompatTokenizer.train("corpus.txt", vocab_size=1024)
    encoding = tok.encode("Hello, world!")
    print(encoding.ids)
    print(encoding.tokens)
    print(encoding.offsets)
    text = tok.decode(encoding.ids)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from entropy_tokenizer.entropy_tokenizer_core import EOTTokenizer


@dataclass
class Encoding:
    """Mimics tokenizers.Encoding from HuggingFace."""

    ids: List[int] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    offsets: List[Tuple[int, int]] = field(default_factory=list)
    type_ids: List[int] = field(default_factory=list)
    attention_mask: List[int] = field(default_factory=list)
    special_tokens_mask: List[int] = field(default_factory=list)
    overflowing: List["Encoding"] = field(default_factory=list)

    @property
    def n_sequences(self) -> int:
        return 1

    def word_to_tokens(self, word_index: int) -> Optional[Tuple[int, int]]:
        """Not supported — EOT is byte-level, not word-level."""
        return None

    def token_to_word(self, token_index: int) -> Optional[int]:
        """Not supported — EOT is byte-level, not word-level."""
        return None


class HFCompatTokenizer:
    """
    HuggingFace-compatible wrapper around EOTTokenizer.

    Provides the same API as ``tokenizers.Tokenizer`` so you can use
    EOT as a drop-in replacement in HuggingFace pipelines.
    """

    def __init__(self, eot: EOTTokenizer):
        self._eot = eot

    # ── Construction ──

    @classmethod
    def train(
        cls,
        corpus: Union[str, bytes],
        vocab_size: int = 8192,
        context_weight: float = 0.3,
        verbose: bool = False,
    ) -> "HFCompatTokenizer":
        """Train a new tokenizer from a corpus string or bytes."""
        eot = EOTTokenizer(corpus, vocab_size=vocab_size,
                           context_weight=context_weight, verbose=verbose)
        return cls(eot)

    @classmethod
    def train_from_file(
        cls,
        path: str,
        vocab_size: int = 8192,
        context_weight: float = 0.3,
        verbose: bool = False,
    ) -> "HFCompatTokenizer":
        """Train a new tokenizer from a file."""
        with open(path, "rb") as f:
            corpus = f.read()
        eot = EOTTokenizer(corpus, vocab_size=vocab_size,
                           context_weight=context_weight, verbose=verbose)
        return cls(eot)

    @classmethod
    def from_file(cls, path: str, context_weight: float = 0.3) -> "HFCompatTokenizer":
        """Load a pre-trained model."""
        eot = EOTTokenizer.from_file(path, context_weight=context_weight)
        return cls(eot)

    # ── Core API (matches tokenizers.Tokenizer) ──

    def encode(
        self,
        sequence: Union[str, bytes],
        pair: Optional[Union[str, bytes]] = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> Encoding:
        """
        Encode a string into an Encoding object.

        Compatible with ``tokenizers.Tokenizer.encode()``.
        ``pair``, ``is_pretokenized``, and ``add_special_tokens`` are
        accepted for API compatibility but ignored (EOT is byte-level
        and has no special tokens).
        """
        ids, offsets = self._eot.encode_with_offsets(sequence)
        tokens = [self._eot.id_to_token(tid) for tid in ids]
        return Encoding(
            ids=list(ids),
            tokens=tokens,
            offsets=offsets,
            type_ids=[0] * len(ids),
            attention_mask=[1] * len(ids),
            special_tokens_mask=[0] * len(ids),
        )

    def encode_batch(
        self,
        sequences: List[Union[str, bytes]],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> List[Encoding]:
        """Encode a batch of strings."""
        return [self.encode(seq) for seq in sequences]

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to a string."""
        return self._eot.decode(ids)

    def decode_batch(
        self,
        sequences: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode a batch of token ID lists."""
        return [self.decode(ids) for ids in sequences]

    def token_to_id(self, token: str) -> Optional[int]:
        """Convert a token string to its ID. Returns None if not found."""
        for tid in range(self._eot.vocab_size):
            if self._eot.id_to_token(tid) == token:
                return tid
        return None

    def id_to_token(self, id: int) -> Optional[str]:
        """Convert a token ID to its string representation."""
        try:
            return self._eot.id_to_token(id)
        except (ValueError, IndexError):
            return None

    def save(self, path: str) -> None:
        """Save the model to a JSON file."""
        self._eot.save(path)

    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self._eot.vocab_size

    def get_vocab(self) -> dict:
        """Get the full vocabulary as {token_string: id}."""
        vocab = {}
        for tid in range(self._eot.vocab_size):
            try:
                token_str = self._eot.id_to_token(tid)
                vocab[token_str] = tid
            except (ValueError, IndexError):
                pass
        return vocab

    @property
    def vocab_size(self) -> int:
        return self._eot.vocab_size

    @property
    def is_lossless(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"HFCompatTokenizer(vocab_size={self._eot.vocab_size}, lossless=True)"
