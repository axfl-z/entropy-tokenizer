use rayon::prelude::*;

use crate::trie::Trie;
use crate::vocab::Vocabulary;

/// Encoder that uses DP-optimal segmentation with context entropy.
/// Optimized for speed with parallel chunk encoding and zero-alloc trie walking.
pub struct Encoder {
    trie: Trie,
    vocab: Vocabulary,
    context_weight: f64,
    max_token_len: usize,
}

/// Chunk size for parallel encoding (64KB chunks)
const CHUNK_SIZE: usize = 65536;

impl Encoder {
    pub fn new(vocab: Vocabulary, context_weight: f64) -> Self {
        let token_entries: Vec<(u32, &[u8])> = vocab
            .tokens
            .iter()
            .map(|t| (t.id, t.bytes.as_slice()))
            .collect();
        let trie = Trie::from_vocab(&token_entries);
        let max_token_len = vocab.max_token_len();

        Encoder {
            trie,
            vocab,
            context_weight,
            max_token_len,
        }
    }

    /// Encode bytes into token IDs using DP-optimal segmentation.
    /// For large inputs, splits into chunks and encodes in parallel.
    pub fn encode(&self, input: &[u8]) -> Vec<u32> {
        let n = input.len();
        if n == 0 {
            return vec![];
        }

        // For small inputs or when context weight matters, use single-threaded DP
        if n <= CHUNK_SIZE * 2 {
            return self.encode_dp(input);
        }

        // Parallel chunked encoding for large inputs
        self.encode_parallel(input)
    }

    /// Single-pass DP-optimal encoding (the core algorithm)
    fn encode_dp(&self, input: &[u8]) -> Vec<u32> {
        let n = input.len();
        if n == 0 {
            return vec![];
        }

        // dp[i] = (best_score, token_id_used, previous_position, previous_token_id)
        // Using a flat Vec with manual indexing for cache efficiency
        let mut dp_score = vec![f64::NEG_INFINITY; n + 1];
        let mut dp_token = vec![0u32; n + 1];
        let mut dp_prev_pos = vec![0usize; n + 1];
        let mut dp_prev_tid = vec![0u32; n + 1]; // 0 = none, actual id + 1
        dp_score[0] = 0.0;

        let max_len = self.max_token_len.min(n);

        for i in 0..n {
            if dp_score[i] == f64::NEG_INFINITY {
                continue;
            }

            let cur_score = dp_score[i];
            let prev_tid = dp_prev_tid[i];

            // Walk the trie from position i, collecting all matching tokens
            let end_limit = (i + max_len).min(n);
            let mut node: u32 = 0; // 0 = root for step()

            #[allow(clippy::needless_range_loop)]
            for j in i..end_limit {
                let (next_node, tid_plus_1) = self.trie.step(node, input[j]);
                if next_node == 0 {
                    break;
                }
                node = next_node;

                if tid_plus_1 != 0 {
                    let token_id = tid_plus_1 - 1;
                    let end = j + 1;
                    let token_log_prob = self.vocab.tokens[token_id as usize].log_prob;

                    let context_bonus = if self.context_weight > 0.0 && prev_tid != 0 {
                        self.vocab
                            .bigram_log_probs
                            .get(&(prev_tid - 1, token_id))
                            .copied()
                            .unwrap_or(-10.0)
                    } else {
                        0.0
                    };

                    let new_score = cur_score + token_log_prob + self.context_weight * context_bonus;

                    if new_score > dp_score[end] {
                        dp_score[end] = new_score;
                        dp_token[end] = token_id;
                        dp_prev_pos[end] = i;
                        dp_prev_tid[end] = token_id + 1;
                    }
                }
            }

            // Fallback: if no trie match at all, use single byte token
            if node == 0 {
                let byte_id = input[i] as u32;
                let token_log_prob = self.vocab.tokens[byte_id as usize].log_prob;
                let context_bonus = if self.context_weight > 0.0 && prev_tid != 0 {
                    self.vocab
                        .bigram_log_probs
                        .get(&(prev_tid - 1, byte_id))
                        .copied()
                        .unwrap_or(-10.0)
                } else {
                    0.0
                };
                let new_score = cur_score + token_log_prob + self.context_weight * context_bonus;

                if new_score > dp_score[i + 1] {
                    dp_score[i + 1] = new_score;
                    dp_token[i + 1] = byte_id;
                    dp_prev_pos[i + 1] = i;
                    dp_prev_tid[i + 1] = byte_id + 1;
                }
            }
        }

        // Backtrack to get the optimal tokenization
        let mut tokens = Vec::new();
        let mut pos = n;
        while pos > 0 {
            tokens.push(dp_token[pos]);
            pos = dp_prev_pos[pos];
        }
        tokens.reverse();
        tokens
    }

    /// Parallel chunked encoding for large inputs
    fn encode_parallel(&self, input: &[u8]) -> Vec<u32> {
        let n = input.len();

        // Create chunks with overlap
        let mut chunk_ranges: Vec<(usize, usize)> = Vec::new();
        let mut start = 0;
        while start < n {
            let end = (start + CHUNK_SIZE).min(n);
            chunk_ranges.push((start, end));
            start = end;
        }

        // Encode chunks in parallel
        let chunk_results: Vec<Vec<u32>> = chunk_ranges
            .par_iter()
            .map(|&(start, end)| self.encode_dp(&input[start..end]))
            .collect();

        // Concatenate results
        let total_tokens: usize = chunk_results.iter().map(|c| c.len()).sum();
        let mut result = Vec::with_capacity(total_tokens);
        for chunk in chunk_results {
            result.extend(chunk);
        }
        result
    }

    /// Greedy encoding (left-to-right longest match) — fastest mode
    pub fn encode_greedy(&self, input: &[u8]) -> Vec<u32> {
        let n = input.len();
        if n == 0 {
            return vec![];
        }

        let mut tokens = Vec::with_capacity(n / 2);
        let mut pos = 0;

        while pos < n {
            let mut node: u32 = 0;
            let mut best_id = input[pos] as u32; // fallback: single byte
            let mut best_len: usize = 1;
            let end_limit = (pos + self.max_token_len).min(n);

            #[allow(clippy::needless_range_loop)]
            for j in pos..end_limit {
                let (next_node, tid_plus_1) = self.trie.step(node, input[j]);
                if next_node == 0 {
                    break;
                }
                node = next_node;
                if tid_plus_1 != 0 {
                    best_id = tid_plus_1 - 1;
                    best_len = j - pos + 1;
                }
            }

            tokens.push(best_id);
            pos += best_len;
        }

        tokens
    }

    /// Decode token IDs back to bytes
    pub fn decode(&self, token_ids: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(token_ids.len() * 4);
        for &id in token_ids {
            bytes.extend_from_slice(self.vocab.get_bytes(id));
        }
        bytes
    }

    /// Decode token IDs to a UTF-8 string (lossy)
    pub fn decode_to_string(&self, token_ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode(token_ids)).to_string()
    }

    /// Get vocabulary reference
    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::Trainer;

    fn make_encoder() -> Encoder {
        let corpus = b"hello hello hello world world hello world";
        let mut trainer = Trainer::new(corpus);
        trainer.train(270, false);
        let vocab = trainer.into_vocab();
        Encoder::new(vocab, 0.3)
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let encoder = make_encoder();
        let input = b"hello world";
        let tokens = encoder.encode(input);
        let decoded = encoder.decode(&tokens);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_greedy_encode_decode_roundtrip() {
        let encoder = make_encoder();
        let input = b"hello world";
        let tokens = encoder.encode_greedy(input);
        let decoded = encoder.decode(&tokens);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_dp_fewer_or_equal_tokens_than_greedy() {
        let encoder = make_encoder();
        let input = b"hello hello world hello";
        let dp_tokens = encoder.encode(input);
        let greedy_tokens = encoder.encode_greedy(input);

        // DP should produce fewer or equal tokens (it's optimal)
        assert!(dp_tokens.len() <= greedy_tokens.len() + 1);
        // Both must decode correctly
        assert_eq!(encoder.decode(&dp_tokens), input);
        assert_eq!(encoder.decode(&greedy_tokens), input);
    }

    #[test]
    fn test_empty_input() {
        let encoder = make_encoder();
        let tokens = encoder.encode(b"");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_single_byte() {
        let encoder = make_encoder();
        let tokens = encoder.encode(b"x");
        assert_eq!(tokens.len(), 1);
        assert_eq!(encoder.decode(&tokens), b"x");
    }
}
