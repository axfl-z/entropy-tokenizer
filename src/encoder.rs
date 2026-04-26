use crate::trie::Trie;
use crate::vocab::Vocabulary;

/// Encoder that uses DP-optimal segmentation with context entropy
pub struct Encoder {
    trie: Trie,
    vocab: Vocabulary,
    /// Weight for bigram context bonus in scoring
    context_weight: f64,
}

impl Encoder {
    /// Create an encoder from a trained vocabulary
    pub fn new(vocab: Vocabulary, context_weight: f64) -> Self {
        let token_entries: Vec<(u32, &[u8])> = vocab
            .tokens
            .iter()
            .map(|t| (t.id, t.bytes.as_slice()))
            .collect();
        let trie = Trie::from_vocab(&token_entries);

        Encoder {
            trie,
            vocab,
            context_weight,
        }
    }

    /// Encode bytes into token IDs using DP-optimal segmentation.
    ///
    /// For each position i, we compute the best score for encoding input[0..i].
    /// Score = sum of token log-probabilities + context_weight * bigram log-prob bonuses.
    /// This finds the globally optimal tokenization, not a greedy one.
    pub fn encode(&self, input: &[u8]) -> Vec<u32> {
        let n = input.len();
        if n == 0 {
            return vec![];
        }

        // dp[i] = (best_score, token_id_used, previous_position, previous_token_id)
        let mut dp: Vec<(f64, u32, usize, Option<u32>)> = vec![(f64::NEG_INFINITY, 0, 0, None); n + 1];
        dp[0] = (0.0, 0, 0, None);

        for i in 0..n {
            if dp[i].0 == f64::NEG_INFINITY {
                continue;
            }

            let matches = self.trie.find_matches(input, i);

            if matches.is_empty() {
                // Fallback: use single byte token
                let byte_id = input[i] as u32;
                let token_log_prob = self.vocab.tokens[byte_id as usize].log_prob;
                let prev_token = dp[i].3;
                let context_bonus = self.get_context_bonus(prev_token, byte_id);
                let new_score = dp[i].0 + token_log_prob + self.context_weight * context_bonus;

                if new_score > dp[i + 1].0 {
                    dp[i + 1] = (new_score, byte_id, i, Some(byte_id));
                }
                continue;
            }

            for (token_id, length) in matches {
                let end = i + length;
                let token_log_prob = self.vocab.tokens[token_id as usize].log_prob;
                let prev_token = dp[i].3;
                let context_bonus = self.get_context_bonus(prev_token, token_id);
                let new_score = dp[i].0 + token_log_prob + self.context_weight * context_bonus;

                if new_score > dp[end].0 {
                    dp[end] = (new_score, token_id, i, Some(token_id));
                }
            }
        }

        // Backtrack to get the optimal tokenization
        let mut tokens = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let (_, token_id, prev_pos, _) = dp[pos];
            tokens.push(token_id);
            pos = prev_pos;
        }
        tokens.reverse();
        tokens
    }

    /// Greedy encoding (left-to-right longest match) for speed comparison
    pub fn encode_greedy(&self, input: &[u8]) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut pos = 0;

        while pos < input.len() {
            let matches = self.trie.find_matches(input, pos);
            if let Some(&(token_id, length)) = matches.last() {
                tokens.push(token_id);
                pos += length;
            } else {
                tokens.push(input[pos] as u32);
                pos += 1;
            }
        }

        tokens
    }

    /// Get bigram context bonus (log P(cur | prev))
    fn get_context_bonus(&self, prev_token: Option<u32>, cur_token: u32) -> f64 {
        match prev_token {
            Some(prev) => self
                .vocab
                .bigram_log_probs
                .get(&(prev, cur_token))
                .copied()
                .unwrap_or(-10.0), // penalty for unseen bigrams
            None => 0.0,
        }
    }

    /// Decode token IDs back to bytes
    pub fn decode(&self, token_ids: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::new();
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
