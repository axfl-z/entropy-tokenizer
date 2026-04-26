use std::collections::HashMap;

use crate::stats;
use crate::vocab::Vocabulary;

/// Represents a document as a sequence of token IDs for training
#[derive(Debug, Clone)]
struct TokenizedDoc {
    tokens: Vec<u32>,
}

/// Trainer that builds vocabulary using entropy-guided merge selection
pub struct Trainer {
    pub vocab: Vocabulary,
    docs: Vec<TokenizedDoc>,
    pair_counts: HashMap<(u32, u32), u64>,
    token_counts: HashMap<u32, u64>,
    bigram_counts: HashMap<(u32, u32), u64>,
}

impl Trainer {
    /// Create a new trainer from raw byte corpus
    pub fn new(corpus: &[u8]) -> Self {
        let vocab = Vocabulary::new_byte_level();

        // Split corpus into documents (by newline) for efficiency
        let docs: Vec<TokenizedDoc> = corpus
            .split(|&b| b == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| TokenizedDoc {
                tokens: line.iter().map(|&b| b as u32).collect(),
            })
            .collect();

        let mut trainer = Trainer {
            vocab,
            docs,
            pair_counts: HashMap::new(),
            token_counts: HashMap::new(),
            bigram_counts: HashMap::new(),
        };
        trainer.recount_all();
        trainer
    }

    /// Recount all pair and token frequencies from scratch
    fn recount_all(&mut self) {
        self.pair_counts.clear();
        self.token_counts.clear();
        self.bigram_counts.clear();

        for doc in &self.docs {
            for (i, &tid) in doc.tokens.iter().enumerate() {
                *self.token_counts.entry(tid).or_insert(0) += 1;
                if i + 1 < doc.tokens.len() {
                    let pair = (tid, doc.tokens[i + 1]);
                    *self.pair_counts.entry(pair).or_insert(0) += 1;
                    *self.bigram_counts.entry(pair).or_insert(0) += 1;
                }
            }
        }
    }

    /// Compute the entropy-guided merge score for a candidate pair.
    /// score = compression_gain * (1 + alpha * entropy_bonus)
    /// where compression_gain = pair_count (tokens saved)
    ///       entropy_bonus = normalized entropy improvement from this merge
    fn merge_score(
        &self,
        left_id: u32,
        right_id: u32,
        pair_count: u64,
        next_id: u32,
    ) -> f64 {
        let compression_gain = pair_count as f64;

        let entropy_delta = stats::entropy_delta_for_merge(
            &self.token_counts,
            left_id,
            right_id,
            next_id,
            pair_count,
        );

        // Alpha controls how much we weigh entropy improvement vs pure compression.
        // Higher alpha = more emphasis on creating uniform token distributions.
        let alpha = 0.5;

        // The score balances frequency (compression) with entropy improvement.
        // entropy_delta can be negative (merge reduces entropy) or positive (increases it).
        // We want merges that compress well AND improve entropy uniformity.
        compression_gain * (1.0 + alpha * entropy_delta)
    }

    /// Find the best merge candidate using entropy-guided scoring
    fn find_best_merge(&self) -> Option<((u32, u32), u64, f64)> {
        let next_id = self.vocab.tokens.len() as u32;

        let mut best: Option<((u32, u32), u64, f64)> = None;

        for (&(left, right), &count) in &self.pair_counts {
            if count < 2 {
                continue;
            }
            let score = self.merge_score(left, right, count, next_id);
            if let Some((_, _, best_score)) = &best {
                if score > *best_score {
                    best = Some(((left, right), count, score));
                }
            } else {
                best = Some(((left, right), count, score));
            }
        }

        best
    }

    /// Apply a merge: replace all occurrences of (left, right) with new_id in all docs
    fn apply_merge(&mut self, left_id: u32, right_id: u32, new_id: u32) {
        for doc in &mut self.docs {
            let mut new_tokens = Vec::with_capacity(doc.tokens.len());
            let mut i = 0;
            while i < doc.tokens.len() {
                if i + 1 < doc.tokens.len()
                    && doc.tokens[i] == left_id
                    && doc.tokens[i + 1] == right_id
                {
                    new_tokens.push(new_id);
                    i += 2;
                } else {
                    new_tokens.push(doc.tokens[i]);
                    i += 1;
                }
            }
            doc.tokens = new_tokens;
        }
    }

    /// Train the tokenizer to the target vocabulary size
    /// Consume the trainer and return the trained vocabulary
    pub fn into_vocab(self) -> Vocabulary {
        self.vocab
    }

    pub fn train(&mut self, target_vocab_size: usize, verbose: bool) {
        let start_size = self.vocab.vocab_size;
        let num_merges = target_vocab_size.saturating_sub(start_size);

        if verbose {
            let total_tokens: u64 = self.token_counts.values().sum();
            let entropy = stats::entropy_from_map(&self.token_counts);
            eprintln!(
                "Starting training: {} base tokens, target {} merges",
                start_size, num_merges
            );
            eprintln!(
                "Initial: {} total tokens, entropy = {:.4} bits",
                total_tokens, entropy
            );
        }

        for step in 0..num_merges {
            let best = self.find_best_merge();
            let ((left_id, right_id), pair_count, score) = match best {
                Some(b) => b,
                None => {
                    if verbose {
                        eprintln!("No more merges available at step {}", step);
                    }
                    break;
                }
            };

            let new_id = self.vocab.add_merge(left_id, right_id);
            self.apply_merge(left_id, right_id, new_id);
            self.recount_all();

            if verbose && (step + 1) % 100 == 0 {
                let total_tokens: u64 = self.token_counts.values().sum();
                let entropy = stats::entropy_from_map(&self.token_counts);
                let left_bytes = self.vocab.get_bytes(left_id);
                let right_bytes = self.vocab.get_bytes(right_id);
                let merged = self.vocab.get_bytes(new_id);
                eprintln!(
                    "Step {}/{}: merged {:?} + {:?} -> {:?} (count={}, score={:.2}, tokens={}, H={:.4})",
                    step + 1,
                    num_merges,
                    String::from_utf8_lossy(left_bytes),
                    String::from_utf8_lossy(right_bytes),
                    String::from_utf8_lossy(merged),
                    pair_count,
                    score,
                    total_tokens,
                    entropy,
                );
            }
        }

        // Compute final token and bigram probabilities
        self.recount_all();
        self.vocab.update_log_probs(&self.token_counts);
        self.vocab.update_bigram_log_probs(&self.bigram_counts);

        if verbose {
            let total_tokens: u64 = self.token_counts.values().sum();
            let entropy = stats::entropy_from_map(&self.token_counts);
            let active_tokens = self.token_counts.len();
            eprintln!(
                "Training complete: vocab_size={}, active_tokens={}, total_tokens={}, entropy={:.4} bits",
                self.vocab.vocab_size, active_tokens, total_tokens, entropy
            );
        }

    }

    /// Get current corpus statistics
    pub fn corpus_stats(&self) -> (u64, f64, f64) {
        let total_tokens: u64 = self.token_counts.values().sum();
        let entropy = stats::entropy_from_map(&self.token_counts);
        let bigram_h = stats::bigram_entropy(&self.bigram_counts);
        (total_tokens, entropy, bigram_h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_basic() {
        let corpus = b"aaabbbaaabbb";
        let mut trainer = Trainer::new(corpus);
        assert_eq!(trainer.vocab.vocab_size, 256);

        trainer.train(258, false);
        assert!(trainer.vocab.vocab_size >= 257);
    }

    #[test]
    fn test_trainer_merge_reduces_tokens() {
        let corpus = b"abababababababababab";
        let mut trainer = Trainer::new(corpus);
        let (tokens_before, _, _) = trainer.corpus_stats();

        trainer.train(257, false);
        let (tokens_after, _, _) = trainer.corpus_stats();

        assert!(tokens_after < tokens_before);
    }

    #[test]
    fn test_trainer_entropy_tracked() {
        let corpus = b"the quick brown fox jumps over the lazy dog";
        let mut trainer = Trainer::new(corpus);
        let (_, entropy_before, _) = trainer.corpus_stats();
        assert!(entropy_before > 0.0);

        trainer.train(270, false);
        let (_, entropy_after, _) = trainer.corpus_stats();
        assert!(entropy_after > 0.0);
    }
}
