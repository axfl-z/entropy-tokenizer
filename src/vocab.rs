use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single token in the vocabulary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub id: u32,
    pub bytes: Vec<u8>,
    pub log_prob: f64,
    /// If this token was created by merging two tokens, store their IDs
    pub merge_pair: Option<(u32, u32)>,
}

/// JSON-friendly bigram entry for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BigramEntry {
    left: u32,
    right: u32,
    log_prob: f64,
}

/// JSON-friendly representation for save/load
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VocabFile {
    tokens: Vec<Token>,
    bigrams: Vec<BigramEntry>,
    vocab_size: usize,
}

/// The full vocabulary including merge history and token statistics
#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub tokens: Vec<Token>,
    pub byte_to_id: HashMap<u8, u32>,
    pub bytes_to_id: HashMap<Vec<u8>, u32>,
    /// Bigram log-probabilities for context-aware encoding: (prev_id, cur_id) -> log_prob
    pub bigram_log_probs: HashMap<(u32, u32), f64>,
    pub vocab_size: usize,
}

impl Vocabulary {
    /// Create a new vocabulary with byte-level base tokens (256 tokens)
    pub fn new_byte_level() -> Self {
        let mut tokens = Vec::with_capacity(256);
        let mut byte_to_id = HashMap::new();
        let mut bytes_to_id = HashMap::new();

        for b in 0u16..256u16 {
            let byte = b as u8;
            let id = b as u32;
            tokens.push(Token {
                id,
                bytes: vec![byte],
                log_prob: 0.0,
                merge_pair: None,
            });
            byte_to_id.insert(byte, id);
            bytes_to_id.insert(vec![byte], id);
        }

        Vocabulary {
            tokens,
            byte_to_id,
            bytes_to_id,
            bigram_log_probs: HashMap::new(),
            vocab_size: 256,
        }
    }

    /// Add a new token formed by merging left_id and right_id
    pub fn add_merge(&mut self, left_id: u32, right_id: u32) -> u32 {
        let new_id = self.tokens.len() as u32;
        let mut new_bytes = self.tokens[left_id as usize].bytes.clone();
        new_bytes.extend_from_slice(&self.tokens[right_id as usize].bytes);

        self.bytes_to_id.insert(new_bytes.clone(), new_id);
        self.tokens.push(Token {
            id: new_id,
            bytes: new_bytes,
            log_prob: 0.0,
            merge_pair: Some((left_id, right_id)),
        });
        self.vocab_size += 1;
        new_id
    }

    /// Look up token ID by byte sequence
    pub fn lookup(&self, bytes: &[u8]) -> Option<u32> {
        self.bytes_to_id.get(bytes).copied()
    }

    /// Get bytes for a token ID
    pub fn get_bytes(&self, id: u32) -> &[u8] {
        &self.tokens[id as usize].bytes
    }

    /// Get the maximum token length in bytes
    pub fn max_token_len(&self) -> usize {
        self.tokens.iter().map(|t| t.bytes.len()).max().unwrap_or(1)
    }

    /// Update token log-probabilities from corpus frequency counts
    pub fn update_log_probs(&mut self, token_counts: &HashMap<u32, u64>) {
        let total: u64 = token_counts.values().sum();
        if total == 0 {
            return;
        }
        let total_f = total as f64;
        for token in &mut self.tokens {
            let count = token_counts.get(&token.id).copied().unwrap_or(0);
            if count > 0 {
                token.log_prob = (count as f64 / total_f).ln();
            } else {
                // Smoothing: assign very low probability
                token.log_prob = (0.5 / total_f).ln();
            }
        }
    }

    /// Update bigram log-probabilities from bigram counts
    pub fn update_bigram_log_probs(&mut self, bigram_counts: &HashMap<(u32, u32), u64>) {
        let mut left_totals: HashMap<u32, u64> = HashMap::new();
        for (&(left, _), &count) in bigram_counts {
            *left_totals.entry(left).or_insert(0) += count;
        }

        self.bigram_log_probs.clear();
        for (&(left, right), &count) in bigram_counts {
            if count > 0 {
                let left_total = left_totals[&left];
                let log_prob = (count as f64 / left_total as f64).ln();
                self.bigram_log_probs.insert((left, right), log_prob);
            }
        }
    }

    /// Save vocabulary to JSON file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let bigrams: Vec<BigramEntry> = self
            .bigram_log_probs
            .iter()
            .map(|(&(left, right), &log_prob)| BigramEntry {
                left,
                right,
                log_prob,
            })
            .collect();

        let file = VocabFile {
            tokens: self.tokens.clone(),
            bigrams,
            vocab_size: self.vocab_size,
        };

        let json = serde_json::to_string_pretty(&file)
            .map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Load vocabulary from JSON file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let file: VocabFile =
            serde_json::from_str(&json).map_err(std::io::Error::other)?;

        let mut byte_to_id = HashMap::new();
        let mut bytes_to_id = HashMap::new();
        for token in &file.tokens {
            if token.bytes.len() == 1 {
                byte_to_id.insert(token.bytes[0], token.id);
            }
            bytes_to_id.insert(token.bytes.clone(), token.id);
        }

        let mut bigram_log_probs = HashMap::new();
        for entry in &file.bigrams {
            bigram_log_probs.insert((entry.left, entry.right), entry.log_prob);
        }

        Ok(Vocabulary {
            tokens: file.tokens,
            byte_to_id,
            bytes_to_id,
            bigram_log_probs,
            vocab_size: file.vocab_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_level_vocab() {
        let vocab = Vocabulary::new_byte_level();
        assert_eq!(vocab.vocab_size, 256);
        assert_eq!(vocab.tokens.len(), 256);
        assert_eq!(vocab.lookup(&[b'A']), Some(65));
        assert_eq!(vocab.get_bytes(65), &[b'A']);
    }

    #[test]
    fn test_add_merge() {
        let mut vocab = Vocabulary::new_byte_level();
        let new_id = vocab.add_merge(b'h' as u32, b'e' as u32);
        assert_eq!(new_id, 256);
        assert_eq!(vocab.get_bytes(new_id), b"he");
        assert_eq!(vocab.lookup(b"he"), Some(256));
        assert_eq!(vocab.vocab_size, 257);
    }

    #[test]
    fn test_max_token_len() {
        let mut vocab = Vocabulary::new_byte_level();
        assert_eq!(vocab.max_token_len(), 1);
        vocab.add_merge(b'a' as u32, b'b' as u32);
        assert_eq!(vocab.max_token_len(), 2);
    }
}
