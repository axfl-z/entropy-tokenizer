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

/// A special token with a string representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialToken {
    pub name: String,
    pub id: u32,
    pub bytes: Vec<u8>,
}

/// JSON-friendly representation for save/load
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VocabFile {
    version: String,
    tokens: Vec<Token>,
    bigrams: Vec<BigramEntry>,
    vocab_size: usize,
    #[serde(default)]
    special_tokens: Vec<SpecialToken>,
}

/// The full vocabulary including merge history and token statistics
#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub tokens: Vec<Token>,
    pub byte_to_id: HashMap<u8, u32>,
    pub bytes_to_id: HashMap<Vec<u8>, u32>,
    pub bigram_log_probs: HashMap<(u32, u32), f64>,
    pub vocab_size: usize,
    pub special_tokens: Vec<SpecialToken>,
    pub special_name_to_id: HashMap<String, u32>,
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
            special_tokens: Vec::new(),
            special_name_to_id: HashMap::new(),
        }
    }

    /// Add a special token (e.g., [PAD], [UNK], [CLS], [SEP]).
    /// Returns the assigned token ID.
    pub fn add_special_token(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.special_name_to_id.get(name) {
            return id;
        }
        let id = self.tokens.len() as u32;
        let bytes = name.as_bytes().to_vec();
        self.tokens.push(Token {
            id,
            bytes: bytes.clone(),
            log_prob: 0.0,
            merge_pair: None,
        });
        self.bytes_to_id.insert(bytes.clone(), id);
        let special = SpecialToken {
            name: name.to_string(),
            id,
            bytes,
        };
        self.special_tokens.push(special);
        self.special_name_to_id.insert(name.to_string(), id);
        self.vocab_size += 1;
        id
    }

    /// Get special token ID by name
    pub fn get_special_token(&self, name: &str) -> Option<u32> {
        self.special_name_to_id.get(name).copied()
    }

    /// Check if a token ID is a special token
    pub fn is_special(&self, id: u32) -> bool {
        self.special_tokens.iter().any(|s| s.id == id)
    }

    /// Prune vocabulary to target size by keeping the most useful tokens.
    /// Tokens are ranked by frequency (from token_counts). Base byte tokens (0-255)
    /// and special tokens are always kept.
    pub fn prune(&mut self, target_size: usize, token_counts: &HashMap<u32, u64>) {
        if self.vocab_size <= target_size {
            return;
        }

        // Collect merge tokens (id >= 256) with their counts, excluding special tokens
        let special_ids: std::collections::HashSet<u32> = self.special_tokens.iter().map(|s| s.id).collect();
        let mut merge_tokens: Vec<(u32, u64)> = Vec::new();
        for token in &self.tokens {
            if token.id >= 256 && !special_ids.contains(&token.id) {
                let count = token_counts.get(&token.id).copied().unwrap_or(0);
                merge_tokens.push((token.id, count));
            }
        }

        // Sort by count descending (keep most frequent)
        merge_tokens.sort_by_key(|t| std::cmp::Reverse(t.1));

        // How many merge tokens can we keep?
        let special_count = self.special_tokens.len();
        let keep_merges = target_size.saturating_sub(256 + special_count);
        let keep_ids: std::collections::HashSet<u32> = merge_tokens.iter()
            .take(keep_merges)
            .map(|&(id, _)| id)
            .collect();

        // Rebuild vocab with only kept tokens, re-assigning IDs
        let mut new_tokens = Vec::new();
        let mut id_map: HashMap<u32, u32> = HashMap::new();
        let mut new_bytes_to_id = HashMap::new();
        let mut new_byte_to_id = HashMap::new();

        // Keep base byte tokens (0-255)
        for i in 0u32..256 {
            let old = &self.tokens[i as usize];
            let new_id = new_tokens.len() as u32;
            id_map.insert(i, new_id);
            new_tokens.push(Token {
                id: new_id,
                bytes: old.bytes.clone(),
                log_prob: old.log_prob,
                merge_pair: None,
            });
            new_byte_to_id.insert(old.bytes[0], new_id);
            new_bytes_to_id.insert(old.bytes.clone(), new_id);
        }

        // Keep selected merge tokens
        for token in &self.tokens {
            if token.id >= 256 && !special_ids.contains(&token.id) && keep_ids.contains(&token.id) {
                let new_id = new_tokens.len() as u32;
                id_map.insert(token.id, new_id);
                let merge_pair = token.merge_pair.map(|(l, r)| {
                    (*id_map.get(&l).unwrap_or(&l), *id_map.get(&r).unwrap_or(&r))
                });
                new_tokens.push(Token {
                    id: new_id,
                    bytes: token.bytes.clone(),
                    log_prob: token.log_prob,
                    merge_pair,
                });
                new_bytes_to_id.insert(token.bytes.clone(), new_id);
            }
        }

        // Re-add special tokens
        let mut new_specials = Vec::new();
        let mut new_special_name_to_id = HashMap::new();
        for special in &self.special_tokens {
            let new_id = new_tokens.len() as u32;
            id_map.insert(special.id, new_id);
            new_tokens.push(Token {
                id: new_id,
                bytes: special.bytes.clone(),
                log_prob: 0.0,
                merge_pair: None,
            });
            new_bytes_to_id.insert(special.bytes.clone(), new_id);
            new_specials.push(SpecialToken {
                name: special.name.clone(),
                id: new_id,
                bytes: special.bytes.clone(),
            });
            new_special_name_to_id.insert(special.name.clone(), new_id);
        }

        // Rebuild bigram_log_probs with new IDs
        let mut new_bigrams = HashMap::new();
        for (&(l, r), &prob) in &self.bigram_log_probs {
            if let (Some(&nl), Some(&nr)) = (id_map.get(&l), id_map.get(&r)) {
                new_bigrams.insert((nl, nr), prob);
            }
        }

        self.tokens = new_tokens;
        self.bytes_to_id = new_bytes_to_id;
        self.byte_to_id = new_byte_to_id;
        self.bigram_log_probs = new_bigrams;
        self.vocab_size = self.tokens.len();
        self.special_tokens = new_specials;
        self.special_name_to_id = new_special_name_to_id;
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
            version: "0.1.1".to_string(),
            tokens: self.tokens.clone(),
            bigrams,
            vocab_size: self.vocab_size,
            special_tokens: self.special_tokens.clone(),
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

        let mut special_name_to_id = HashMap::new();
        for st in &file.special_tokens {
            special_name_to_id.insert(st.name.clone(), st.id);
        }

        Ok(Vocabulary {
            tokens: file.tokens,
            byte_to_id,
            bytes_to_id,
            bigram_log_probs,
            vocab_size: file.vocab_size,
            special_tokens: file.special_tokens,
            special_name_to_id,
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
