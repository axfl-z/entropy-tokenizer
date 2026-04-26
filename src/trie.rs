use std::collections::HashMap;

/// A trie node for fast prefix/token lookup during encoding
#[derive(Debug, Clone)]
struct TrieNode {
    children: HashMap<u8, usize>,
    token_id: Option<u32>,
}

/// Byte-level trie for vocabulary lookup.
/// Supports finding all tokens that match a prefix of the input at any position.
#[derive(Debug, Clone)]
pub struct Trie {
    nodes: Vec<TrieNode>,
}

impl Default for Trie {
    fn default() -> Self {
        Self::new()
    }
}

impl Trie {
    pub fn new() -> Self {
        Trie {
            nodes: vec![TrieNode {
                children: HashMap::new(),
                token_id: None,
            }],
        }
    }

    /// Build a trie from vocabulary (token_id -> bytes mapping)
    pub fn from_vocab(tokens: &[(u32, &[u8])]) -> Self {
        let mut trie = Self::new();
        for &(id, bytes) in tokens {
            trie.insert(bytes, id);
        }
        trie
    }

    /// Insert a byte sequence with its token ID
    pub fn insert(&mut self, bytes: &[u8], token_id: u32) {
        let mut node_idx = 0;
        for &b in bytes {
            let next_idx = if let Some(&child_idx) = self.nodes[node_idx].children.get(&b) {
                child_idx
            } else {
                let new_idx = self.nodes.len();
                self.nodes.push(TrieNode {
                    children: HashMap::new(),
                    token_id: None,
                });
                self.nodes[node_idx].children.insert(b, new_idx);
                new_idx
            };
            node_idx = next_idx;
        }
        self.nodes[node_idx].token_id = Some(token_id);
    }

    /// Find all tokens that match a prefix of input[start..].
    /// Returns Vec<(token_id, length)> sorted by length ascending.
    pub fn find_matches(&self, input: &[u8], start: usize) -> Vec<(u32, usize)> {
        let mut matches = Vec::new();
        let mut node_idx = 0;

        for (i, &b) in input[start..].iter().enumerate() {
            if let Some(&child_idx) = self.nodes[node_idx].children.get(&b) {
                node_idx = child_idx;
                if let Some(token_id) = self.nodes[node_idx].token_id {
                    matches.push((token_id, i + 1));
                }
            } else {
                break;
            }
        }

        matches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trie_insert_and_find() {
        let tokens: Vec<(u32, &[u8])> = vec![
            (0, b"h"),
            (1, b"he"),
            (2, b"hel"),
            (3, b"hello"),
            (4, b"e"),
        ];
        let trie = Trie::from_vocab(&tokens);

        let input = b"hello world";
        let matches = trie.find_matches(input, 0);
        assert_eq!(matches, vec![(0, 1), (1, 2), (2, 3), (3, 5)]);

        let matches = trie.find_matches(input, 1);
        assert_eq!(matches, vec![(4, 1)]);
    }

    #[test]
    fn test_trie_no_match() {
        let tokens: Vec<(u32, &[u8])> = vec![(0, b"x")];
        let trie = Trie::from_vocab(&tokens);
        let matches = trie.find_matches(b"abc", 0);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_trie_single_byte() {
        let mut trie = Trie::new();
        for b in 0u16..256u16 {
            trie.insert(&[b as u8], b as u32);
        }
        let matches = trie.find_matches(b"A", 0);
        assert_eq!(matches, vec![(65, 1)]);
    }
}
