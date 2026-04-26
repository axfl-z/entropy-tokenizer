/// A trie node for fast prefix/token lookup during encoding.
/// Uses a fixed-size array for O(1) child lookup (256 possible byte values).
#[derive(Debug, Clone)]
struct TrieNode {
    children: [u32; 256], // 0 = no child, otherwise node index + 1
    token_id: u32,        // 0 = no token, otherwise token_id + 1
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: [0u32; 256],
            token_id: 0,
        }
    }
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
            nodes: vec![TrieNode::new()],
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
        let mut node_idx: usize = 0;
        for &b in bytes {
            let child = self.nodes[node_idx].children[b as usize];
            if child != 0 {
                node_idx = (child - 1) as usize;
            } else {
                let new_idx = self.nodes.len();
                self.nodes.push(TrieNode::new());
                self.nodes[node_idx].children[b as usize] = (new_idx + 1) as u32;
                node_idx = new_idx;
            }
        }
        self.nodes[node_idx].token_id = token_id + 1;
    }

    /// Find all tokens that match a prefix of input[start..].
    /// Returns Vec<(token_id, length)> sorted by length ascending.
    pub fn find_matches(&self, input: &[u8], start: usize) -> Vec<(u32, usize)> {
        let mut matches = Vec::new();
        let mut node_idx: usize = 0;

        for (i, &b) in input[start..].iter().enumerate() {
            let child = self.nodes[node_idx].children[b as usize];
            if child == 0 {
                break;
            }
            node_idx = (child - 1) as usize;
            let tid = self.nodes[node_idx].token_id;
            if tid != 0 {
                matches.push((tid - 1, i + 1));
            }
        }

        matches
    }

    /// Walk the trie one byte at a time. Returns (next_node_plus_1, token_id_plus_1).
    /// node_plus_1=0 means root. Returns 0 for next if no child exists.
    #[inline(always)]
    pub fn step(&self, node_plus_1: u32, byte: u8) -> (u32, u32) {
        let node_idx = if node_plus_1 == 0 {
            0
        } else {
            (node_plus_1 - 1) as usize
        };
        let child = self.nodes[node_idx].children[byte as usize];
        if child == 0 {
            (0, 0)
        } else {
            let n = &self.nodes[(child - 1) as usize];
            (child, n.token_id)
        }
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

    #[test]
    fn test_trie_step() {
        let tokens: Vec<(u32, &[u8])> = vec![(0, b"h"), (1, b"he")];
        let trie = Trie::from_vocab(&tokens);

        let (n1, t1) = trie.step(0, b'h');
        assert_ne!(n1, 0);
        assert_eq!(t1, 0 + 1);

        let (n2, t2) = trie.step(n1, b'e');
        assert_ne!(n2, 0);
        assert_eq!(t2, 1 + 1);
    }
}
