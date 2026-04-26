use std::collections::HashMap;

/// Compute Shannon entropy of a frequency distribution: H = -sum(p * log2(p))
pub fn shannon_entropy(counts: &[u64]) -> f64 {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    let mut entropy = 0.0;
    for &c in counts {
        if c > 0 {
            let p = c as f64 / total_f;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Compute entropy from a HashMap of counts
pub fn entropy_from_map(freq: &HashMap<u32, u64>) -> f64 {
    let counts: Vec<u64> = freq.values().copied().collect();
    shannon_entropy(&counts)
}

/// Compute normalized entropy (0..1 range) given the vocabulary size
pub fn normalized_entropy(counts: &[u64], vocab_size: usize) -> f64 {
    if vocab_size <= 1 {
        return 0.0;
    }
    let h = shannon_entropy(counts);
    let max_h = (vocab_size as f64).log2();
    if max_h == 0.0 {
        return 0.0;
    }
    h / max_h
}

/// Compute the entropy delta if we merge two tokens (pair) into one new token.
/// Returns the change in entropy (positive = entropy increased = more uniform).
pub fn entropy_delta_for_merge(
    token_counts: &HashMap<u32, u64>,
    left_id: u32,
    right_id: u32,
    new_id: u32,
    pair_count: u64,
) -> f64 {
    let old_entropy = entropy_from_map(token_counts);

    let mut new_counts = token_counts.clone();

    let left_count = new_counts.get(&left_id).copied().unwrap_or(0);
    let right_count = new_counts.get(&right_id).copied().unwrap_or(0);

    if left_count >= pair_count {
        let new_left = left_count - pair_count;
        if new_left == 0 {
            new_counts.remove(&left_id);
        } else {
            new_counts.insert(left_id, new_left);
        }
    }
    if right_count >= pair_count {
        let new_right = right_count - pair_count;
        if new_right == 0 {
            new_counts.remove(&right_id);
        } else {
            new_counts.insert(right_id, new_right);
        }
    }

    new_counts.insert(new_id, pair_count);

    let new_entropy = entropy_from_map(&new_counts);
    new_entropy - old_entropy
}

/// Compute bigram entropy: H(T_{i+1} | T_i) — context entropy
pub fn bigram_entropy(bigram_counts: &HashMap<(u32, u32), u64>) -> f64 {
    let mut left_totals: HashMap<u32, u64> = HashMap::new();
    for (&(left, _), &count) in bigram_counts {
        *left_totals.entry(left).or_insert(0) += count;
    }

    let grand_total: u64 = bigram_counts.values().sum();
    if grand_total == 0 {
        return 0.0;
    }

    let mut h = 0.0;
    for (&(left, _right), &count) in bigram_counts {
        if count == 0 {
            continue;
        }
        let left_total = left_totals[&left];
        let p_bigram = count as f64 / grand_total as f64;
        let p_cond = count as f64 / left_total as f64;
        h -= p_bigram * p_cond.log2();
    }
    h
}

/// Compression ratio: original_bytes / num_tokens
pub fn compression_ratio(original_len: usize, num_tokens: usize) -> f64 {
    if num_tokens == 0 {
        return 0.0;
    }
    original_len as f64 / num_tokens as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_uniform() {
        // Uniform distribution over 4 items => entropy = 2.0 bits
        let counts = vec![100, 100, 100, 100];
        let h = shannon_entropy(&counts);
        assert!((h - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_single() {
        let counts = vec![100];
        let h = shannon_entropy(&counts);
        assert!((h - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_skewed() {
        let counts = vec![99, 1];
        let h = shannon_entropy(&counts);
        assert!(h > 0.0 && h < 1.0);
    }

    #[test]
    fn test_normalized_entropy() {
        let counts = vec![100, 100, 100, 100];
        let nh = normalized_entropy(&counts, 4);
        assert!((nh - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compression_ratio_basic() {
        assert!((compression_ratio(100, 25) - 4.0).abs() < 1e-10);
    }
}
