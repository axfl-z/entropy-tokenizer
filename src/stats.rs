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

/// Fast analytical approximation of entropy delta for a merge.
/// Avoids cloning the entire token_counts HashMap.
/// Returns the change in entropy (positive = entropy increased = more uniform).
pub fn entropy_delta_for_merge(
    token_counts: &HashMap<u32, u64>,
    left_id: u32,
    right_id: u32,
    _new_id: u32,
    pair_count: u64,
) -> f64 {
    let total: u64 = token_counts.values().sum();
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    let pc = pair_count as f64;
    let left_count = token_counts.get(&left_id).copied().unwrap_or(0) as f64;
    let right_count = token_counts.get(&right_id).copied().unwrap_or(0) as f64;

    // The merge reduces total token count by pair_count (two tokens become one)
    let new_n = n - pc;
    if new_n <= 0.0 {
        return 0.0;
    }

    // Compute entropy contribution changes analytically
    // Old contributions from left, right tokens
    let h_contrib = |count: f64, tot: f64| -> f64 {
        if count <= 0.0 || tot <= 0.0 {
            0.0
        } else {
            let p = count / tot;
            -p * p.log2()
        }
    };

    let old_left = h_contrib(left_count, n);
    let old_right = h_contrib(right_count, n);

    let new_left_count = left_count - pc;
    let new_right_count = right_count - pc;

    let new_left = h_contrib(new_left_count, new_n);
    let new_right = h_contrib(new_right_count, new_n);
    let new_merged = h_contrib(pc, new_n);

    // Approximate: only the changed tokens matter
    (new_left + new_right + new_merged) - (old_left + old_right)
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

/// Compute trigram entropy: H(T_{i+2} | T_i, T_{i+1}) — higher-order context entropy
pub fn trigram_entropy(trigram_counts: &HashMap<(u32, u32, u32), u64>) -> f64 {
    let mut context_totals: HashMap<(u32, u32), u64> = HashMap::new();
    for (&(t0, t1, _t2), &count) in trigram_counts {
        *context_totals.entry((t0, t1)).or_insert(0) += count;
    }

    let grand_total: u64 = trigram_counts.values().sum();
    if grand_total == 0 {
        return 0.0;
    }

    let mut h = 0.0;
    for (&(t0, t1, _t2), &count) in trigram_counts {
        if count == 0 {
            continue;
        }
        let ctx_total = context_totals[&(t0, t1)];
        let p_trigram = count as f64 / grand_total as f64;
        let p_cond = count as f64 / ctx_total as f64;
        h -= p_trigram * p_cond.log2();
    }
    h
}

/// Collect trigram counts from a token sequence
pub fn collect_trigram_counts(tokens: &[u32]) -> HashMap<(u32, u32, u32), u64> {
    let mut counts: HashMap<(u32, u32, u32), u64> = HashMap::new();
    if tokens.len() < 3 {
        return counts;
    }
    for i in 0..tokens.len() - 2 {
        let tri = (tokens[i], tokens[i + 1], tokens[i + 2]);
        *counts.entry(tri).or_insert(0) += 1;
    }
    counts
}

/// Collect bigram counts from a token sequence
pub fn collect_bigram_counts(tokens: &[u32]) -> HashMap<(u32, u32), u64> {
    let mut counts: HashMap<(u32, u32), u64> = HashMap::new();
    if tokens.len() < 2 {
        return counts;
    }
    for i in 0..tokens.len() - 1 {
        let pair = (tokens[i], tokens[i + 1]);
        *counts.entry(pair).or_insert(0) += 1;
    }
    counts
}

/// Collect unigram counts from a token sequence
pub fn collect_unigram_counts(tokens: &[u32]) -> HashMap<u32, u64> {
    let mut counts: HashMap<u32, u64> = HashMap::new();
    for &t in tokens {
        *counts.entry(t).or_insert(0) += 1;
    }
    counts
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

    #[test]
    fn test_bigram_entropy_basic() {
        // Context 0 -> either 1 or 2 (ambiguous), so H2 > 0
        let mut counts: HashMap<(u32, u32), u64> = HashMap::new();
        counts.insert((0, 1), 50);
        counts.insert((0, 2), 50);
        counts.insert((1, 0), 100);
        let h = bigram_entropy(&counts);
        assert!(h > 0.0, "bigram entropy should be positive, got {}", h);
        assert!(h < 2.0, "bigram entropy should be bounded");
    }

    #[test]
    fn test_bigram_entropy_deterministic() {
        // If every context perfectly predicts the next token, H2 = 0
        let mut counts: HashMap<(u32, u32), u64> = HashMap::new();
        counts.insert((0, 1), 100);
        counts.insert((1, 0), 100);
        let h = bigram_entropy(&counts);
        assert!((h - 0.0).abs() < 1e-10, "deterministic bigrams should have H2=0");
    }

    #[test]
    fn test_trigram_entropy_basic() {
        // Context (0,1) -> either 2 or 3 (ambiguous), so H3 > 0
        let mut counts: HashMap<(u32, u32, u32), u64> = HashMap::new();
        counts.insert((0, 1, 2), 50);
        counts.insert((0, 1, 3), 50);
        counts.insert((1, 2, 0), 100);
        let h = trigram_entropy(&counts);
        assert!(h > 0.0, "trigram entropy should be positive, got {}", h);
    }

    #[test]
    fn test_trigram_entropy_deterministic() {
        let mut counts: HashMap<(u32, u32, u32), u64> = HashMap::new();
        counts.insert((0, 1, 2), 100);
        counts.insert((1, 2, 3), 100);
        let h = trigram_entropy(&counts);
        assert!((h - 0.0).abs() < 1e-10, "deterministic trigrams should have H3=0");
    }

    #[test]
    fn test_collect_unigram_counts() {
        let tokens = vec![0, 1, 0, 1, 0];
        let counts = collect_unigram_counts(&tokens);
        assert_eq!(counts[&0], 3);
        assert_eq!(counts[&1], 2);
    }

    #[test]
    fn test_collect_bigram_counts() {
        let tokens = vec![0, 1, 0, 1, 0];
        let counts = collect_bigram_counts(&tokens);
        assert_eq!(counts[&(0, 1)], 2);
        assert_eq!(counts[&(1, 0)], 2);
    }

    #[test]
    fn test_collect_trigram_counts() {
        let tokens = vec![0, 1, 2, 0, 1, 2];
        let counts = collect_trigram_counts(&tokens);
        assert_eq!(counts[&(0, 1, 2)], 2);
        assert_eq!(counts[&(1, 2, 0)], 1);
        assert_eq!(counts[&(2, 0, 1)], 1);
    }

    #[test]
    fn test_entropy_hierarchy_h1_ge_h2_ge_h3() {
        // For natural-like sequences, H1 >= H2 >= H3 (conditioning reduces entropy)
        let tokens: Vec<u32> = "hello world hello world foo bar"
            .bytes()
            .map(|b| b as u32)
            .collect();
        let uni = collect_unigram_counts(&tokens);
        let bi = collect_bigram_counts(&tokens);
        let tri = collect_trigram_counts(&tokens);
        let h1 = entropy_from_map(&uni);
        let h2 = bigram_entropy(&bi);
        let h3 = trigram_entropy(&tri);
        assert!(h1 >= h2, "H1 ({h1}) should be >= H2 ({h2})");
        assert!(h2 >= h3, "H2 ({h2}) should be >= H3 ({h3})");
    }
}
