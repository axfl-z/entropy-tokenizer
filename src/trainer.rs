use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use crate::stats;
use crate::vocab::Vocabulary;

/// A word (byte sequence) with its frequency count.
/// Merges happen within words, weighted by frequency.
#[derive(Debug, Clone)]
struct Word {
    tokens: Vec<u32>,
    count: u64,
}

/// Training progress callback info
#[derive(Debug, Clone)]
pub struct TrainProgress {
    pub step: usize,
    pub total_steps: usize,
    pub vocab_size: usize,
    pub total_tokens: u64,
    pub entropy: f64,
    pub elapsed_secs: f64,
    pub eta_secs: f64,
    pub phase: u8,
}

/// Dataset quality report
#[derive(Debug, Clone)]
pub struct DatasetQuality {
    pub total_bytes: usize,
    pub unique_bytes: usize,
    pub byte_entropy: f64,
    pub avg_word_len: f64,
    pub unique_words: usize,
    pub total_words: usize,
    pub warnings: Vec<String>,
    pub quality_score: f64,
}

pub struct Trainer {
    pub vocab: Vocabulary,
    words: Vec<Word>,
    pair_counts: HashMap<(u32, u32), i64>,
    token_counts: HashMap<u32, u64>,
    bigram_counts: HashMap<(u32, u32), u64>,
    phase: u8,
}

impl Trainer {
    pub fn new(corpus: &[u8]) -> Self {
        let vocab = Vocabulary::new_byte_level();

        // Pre-tokenize: split corpus into "words" at whitespace boundaries.
        // Each word includes leading whitespace (like GPT-2 BPE).
        // This focuses merges on within-word patterns for better compression.
        let mut word_freq: HashMap<Vec<u8>, u64> = HashMap::new();
        let mut current_word: Vec<u8> = Vec::new();

        for &b in corpus {
            if b == b' ' || b == b'\n' || b == b'\t' || b == b'\r' {
                if !current_word.is_empty() {
                    *word_freq.entry(current_word.clone()).or_insert(0) += 1;
                    current_word.clear();
                }
                // Start new word with the whitespace char as prefix
                current_word.push(b);
            } else {
                current_word.push(b);
            }
        }
        if !current_word.is_empty() {
            *word_freq.entry(current_word).or_insert(0) += 1;
        }

        let words: Vec<Word> = word_freq
            .into_iter()
            .map(|(bytes, count)| Word {
                tokens: bytes.iter().map(|&b| b as u32).collect(),
                count,
            })
            .collect();

        let mut trainer = Trainer {
            vocab,
            words,
            pair_counts: HashMap::new(),
            token_counts: HashMap::new(),
            bigram_counts: HashMap::new(),
            phase: 0,
        };
        trainer.recount_all();
        trainer
    }

    fn recount_all(&mut self) {
        self.pair_counts.clear();
        self.token_counts.clear();
        self.bigram_counts.clear();

        for word in &self.words {
            let freq = word.count as i64;
            for (i, &tid) in word.tokens.iter().enumerate() {
                *self.token_counts.entry(tid).or_insert(0) += word.count;
                if i + 1 < word.tokens.len() {
                    let pair = (tid, word.tokens[i + 1]);
                    *self.pair_counts.entry(pair).or_insert(0) += freq;
                    *self.bigram_counts.entry(pair).or_insert(0) += word.count;
                }
            }
        }
    }

    fn merge_score(
        &self,
        left_id: u32,
        right_id: u32,
        pair_count: u64,
        next_id: u32,
    ) -> f64 {
        let freq = pair_count as f64;

        match self.phase {
            0 => {
                // Phase 0: Pure compression (maximize bytes saved per merge).
                // Score = freq * merged_len (bytes saved = freq * (left_len + right_len - 1))
                // This directly optimizes for density like BPE but weighted by
                // how many bytes each merge actually saves.
                let left_len = self.vocab.get_bytes(left_id).len() as f64;
                let right_len = self.vocab.get_bytes(right_id).len() as f64;
                freq * (left_len + right_len)
            }
            _ => {
                // Phase 1: Entropy-optimized refinement
                let entropy_delta = stats::entropy_delta_for_merge(
                    &self.token_counts,
                    left_id,
                    right_id,
                    next_id,
                    pair_count,
                );
                let left_len = self.vocab.get_bytes(left_id).len() as f64;
                let right_len = self.vocab.get_bytes(right_id).len() as f64;
                let byte_score = freq * (left_len + right_len);
                byte_score.powf(0.7) * (1.0 + 0.3 * entropy_delta)
            }
        }
    }

    fn find_best_merge(&self) -> Option<((u32, u32), u64, f64)> {
        let next_id = self.vocab.tokens.len() as u32;
        let mut best: Option<((u32, u32), u64, f64)> = None;

        for (&(left, right), &count) in &self.pair_counts {
            if count < 1 {
                continue;
            }
            let count_u64 = count as u64;
            let score = self.merge_score(left, right, count_u64, next_id);
            if let Some((_, _, best_score)) = &best {
                if score > *best_score {
                    best = Some(((left, right), count_u64, score));
                }
            } else {
                best = Some(((left, right), count_u64, score));
            }
        }
        best
    }

    fn apply_merge(&mut self, left_id: u32, right_id: u32, new_id: u32) {
        let mut pair_deltas: HashMap<(u32, u32), i64> = HashMap::new();
        let mut token_deltas: HashMap<u32, i64> = HashMap::new();

        for word in &mut self.words {
            let freq = word.count as i64;
            let mut new_tokens = Vec::with_capacity(word.tokens.len());
            let mut i = 0;
            let mut had_merge = false;
            while i < word.tokens.len() {
                if i + 1 < word.tokens.len()
                    && word.tokens[i] == left_id
                    && word.tokens[i + 1] == right_id
                {
                    had_merge = true;
                    if !new_tokens.is_empty() {
                        let prev = *new_tokens.last().unwrap();
                        *pair_deltas.entry((prev, left_id)).or_insert(0) -= freq;
                        *pair_deltas.entry((prev, new_id)).or_insert(0) += freq;
                    }
                    if i + 2 < word.tokens.len() {
                        let next = word.tokens[i + 2];
                        *pair_deltas.entry((right_id, next)).or_insert(0) -= freq;
                        *pair_deltas.entry((new_id, next)).or_insert(0) += freq;
                    }
                    *pair_deltas.entry((left_id, right_id)).or_insert(0) -= freq;
                    *token_deltas.entry(left_id).or_insert(0) -= freq;
                    *token_deltas.entry(right_id).or_insert(0) -= freq;
                    *token_deltas.entry(new_id).or_insert(0) += freq;
                    new_tokens.push(new_id);
                    i += 2;
                } else {
                    new_tokens.push(word.tokens[i]);
                    i += 1;
                }
            }
            if had_merge {
                word.tokens = new_tokens;
            }
        }

        for ((l, r), delta) in &pair_deltas {
            let entry = self.pair_counts.entry((*l, *r)).or_insert(0);
            *entry += delta;
            if *entry <= 0 {
                self.pair_counts.remove(&(*l, *r));
            }
        }

        for (&tid, &delta) in &token_deltas {
            if delta > 0 {
                *self.token_counts.entry(tid).or_insert(0) += delta as u64;
            } else {
                let count = self.token_counts.entry(tid).or_insert(0);
                let sub = (-delta) as u64;
                if *count <= sub {
                    self.token_counts.remove(&tid);
                } else {
                    *count -= sub;
                }
            }
        }
    }

    pub fn into_vocab(self) -> Vocabulary {
        self.vocab
    }

    /// Check dataset quality before training. Returns warnings if dataset is problematic.
    pub fn check_dataset_quality(corpus: &[u8]) -> DatasetQuality {
        let total_bytes = corpus.len();
        let mut byte_counts = [0u64; 256];
        for &b in corpus {
            byte_counts[b as usize] += 1;
        }
        let unique_bytes = byte_counts.iter().filter(|&&c| c > 0).count();
        let byte_entropy = stats::shannon_entropy(&byte_counts);

        let mut word_freq: HashMap<Vec<u8>, u64> = HashMap::new();
        let mut current_word: Vec<u8> = Vec::new();
        let mut total_word_len: usize = 0;

        for &b in corpus {
            if b == b' ' || b == b'\n' || b == b'\t' || b == b'\r' {
                if !current_word.is_empty() {
                    total_word_len += current_word.len();
                    *word_freq.entry(current_word.clone()).or_insert(0) += 1;
                    current_word.clear();
                }
            } else {
                current_word.push(b);
            }
        }
        if !current_word.is_empty() {
            total_word_len += current_word.len();
            *word_freq.entry(current_word).or_insert(0) += 1;
        }

        let total_words: u64 = word_freq.values().sum();
        let unique_words = word_freq.len();
        let avg_word_len = if total_words > 0 {
            total_word_len as f64 / total_words as f64
        } else {
            0.0
        };

        let mut warnings = Vec::new();
        let mut quality_score: f64 = 1.0;

        if total_bytes < 100 {
            warnings.push(format!("Dataset very small ({} bytes). Tokenizer quality will be poor.", total_bytes));
            quality_score -= 0.4;
        } else if total_bytes < 1000 {
            warnings.push(format!("Dataset small ({} bytes). Consider using more data for better results.", total_bytes));
            quality_score -= 0.2;
        }

        if byte_entropy < 3.0 {
            warnings.push(format!("Very low byte entropy ({:.2} bits). Data may be too repetitive or uniform.", byte_entropy));
            quality_score -= 0.3;
        } else if byte_entropy < 4.0 {
            warnings.push(format!("Low byte entropy ({:.2} bits). Data has limited diversity.", byte_entropy));
            quality_score -= 0.15;
        }

        if unique_bytes < 20 {
            warnings.push(format!("Only {} unique byte values. Data uses very limited character set.", unique_bytes));
            quality_score -= 0.2;
        }

        if avg_word_len > 50.0 {
            warnings.push(format!("Average word length {:.1} chars — extremely long. Data may be binary or encoded.", avg_word_len));
            quality_score -= 0.3;
        } else if avg_word_len < 1.5 {
            warnings.push(format!("Average word length {:.1} chars — very short. Data is unusually fragmented.", avg_word_len));
            quality_score -= 0.15;
        }

        if total_words > 0 && unique_words as f64 / total_words as f64 > 0.95 {
            warnings.push("Almost all words are unique. Data may be random/noisy — tokenizer won't learn good merges.".to_string());
            quality_score -= 0.25;
        }

        let non_printable: u64 = byte_counts[..32].iter().sum::<u64>() - byte_counts[b'\n' as usize] - byte_counts[b'\t' as usize] - byte_counts[b'\r' as usize];
        let non_printable_ratio = non_printable as f64 / total_bytes.max(1) as f64;
        if non_printable_ratio > 0.1 {
            warnings.push(format!("High non-printable byte ratio ({:.1}%). Data may be binary.", non_printable_ratio * 100.0));
            quality_score -= 0.2;
        }

        quality_score = quality_score.clamp(0.0, 1.0);

        DatasetQuality {
            total_bytes,
            unique_bytes,
            byte_entropy,
            avg_word_len,
            unique_words,
            total_words: total_words as usize,
            warnings,
            quality_score,
        }
    }

    pub fn train(&mut self, target_vocab_size: usize, verbose: bool) {
        self.train_with_progress(target_vocab_size, verbose, true);
    }

    /// Train with optional progress bar
    pub fn train_with_progress(&mut self, target_vocab_size: usize, verbose: bool, show_progress: bool) {
        let start_size = self.vocab.vocab_size;
        let num_merges = target_vocab_size.saturating_sub(start_size);
        let train_start = Instant::now();

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

        let progress_interval = if num_merges > 1000 { num_merges / 100 } else if num_merges > 100 { num_merges / 20 } else { 1 };

        for step in 0..num_merges {
            let phase_boundary = (num_merges * 9) / 10;
            if step < phase_boundary {
                self.phase = 0;
            } else {
                self.phase = 1;
            }

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

            if show_progress && progress_interval > 0 && (step + 1) % progress_interval == 0 {
                let elapsed = train_start.elapsed().as_secs_f64();
                let progress = (step + 1) as f64 / num_merges as f64;
                let eta = if progress > 0.0 { elapsed / progress - elapsed } else { 0.0 };
                let bar_width = 30;
                let filled = (progress * bar_width as f64) as usize;
                let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
                let phase_str = if self.phase == 0 { "compress" } else { "entropy" };
                eprint!("\r  [{bar}] {:.0}% | {}/{} merges | {phase_str} | {:.1}s elapsed | ETA {:.1}s  ",
                    progress * 100.0, step + 1, num_merges, elapsed, eta);
                let _ = std::io::stderr().flush();
            }

            if verbose && (step + 1) % 100 == 0 {
                let total_tokens: u64 = self.token_counts.values().sum();
                let entropy = stats::entropy_from_map(&self.token_counts);
                let left_bytes = self.vocab.get_bytes(left_id);
                let right_bytes = self.vocab.get_bytes(right_id);
                let merged = self.vocab.get_bytes(new_id);
                eprintln!(
                    "\nStep {}/{}: merged {:?} + {:?} -> {:?} (count={}, score={:.2}, tokens={}, H={:.4})",
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

        if show_progress && num_merges > 0 {
            let elapsed = train_start.elapsed().as_secs_f64();
            let bar: String = "█".repeat(30);
            eprint!("\r  [{bar}] 100% | {num_merges}/{num_merges} merges | done | {elapsed:.1}s total          \n");
            let _ = std::io::stderr().flush();
        }

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
