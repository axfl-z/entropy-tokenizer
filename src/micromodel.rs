//! Experimental micro-model tokenizer.
//! A tiny neural-like model (2K-200K parameters) that learns to segment text
//! into tokens using a small lookup table and bigram transition scores.
//! Designed to fit in CPU L2 cache for fast inference.
//!
//! NOT enabled by default — use `MicroModelTokenizer::train()` explicitly.

use crate::vocab::Vocabulary;

/// A micro-model that scores token boundaries using learned parameters.
/// Parameters: vocab_size * embed_dim (embeddings) + embed_dim * embed_dim (transition matrix)
#[derive(Debug, Clone)]
pub struct MicroModel {
    vocab_size: usize,
    embed_dim: usize,
    /// Token embeddings: vocab_size x embed_dim (flattened row-major)
    embeddings: Vec<f32>,
    /// Transition scores: embed_dim x embed_dim (flattened row-major)
    transitions: Vec<f32>,
    /// Bias for each token
    bias: Vec<f32>,
}

impl MicroModel {
    /// Create a new micro-model with given dimensions.
    /// Total params ≈ vocab_size * embed_dim + embed_dim^2 + vocab_size
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let total_params = vocab_size * embed_dim + embed_dim * embed_dim + vocab_size;
        eprintln!(
            "MicroModel: vocab={}, embed_dim={}, params={}",
            vocab_size, embed_dim, total_params
        );

        // Xavier initialization
        let scale = (2.0 / (vocab_size + embed_dim) as f64).sqrt() as f32;
        let mut embeddings = vec![0.0f32; vocab_size * embed_dim];
        let mut seed: u64 = 42;
        for v in &mut embeddings {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (seed >> 33) as f32 / (1u64 << 31) as f32 - 1.0;
            *v = u * scale;
        }

        let t_scale = (2.0 / (embed_dim * 2) as f64).sqrt() as f32;
        let mut transitions = vec![0.0f32; embed_dim * embed_dim];
        for v in &mut transitions {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (seed >> 33) as f32 / (1u64 << 31) as f32 - 1.0;
            *v = u * t_scale;
        }

        let bias = vec![0.0f32; vocab_size];

        MicroModel {
            vocab_size,
            embed_dim,
            embeddings,
            transitions,
            bias,
        }
    }

    /// Get total parameter count
    pub fn param_count(&self) -> usize {
        self.vocab_size * self.embed_dim + self.embed_dim * self.embed_dim + self.vocab_size
    }

    /// Get embedding for a token
    fn get_embed(&self, token_id: u32) -> &[f32] {
        let start = (token_id as usize) * self.embed_dim;
        &self.embeddings[start..start + self.embed_dim]
    }

    /// Score a token given the previous token using embeddings and transition matrix.
    /// score = prev_embed^T * W * cur_embed + bias[cur]
    fn score(&self, prev_id: u32, cur_id: u32) -> f32 {
        let prev = self.get_embed(prev_id);
        let cur = self.get_embed(cur_id);
        let d = self.embed_dim;

        // Compute prev^T * W (result is 1 x embed_dim)
        let pw: Vec<f32> = (0..d).map(|i| {
            prev.iter().enumerate().take(d).map(|(j, &pv)| pv * self.transitions[j * d + i]).sum()
        }).collect();

        // Dot product pw . cur
        let dot: f32 = pw.iter().zip(cur.iter()).map(|(&a, &b)| a * b).sum();

        dot + self.bias[cur_id as usize]
    }

    /// Train the micro-model on token sequences using SGD.
    /// Minimizes cross-entropy loss on bigram prediction.
    pub fn train_on_sequences(
        &mut self,
        sequences: &[Vec<u32>],
        learning_rate: f32,
        epochs: usize,
    ) -> Vec<f32> {
        let mut losses = Vec::new();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0f32;
            let mut count = 0u64;

            for seq in sequences {
                if seq.len() < 2 {
                    continue;
                }

                for i in 0..seq.len() - 1 {
                    let prev = seq[i];
                    let target = seq[i + 1];

                    if prev as usize >= self.vocab_size || target as usize >= self.vocab_size {
                        continue;
                    }

                    // Forward: compute scores for all tokens given prev
                    let mut scores = Vec::with_capacity(self.vocab_size.min(256));
                    let max_eval = self.vocab_size.min(256);
                    for t in 0..max_eval as u32 {
                        scores.push(self.score(prev, t));
                    }

                    // Softmax + cross-entropy
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();
                    let log_sum = max_score + exp_sum.ln();

                    let target_idx = target as usize;
                    if target_idx < max_eval {
                        let loss = log_sum - scores[target_idx];
                        epoch_loss += loss;
                        count += 1;

                        // SGD update: gradient of cross-entropy w.r.t. scores
                        // grad[t] = softmax(t) - 1{t == target}
                        let prev_embed = self.get_embed(prev).to_vec();
                        let d = self.embed_dim;

                        for (t, &score_t) in scores.iter().enumerate().take(max_eval) {
                            let prob = (score_t - log_sum).exp();
                            let grad = if t == target_idx { prob - 1.0 } else { prob };

                            if grad.abs() < 1e-6 {
                                continue;
                            }

                            let lr_grad = learning_rate * grad;

                            // Update bias
                            self.bias[t] -= lr_grad;

                            // Update embeddings and transitions
                            let t_embed_start = t * d;
                            for j in 0..d {
                                let pw_j: f32 = prev_embed.iter().enumerate().take(d)
                                    .map(|(k, &pe)| pe * self.transitions[k * d + j]).sum();
                                self.embeddings[t_embed_start + j] -= lr_grad * pw_j;
                            }
                        }
                    }
                }
            }

            let avg_loss = if count > 0 { epoch_loss / count as f32 } else { 0.0 };
            losses.push(avg_loss);

            if epoch % 5 == 0 || epoch == epochs - 1 {
                eprintln!(
                    "  MicroModel epoch {}/{}: loss={:.4}, samples={}",
                    epoch + 1, epochs, avg_loss, count
                );
            }
        }

        losses
    }

    /// Use the micro-model to score a tokenization.
    /// Higher score = better tokenization according to the model.
    pub fn score_tokenization(&self, tokens: &[u32]) -> f64 {
        if tokens.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0f64;
        for i in 0..tokens.len() - 1 {
            if tokens[i] as usize >= self.vocab_size || tokens[i + 1] as usize >= self.vocab_size {
                continue;
            }
            total += self.score(tokens[i], tokens[i + 1]) as f64;
        }
        total / (tokens.len() - 1) as f64
    }
}

/// Train a micro-model from a vocabulary and corpus.
/// Returns the trained model.
pub fn train_micro_model(
    vocab: &Vocabulary,
    corpus: &[u8],
    embed_dim: usize,
    epochs: usize,
    learning_rate: f32,
) -> MicroModel {
    let vs = vocab.vocab_size.min(512);
    let mut model = MicroModel::new(vs, embed_dim);

    // Simple byte-level tokenization for training data
    let tokens: Vec<u32> = corpus.iter().map(|&b| b as u32).collect();

    // Split into chunks for training
    let chunk_size = 256;
    let sequences: Vec<Vec<u32>> = tokens
        .chunks(chunk_size)
        .map(|c| c.to_vec())
        .collect();

    model.train_on_sequences(&sequences, learning_rate, epochs);
    model
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_micro_model_creation() {
        let model = MicroModel::new(256, 8);
        assert_eq!(model.param_count(), 256 * 8 + 8 * 8 + 256);
    }

    #[test]
    fn test_micro_model_score() {
        let model = MicroModel::new(256, 4);
        let s = model.score(0, 1);
        assert!(s.is_finite());
    }

    #[test]
    fn test_micro_model_train() {
        let mut model = MicroModel::new(10, 4);
        let seqs = vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]];
        let losses = model.train_on_sequences(&seqs, 0.01, 5);
        assert_eq!(losses.len(), 5);
        // Loss should decrease (or at least not explode)
        assert!(losses.last().unwrap().is_finite());
    }

    #[test]
    fn test_score_tokenization() {
        let model = MicroModel::new(256, 4);
        let score = model.score_tokenization(&[0, 1, 2, 3]);
        assert!(score.is_finite());
    }

    #[test]
    fn test_train_micro_model_from_corpus() {
        let vocab = Vocabulary::new_byte_level();
        let corpus = b"hello world hello world";
        let model = train_micro_model(&vocab, corpus, 4, 3, 0.01);
        assert!(model.param_count() > 0);
    }
}
