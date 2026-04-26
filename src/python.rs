/// Python bindings for the Entropy-Optimal Tokenizer (EOT).
/// All encoding/decoding releases the GIL for true parallelism.
#[cfg(feature = "python")]
pub mod pymod {
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    use crate::encoder::Encoder;
    use crate::filters::{FilterPipeline, LowercaseAscii, NormalizeWhitespace, StripNonPrintable};
    use crate::trainer::Trainer;
    use crate::vocab::Vocabulary;

    /// Python-facing EOT tokenizer.
    /// Train from a corpus or load a pre-trained model, then encode/decode.
    #[pyclass(name = "EOTTokenizer")]
    pub struct PyEOTTokenizer {
        encoder: Encoder,
        filters: FilterPipeline,
    }

    #[pymethods]
    impl PyEOTTokenizer {
        /// Create a new tokenizer by training on a byte corpus.
        ///
        /// Args:
        ///     corpus: Training text (str or bytes).
        ///     vocab_size: Target vocabulary size (default 8192).
        ///     context_weight: Weight for bigram context scoring (default 0.3).
        ///     verbose: Print training progress (default False).
        ///     show_progress: Show progress bar with ETA (default True).
        ///     special_tokens: List of special token strings (default None).
        ///     check_quality: Check dataset quality before training (default True).
        #[new]
        #[pyo3(signature = (corpus, vocab_size=8192, context_weight=0.3, verbose=false, show_progress=true, special_tokens=None, check_quality=true))]
        fn new(
            corpus: &Bound<'_, PyAny>,
            vocab_size: usize,
            context_weight: f64,
            verbose: bool,
            show_progress: bool,
            special_tokens: Option<Vec<String>>,
            check_quality: bool,
        ) -> PyResult<Self> {
            let bytes: Vec<u8> = if let Ok(s) = corpus.extract::<String>() {
                s.into_bytes()
            } else if let Ok(b) = corpus.extract::<Vec<u8>>() {
                b
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "corpus must be str or bytes",
                ));
            };

            let specials = special_tokens.unwrap_or_default();

            let (encoder,) = Python::with_gil(|py| {
                py.allow_threads(|| {
                    if check_quality {
                        let quality = Trainer::check_dataset_quality(&bytes);
                        if !quality.warnings.is_empty() {
                            eprintln!("=== Dataset Quality Check ===");
                            eprintln!("  Score: {:.0}%", quality.quality_score * 100.0);
                            for w in &quality.warnings {
                                eprintln!("  WARNING: {}", w);
                            }
                            eprintln!("=============================");
                        }
                    }

                    let mut trainer = Trainer::new(&bytes);
                    trainer.train_with_progress(vocab_size, verbose, show_progress);
                    let mut vocab = trainer.into_vocab();

                    for name in &specials {
                        vocab.add_special_token(name);
                    }

                    (Encoder::new(vocab, context_weight),)
                })
            });

            Ok(PyEOTTokenizer {
                encoder,
                filters: FilterPipeline::new(),
            })
        }

        /// Load a pre-trained model from a JSON file.
        #[staticmethod]
        #[pyo3(signature = (path, context_weight=0.3))]
        fn from_file(path: &str, context_weight: f64) -> PyResult<Self> {
            let vocab = Vocabulary::load(path).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to load model: {}", e))
            })?;
            Ok(PyEOTTokenizer {
                encoder: Encoder::new(vocab, context_weight),
                filters: FilterPipeline::new(),
            })
        }

        /// Save the trained model to a JSON file.
        fn save(&self, path: &str) -> PyResult<()> {
            self.encoder.vocab().save(path).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to save model: {}", e))
            })
        }

        /// Add a pre-tokenization filter by name.
        /// Available: "lowercase", "normalize_whitespace", "strip_non_printable"
        fn add_filter(&mut self, name: &str) -> PyResult<()> {
            match name {
                "lowercase" | "lowercase_ascii" => {
                    self.filters.add(Box::new(LowercaseAscii));
                }
                "normalize_whitespace" => {
                    self.filters.add(Box::new(NormalizeWhitespace));
                }
                "strip_non_printable" => {
                    self.filters.add(Box::new(StripNonPrintable));
                }
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown filter: '{}'. Available: lowercase, normalize_whitespace, strip_non_printable",
                        name
                    )));
                }
            }
            Ok(())
        }

        /// Get list of active filter names.
        fn active_filters(&self) -> Vec<String> {
            self.filters.filter_names().iter().map(|s| s.to_string()).collect()
        }

        /// Add a special token. Returns its ID.
        fn add_special_token(&mut self, name: &str) -> u32 {
            self.encoder.vocab_mut().add_special_token(name)
        }

        /// Get special token ID by name, or None.
        fn get_special_token_id(&self, name: &str) -> Option<u32> {
            self.encoder.vocab().get_special_token(name)
        }

        /// List all special tokens as list of (name, id) tuples.
        fn special_tokens(&self) -> Vec<(String, u32)> {
            self.encoder.vocab().special_tokens.iter()
                .map(|s| (s.name.clone(), s.id))
                .collect()
        }

        /// Prune vocabulary to target size, keeping most frequent tokens.
        /// Base byte tokens (0-255) and special tokens are always kept.
        fn prune_vocab(&mut self, target_size: usize, corpus: &Bound<'_, PyAny>) -> PyResult<()> {
            let bytes: Vec<u8> = if let Ok(s) = corpus.extract::<String>() {
                s.into_bytes()
            } else if let Ok(b) = corpus.extract::<Vec<u8>>() {
                b
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "corpus must be str or bytes",
                ));
            };

            let old_size = self.encoder.vocab().vocab_size;
            let tokens = self.encoder.encode_greedy(&bytes);
            let mut counts = std::collections::HashMap::new();
            for &t in &tokens {
                *counts.entry(t).or_insert(0u64) += 1;
            }

            self.encoder.vocab_mut().prune(target_size, &counts);
            let new_size = self.encoder.vocab().vocab_size;

            // Rebuild encoder with pruned vocab
            let context_weight = self.encoder.context_weight();
            let vocab = self.encoder.vocab().clone();
            self.encoder = Encoder::new(vocab, context_weight);

            eprintln!("Pruned vocab: {} -> {} tokens", old_size, new_size);
            Ok(())
        }

        /// Check dataset quality. Returns dict with score and warnings.
        #[staticmethod]
        fn check_quality(corpus: &Bound<'_, PyAny>) -> PyResult<PyObject> {
            let bytes: Vec<u8> = if let Ok(s) = corpus.extract::<String>() {
                s.into_bytes()
            } else if let Ok(b) = corpus.extract::<Vec<u8>>() {
                b
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "corpus must be str or bytes",
                ));
            };

            let quality = Trainer::check_dataset_quality(&bytes);

            Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("quality_score", quality.quality_score)?;
                dict.set_item("total_bytes", quality.total_bytes)?;
                dict.set_item("unique_bytes", quality.unique_bytes)?;
                dict.set_item("byte_entropy", quality.byte_entropy)?;
                dict.set_item("avg_word_len", quality.avg_word_len)?;
                dict.set_item("unique_words", quality.unique_words)?;
                dict.set_item("total_words", quality.total_words)?;
                dict.set_item("warnings", quality.warnings)?;
                Ok(dict.into())
            })
        }

        /// Encode text into token IDs.
        #[pyo3(signature = (text, greedy=false))]
        fn encode(&self, text: &Bound<'_, PyAny>, greedy: bool) -> PyResult<Vec<u32>> {
            let bytes = self.extract_and_filter(text)?;

            let tokens = Python::with_gil(|py| {
                py.allow_threads(|| {
                    if greedy {
                        self.encoder.encode_greedy(&bytes)
                    } else {
                        self.encoder.encode(&bytes)
                    }
                })
            });

            Ok(tokens)
        }

        /// Encode text using density-optimal DP (minimizes token count).
        fn encode_dense(&self, text: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
            let bytes = self.extract_and_filter(text)?;

            let tokens = Python::with_gil(|py| {
                py.allow_threads(|| self.encoder.encode_dense(&bytes))
            });

            Ok(tokens)
        }

        /// Encode text and return token IDs along with byte offsets.
        #[pyo3(signature = (text, greedy=false))]
        fn encode_with_offsets(
            &self,
            text: &Bound<'_, PyAny>,
            greedy: bool,
        ) -> PyResult<(Vec<u32>, Vec<(usize, usize)>)> {
            let bytes = self.extract_and_filter(text)?;

            let tokens = Python::with_gil(|py| {
                py.allow_threads(|| {
                    if greedy {
                        self.encoder.encode_greedy(&bytes)
                    } else {
                        self.encoder.encode(&bytes)
                    }
                })
            });

            let mut offsets = Vec::with_capacity(tokens.len());
            let mut pos = 0usize;
            for &tid in &tokens {
                let token_bytes = self.encoder.vocab().get_bytes(tid);
                let end = pos + token_bytes.len();
                offsets.push((pos, end));
                pos = end;
            }

            Ok((tokens, offsets))
        }

        /// Decode token IDs back to bytes.
        fn decode_bytes<'py>(
            &self,
            py: Python<'py>,
            token_ids: Vec<u32>,
        ) -> PyResult<Bound<'py, PyBytes>> {
            let bytes = py.allow_threads(|| self.encoder.decode(&token_ids));
            Ok(PyBytes::new_bound(py, &bytes))
        }

        /// Decode token IDs back to a UTF-8 string.
        fn decode(&self, token_ids: Vec<u32>) -> PyResult<String> {
            let bytes = Python::with_gil(|py| {
                py.allow_threads(|| self.encoder.decode(&token_ids))
            });
            Ok(String::from_utf8_lossy(&bytes).to_string())
        }

        /// Get the byte representation of a token.
        fn id_to_bytes<'py>(
            &self,
            py: Python<'py>,
            token_id: u32,
        ) -> PyResult<Bound<'py, PyBytes>> {
            if (token_id as usize) >= self.encoder.vocab().tokens.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "token_id {} out of range (vocab size {})",
                    token_id,
                    self.encoder.vocab().vocab_size
                )));
            }
            let bytes = self.encoder.vocab().get_bytes(token_id);
            Ok(PyBytes::new_bound(py, bytes))
        }

        /// Get the string representation of a token (lossy UTF-8).
        fn id_to_token(&self, token_id: u32) -> PyResult<String> {
            if (token_id as usize) >= self.encoder.vocab().tokens.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "token_id {} out of range (vocab size {})",
                    token_id,
                    self.encoder.vocab().vocab_size
                )));
            }
            let bytes = self.encoder.vocab().get_bytes(token_id);
            Ok(String::from_utf8_lossy(bytes).to_string())
        }

        /// Get the vocabulary size.
        #[getter]
        fn vocab_size(&self) -> usize {
            self.encoder.vocab().vocab_size
        }

        /// Check if this tokenizer is lossless (always True for EOT).
        #[getter]
        fn is_lossless(&self) -> bool {
            true
        }

        fn __repr__(&self) -> String {
            format!(
                "EOTTokenizer(vocab_size={}, lossless=True, filters={})",
                self.encoder.vocab().vocab_size,
                if self.filters.is_empty() { "none".to_string() } else { format!("{:?}", self.filters.filter_names()) }
            )
        }

        fn __len__(&self) -> usize {
            self.encoder.vocab().vocab_size
        }
    }

    impl PyEOTTokenizer {
        fn extract_and_filter(&self, text: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
            let bytes: Vec<u8> = if let Ok(s) = text.extract::<String>() {
                s.into_bytes()
            } else if let Ok(b) = text.extract::<Vec<u8>>() {
                b
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "text must be str or bytes",
                ));
            };
            if self.filters.is_empty() {
                Ok(bytes)
            } else {
                Ok(self.filters.apply(&bytes))
            }
        }
    }

    /// Register the Python module.
    #[pymodule]
    pub fn entropy_tokenizer_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyEOTTokenizer>()?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }
}
