/// Python bindings for the Entropy-Optimal Tokenizer (EOT).
/// All encoding/decoding releases the GIL for true parallelism.
#[cfg(feature = "python")]
pub mod pymod {
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    use crate::encoder::Encoder;
    use crate::trainer::Trainer;
    use crate::vocab::Vocabulary;

    /// Python-facing EOT tokenizer.
    /// Train from a corpus or load a pre-trained model, then encode/decode.
    #[pyclass(name = "EOTTokenizer")]
    pub struct PyEOTTokenizer {
        encoder: Encoder,
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
        #[new]
        #[pyo3(signature = (corpus, vocab_size=8192, context_weight=0.3, verbose=false))]
        fn new(
            corpus: &Bound<'_, PyAny>,
            vocab_size: usize,
            context_weight: f64,
            verbose: bool,
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

            let (encoder,) = Python::with_gil(|py| {
                py.allow_threads(|| {
                    let mut trainer = Trainer::new(&bytes);
                    trainer.train(vocab_size, verbose);
                    let vocab = trainer.into_vocab();
                    (Encoder::new(vocab, context_weight),)
                })
            });

            Ok(PyEOTTokenizer { encoder })
        }

        /// Load a pre-trained model from a JSON file.
        ///
        /// Args:
        ///     path: Path to the model JSON file.
        ///     context_weight: Weight for bigram context scoring (default 0.3).
        #[staticmethod]
        #[pyo3(signature = (path, context_weight=0.3))]
        fn from_file(path: &str, context_weight: f64) -> PyResult<Self> {
            let vocab = Vocabulary::load(path).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to load model: {}", e))
            })?;
            Ok(PyEOTTokenizer {
                encoder: Encoder::new(vocab, context_weight),
            })
        }

        /// Save the trained model to a JSON file.
        fn save(&self, path: &str) -> PyResult<()> {
            self.encoder.vocab().save(path).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to save model: {}", e))
            })
        }

        /// Encode text into token IDs using DP-optimal segmentation.
        /// Releases the GIL during encoding for true parallelism.
        ///
        /// Args:
        ///     text: Input text (str or bytes).
        ///     greedy: Use faster greedy encoding instead of DP-optimal (default False).
        ///
        /// Returns:
        ///     List of token IDs.
        #[pyo3(signature = (text, greedy=false))]
        fn encode(&self, text: &Bound<'_, PyAny>, greedy: bool) -> PyResult<Vec<u32>> {
            let bytes: Vec<u8> = if let Ok(s) = text.extract::<String>() {
                s.into_bytes()
            } else if let Ok(b) = text.extract::<Vec<u8>>() {
                b
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "text must be str or bytes",
                ));
            };

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

        /// Encode text and return token IDs along with byte offsets.
        /// Releases the GIL during encoding.
        ///
        /// Args:
        ///     text: Input text (str or bytes).
        ///     greedy: Use greedy encoding (default False).
        ///
        /// Returns:
        ///     Tuple of (token_ids, offsets) where offsets is a list of (start, end) byte positions.
        #[pyo3(signature = (text, greedy=false))]
        fn encode_with_offsets(
            &self,
            text: &Bound<'_, PyAny>,
            greedy: bool,
        ) -> PyResult<(Vec<u32>, Vec<(usize, usize)>)> {
            let bytes: Vec<u8> = if let Ok(s) = text.extract::<String>() {
                s.into_bytes()
            } else if let Ok(b) = text.extract::<Vec<u8>>() {
                b
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "text must be str or bytes",
                ));
            };

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
        /// Releases the GIL during decoding.
        ///
        /// Args:
        ///     token_ids: List of token IDs.
        ///
        /// Returns:
        ///     Decoded bytes.
        fn decode_bytes<'py>(
            &self,
            py: Python<'py>,
            token_ids: Vec<u32>,
        ) -> PyResult<Bound<'py, PyBytes>> {
            let bytes = py.allow_threads(|| self.encoder.decode(&token_ids));
            Ok(PyBytes::new_bound(py, &bytes))
        }

        /// Decode token IDs back to a UTF-8 string.
        /// Invalid UTF-8 sequences are replaced with the replacement character.
        ///
        /// Args:
        ///     token_ids: List of token IDs.
        ///
        /// Returns:
        ///     Decoded string.
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
                "EOTTokenizer(vocab_size={}, lossless=True)",
                self.encoder.vocab().vocab_size
            )
        }

        fn __len__(&self) -> usize {
            self.encoder.vocab().vocab_size
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
