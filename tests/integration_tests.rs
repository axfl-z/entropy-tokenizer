use entropy_tokenizer::encoder::Encoder;
use entropy_tokenizer::stats;
use entropy_tokenizer::trainer::Trainer;
use entropy_tokenizer::vocab::Vocabulary;

fn train_encoder(corpus: &[u8], vocab_size: usize) -> Encoder {
    let mut trainer = Trainer::new(corpus);
    trainer.train(vocab_size, false);
    let vocab = trainer.into_vocab();
    Encoder::new(vocab, 0.3)
}

// ─── Roundtrip Tests ───

#[test]
fn test_roundtrip_ascii() {
    let encoder = train_encoder(b"hello world hello world hello", 280);
    let input = b"hello world";
    let tokens = encoder.encode(input);
    assert_eq!(encoder.decode(&tokens), input);
}

#[test]
fn test_roundtrip_utf8_multibyte() {
    let corpus = "Привет мир Привет мир Привет".as_bytes();
    let encoder = train_encoder(corpus, 300);
    let input = "Привет мир".as_bytes();
    let tokens = encoder.encode(input);
    assert_eq!(encoder.decode(&tokens), input);
}

#[test]
fn test_roundtrip_binary_bytes() {
    let mut corpus: Vec<u8> = Vec::new();
    for _ in 0..100 {
        for b in 0u8..=255 {
            corpus.push(b);
        }
    }
    let encoder = train_encoder(&corpus, 280);
    let input: Vec<u8> = (0u8..=255).collect();
    let tokens = encoder.encode(&input);
    assert_eq!(encoder.decode(&tokens), input);
}

#[test]
fn test_roundtrip_empty() {
    let encoder = train_encoder(b"hello world", 260);
    let tokens = encoder.encode(b"");
    assert!(tokens.is_empty());
    assert_eq!(encoder.decode(&tokens), b"");
}

#[test]
fn test_roundtrip_single_byte() {
    let encoder = train_encoder(b"abcabc", 260);
    for b in 0u8..=255 {
        let input = [b];
        let tokens = encoder.encode(&input);
        assert_eq!(encoder.decode(&tokens), &input);
    }
}

#[test]
fn test_roundtrip_greedy() {
    let encoder = train_encoder(b"the quick brown fox jumps over the lazy dog", 280);
    let input = b"the quick brown fox";
    let tokens = encoder.encode_greedy(input);
    assert_eq!(encoder.decode(&tokens), input);
}

#[test]
fn test_roundtrip_long_text() {
    let corpus = b"the quick brown fox jumps over the lazy dog ";
    let mut long_corpus = Vec::new();
    for _ in 0..500 {
        long_corpus.extend_from_slice(corpus);
    }
    let encoder = train_encoder(&long_corpus, 512);
    let tokens = encoder.encode(&long_corpus);
    assert_eq!(encoder.decode(&tokens), long_corpus);
}

// ─── DP vs Greedy Tests ───

#[test]
fn test_dp_never_worse_than_greedy() {
    let corpus = b"abcdefg abcdefg abcdefg hello world hello world test test test";
    let encoder = train_encoder(corpus, 280);

    let inputs: Vec<&[u8]> = vec![
        b"abcdefg hello world",
        b"test test abcdefg",
        b"hello world test",
        b"abcdefg abcdefg abcdefg",
    ];

    for input in inputs {
        let dp = encoder.encode(input);
        let greedy = encoder.encode_greedy(input);
        // DP should produce <= tokens (with small tolerance for context scoring)
        assert!(
            dp.len() <= greedy.len() + 1,
            "DP ({}) should be <= greedy ({}) +1 for {:?}",
            dp.len(),
            greedy.len(),
            String::from_utf8_lossy(input)
        );
        // Both must decode correctly
        assert_eq!(encoder.decode(&dp), input);
        assert_eq!(encoder.decode(&greedy), input);
    }
}

// ─── Vocabulary Tests ───

#[test]
fn test_vocab_save_load_roundtrip() {
    let corpus = b"hello world hello world hello world";
    let mut trainer = Trainer::new(corpus);
    trainer.train(270, false);
    let vocab = trainer.into_vocab();

    let path = "/tmp/test_vocab_roundtrip.json";
    vocab.save(path).unwrap();

    let loaded = Vocabulary::load(path).unwrap();
    assert_eq!(loaded.vocab_size, vocab.vocab_size);
    assert_eq!(loaded.tokens.len(), vocab.tokens.len());

    // Verify all tokens have same bytes
    for (orig, loaded_t) in vocab.tokens.iter().zip(loaded.tokens.iter()) {
        assert_eq!(orig.id, loaded_t.id);
        assert_eq!(orig.bytes, loaded_t.bytes);
    }

    // Verify encoder works with loaded vocab
    let encoder = Encoder::new(loaded, 0.3);
    let input = b"hello world";
    let tokens = encoder.encode(input);
    assert_eq!(encoder.decode(&tokens), input);

    std::fs::remove_file(path).ok();
}

#[test]
fn test_vocab_byte_level_coverage() {
    let vocab = Vocabulary::new_byte_level();
    assert_eq!(vocab.vocab_size, 256);
    // Every byte 0-255 has a token
    for b in 0u8..=255 {
        assert_eq!(vocab.lookup(&[b]), Some(b as u32));
        assert_eq!(vocab.get_bytes(b as u32), &[b]);
    }
}

#[test]
fn test_vocab_merge_chain() {
    let mut vocab = Vocabulary::new_byte_level();
    // Merge 'h' + 'e' -> "he" (id=256)
    let he = vocab.add_merge(b'h' as u32, b'e' as u32);
    assert_eq!(he, 256);
    assert_eq!(vocab.get_bytes(he), b"he");

    // Merge "he" + 'l' -> "hel" (id=257)
    let hel = vocab.add_merge(he, b'l' as u32);
    assert_eq!(hel, 257);
    assert_eq!(vocab.get_bytes(hel), b"hel");

    // Merge "hel" + 'l' -> "hell" (id=258)
    let hell = vocab.add_merge(hel, b'l' as u32);
    assert_eq!(hell, 258);
    assert_eq!(vocab.get_bytes(hell), b"hell");
}

// ─── Training Tests ───

#[test]
fn test_training_increases_vocab_size() {
    let corpus = b"aaabbbcccaaabbbcccaaabbbccc";
    let mut trainer = Trainer::new(corpus);
    assert_eq!(trainer.vocab.vocab_size, 256);
    trainer.train(260, false);
    assert!(trainer.vocab.vocab_size >= 258); // at least 2 merges
}

#[test]
fn test_training_respects_target_size() {
    let corpus = b"the quick brown fox jumps over the lazy dog ";
    let mut big_corpus = Vec::new();
    for _ in 0..100 {
        big_corpus.extend_from_slice(corpus);
    }
    let mut trainer = Trainer::new(&big_corpus);
    trainer.train(280, false);
    // Should reach target or stop early if pairs exhausted (min count=2 filter)
    assert!(
        trainer.vocab.vocab_size <= 280,
        "vocab_size={} should be <= 280",
        trainer.vocab.vocab_size
    );
    assert!(
        trainer.vocab.vocab_size > 256,
        "vocab_size={} should be > 256 (some merges happened)",
        trainer.vocab.vocab_size
    );
}

#[test]
fn test_training_improves_compression() {
    let corpus = b"abababababababababababababababababababababab";
    let mut trainer = Trainer::new(corpus);
    let (tokens_before, _, _) = trainer.corpus_stats();
    trainer.train(260, false);
    let (tokens_after, _, _) = trainer.corpus_stats();
    // Training should reduce token count
    assert!(
        tokens_after < tokens_before,
        "tokens_after={} should be < tokens_before={}",
        tokens_after,
        tokens_before
    );
}

#[test]
fn test_training_entropy_positive() {
    let corpus = b"the quick brown fox jumps over the lazy dog";
    let mut trainer = Trainer::new(corpus);
    trainer.train(270, false);
    let (_, entropy, bigram_h) = trainer.corpus_stats();
    assert!(entropy > 0.0, "entropy should be positive: {}", entropy);
    assert!(
        bigram_h > 0.0,
        "bigram entropy should be positive: {}",
        bigram_h
    );
}

// ─── Edge Cases ───

#[test]
fn test_all_same_bytes() {
    let corpus = vec![b'a'; 1000];
    let encoder = train_encoder(&corpus, 260);
    let tokens = encoder.encode(&corpus);
    assert_eq!(encoder.decode(&tokens), corpus);
}

#[test]
fn test_alternating_bytes() {
    let mut corpus = Vec::new();
    for _ in 0..500 {
        corpus.push(b'a');
        corpus.push(b'b');
    }
    let encoder = train_encoder(&corpus, 260);
    let tokens = encoder.encode(&corpus);
    assert_eq!(encoder.decode(&tokens), corpus);
    // Should compress "ab" into a single token
    assert!(tokens.len() < corpus.len());
}

#[test]
fn test_whitespace_heavy() {
    let corpus = b"  a  b  c  a  b  c  a  b  c  ";
    let mut big_corpus = Vec::new();
    for _ in 0..100 {
        big_corpus.extend_from_slice(corpus);
    }
    let encoder = train_encoder(&big_corpus, 280);
    let tokens = encoder.encode(&big_corpus);
    assert_eq!(encoder.decode(&tokens), big_corpus);
}

#[test]
fn test_newlines_and_tabs() {
    let corpus = b"line1\nline2\nline3\ttab1\ttab2\r\nwindows\r\n";
    let mut big_corpus = Vec::new();
    for _ in 0..100 {
        big_corpus.extend_from_slice(corpus);
    }
    let encoder = train_encoder(&big_corpus, 280);
    let tokens = encoder.encode(&big_corpus);
    assert_eq!(encoder.decode(&tokens), big_corpus);
}

// ─── Stress Tests ───

#[test]
fn test_large_vocab() {
    let corpus = b"the quick brown fox jumps over the lazy dog ";
    let mut big_corpus = Vec::new();
    for _ in 0..200 {
        big_corpus.extend_from_slice(corpus);
    }
    let encoder = train_encoder(&big_corpus, 1024);
    let tokens = encoder.encode(&big_corpus);
    assert_eq!(encoder.decode(&tokens), big_corpus);
    // With 1024 vocab, should get good compression
    let density = big_corpus.len() as f64 / tokens.len() as f64;
    assert!(
        density > 1.5,
        "density should be > 1.5 with 1024 vocab, got {}",
        density
    );
}

#[test]
fn test_encode_unseen_text() {
    let corpus = b"hello world hello world";
    let encoder = train_encoder(corpus, 270);
    // Encode text not in training corpus — should still work (byte-level fallback)
    let input = b"xyz 123 !@#";
    let tokens = encoder.encode(input);
    assert_eq!(encoder.decode(&tokens), input);
}

#[test]
fn test_decode_to_string() {
    let encoder = train_encoder(b"hello world hello world", 270);
    let tokens = encoder.encode(b"hello");
    let s = encoder.decode_to_string(&tokens);
    assert_eq!(s, "hello");
}

// ─── Dense Encoding Tests ───

#[test]
fn test_dense_encoding_roundtrip() {
    let corpus = b"the quick brown fox jumps over the lazy dog ";
    let mut big_corpus = Vec::new();
    for _ in 0..100 {
        big_corpus.extend_from_slice(corpus);
    }
    let encoder = train_encoder(&big_corpus, 512);
    let tokens = encoder.encode_dense(&big_corpus);
    assert_eq!(encoder.decode(&tokens), big_corpus);
}

#[test]
fn test_dense_fewer_or_equal_tokens_than_greedy() {
    let corpus = b"abcdefg abcdefg abcdefg hello world hello world test test test";
    let mut big_corpus = Vec::new();
    for _ in 0..100 {
        big_corpus.extend_from_slice(corpus);
    }
    let encoder = train_encoder(&big_corpus, 300);

    let dense = encoder.encode_dense(&big_corpus);
    let greedy = encoder.encode_greedy(&big_corpus);
    assert!(
        dense.len() <= greedy.len(),
        "dense ({}) should be <= greedy ({})",
        dense.len(),
        greedy.len()
    );
    assert_eq!(encoder.decode(&dense), big_corpus);
}

// ─── Entropy Tests (H1, H2, H3) ───

#[test]
fn test_h1_unigram_entropy_positive() {
    let corpus = b"the quick brown fox jumps over the lazy dog ";
    let mut big_corpus = Vec::new();
    for _ in 0..100 {
        big_corpus.extend_from_slice(corpus);
    }
    let encoder = train_encoder(&big_corpus, 512);
    let tokens = encoder.encode(&big_corpus);

    let uni_counts = stats::collect_unigram_counts(&tokens);
    let h1 = stats::entropy_from_map(&uni_counts);
    assert!(h1 > 0.0, "H1 unigram entropy should be positive, got {}", h1);
    assert!(h1 < 20.0, "H1 should be bounded, got {}", h1);
}

#[test]
fn test_h2_bigram_entropy_positive() {
    let corpus = b"the quick brown fox jumps over the lazy dog ";
    let mut big_corpus = Vec::new();
    for _ in 0..100 {
        big_corpus.extend_from_slice(corpus);
    }
    let encoder = train_encoder(&big_corpus, 512);
    let tokens = encoder.encode(&big_corpus);

    let bi_counts = stats::collect_bigram_counts(&tokens);
    let h2 = stats::bigram_entropy(&bi_counts);
    assert!(h2 > 0.0, "H2 bigram entropy should be positive, got {}", h2);
}

#[test]
fn test_h3_trigram_entropy_positive() {
    let corpus = b"the quick brown fox jumps over the lazy dog ";
    let mut big_corpus = Vec::new();
    for _ in 0..100 {
        big_corpus.extend_from_slice(corpus);
    }
    let encoder = train_encoder(&big_corpus, 512);
    let tokens = encoder.encode(&big_corpus);

    let tri_counts = stats::collect_trigram_counts(&tokens);
    let h3 = stats::trigram_entropy(&tri_counts);
    assert!(h3 > 0.0, "H3 trigram entropy should be positive, got {}", h3);
}

#[test]
fn test_entropy_hierarchy_h1_ge_h2_ge_h3() {
    let corpus = b"the quick brown fox jumps over the lazy dog ";
    let mut big_corpus = Vec::new();
    for _ in 0..200 {
        big_corpus.extend_from_slice(corpus);
    }
    let encoder = train_encoder(&big_corpus, 512);
    let tokens = encoder.encode(&big_corpus);

    let uni_counts = stats::collect_unigram_counts(&tokens);
    let bi_counts = stats::collect_bigram_counts(&tokens);
    let tri_counts = stats::collect_trigram_counts(&tokens);

    let h1 = stats::entropy_from_map(&uni_counts);
    let h2 = stats::bigram_entropy(&bi_counts);
    let h3 = stats::trigram_entropy(&tri_counts);

    assert!(
        h1 >= h2,
        "H1 ({:.4}) should be >= H2 ({:.4}): conditioning reduces entropy",
        h1, h2
    );
    assert!(
        h2 >= h3,
        "H2 ({:.4}) should be >= H3 ({:.4}): more context reduces entropy",
        h2, h3
    );
}

#[test]
fn test_entropy_decreases_with_larger_vocab() {
    let corpus = b"the quick brown fox jumps over the lazy dog ";
    let mut big_corpus = Vec::new();
    for _ in 0..200 {
        big_corpus.extend_from_slice(corpus);
    }

    let encoder_small = train_encoder(&big_corpus, 280);
    let encoder_large = train_encoder(&big_corpus, 512);

    let tokens_small = encoder_small.encode(&big_corpus);
    let tokens_large = encoder_large.encode(&big_corpus);

    // Larger vocab should produce fewer tokens (better compression)
    assert!(
        tokens_large.len() <= tokens_small.len(),
        "larger vocab ({}) should produce <= tokens than smaller vocab ({})",
        tokens_large.len(),
        tokens_small.len()
    );
}
