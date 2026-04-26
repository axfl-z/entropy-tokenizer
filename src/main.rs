use clap::{Parser, Subcommand};
use std::time::Instant;

use entropy_tokenizer::encoder::Encoder;
use entropy_tokenizer::stats;
use entropy_tokenizer::trainer::Trainer;
use entropy_tokenizer::vocab::Vocabulary;

#[derive(Parser)]
#[command(name = "eot", about = "Entropy-Optimal Tokenizer")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a tokenizer on a text corpus
    Train {
        /// Input corpus file path
        #[arg(short, long)]
        input: String,

        /// Target vocabulary size
        #[arg(short, long, default_value = "8192")]
        vocab_size: usize,

        /// Output model file path
        #[arg(short, long, default_value = "model.json")]
        output: String,

        /// Print training progress
        #[arg(long, default_value = "false")]
        verbose: bool,
    },

    /// Encode text into token IDs
    Encode {
        /// Model file path
        #[arg(short, long)]
        model: String,

        /// Text to encode (or --file for file input)
        #[arg(short, long)]
        text: Option<String>,

        /// File to encode
        #[arg(short, long)]
        file: Option<String>,

        /// Use greedy encoding instead of DP-optimal
        #[arg(long, default_value = "false")]
        greedy: bool,

        /// Context weight for bigram scoring (0.0 = no context)
        #[arg(long, default_value = "0.3")]
        context_weight: f64,
    },

    /// Decode token IDs back to text
    Decode {
        /// Model file path
        #[arg(short, long)]
        model: String,

        /// Comma-separated token IDs
        #[arg(short, long)]
        tokens: String,
    },

    /// Benchmark tokenizer on a file
    Bench {
        /// Model file path
        #[arg(short, long)]
        model: String,

        /// Input file to benchmark on
        #[arg(short, long)]
        input: String,

        /// Context weight for bigram scoring
        #[arg(long, default_value = "0.3")]
        context_weight: f64,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            input,
            vocab_size,
            output,
            verbose,
        } => cmd_train(&input, vocab_size, &output, verbose),

        Commands::Encode {
            model,
            text,
            file,
            greedy,
            context_weight,
        } => cmd_encode(&model, text, file, greedy, context_weight),

        Commands::Decode { model, tokens } => cmd_decode(&model, &tokens),

        Commands::Bench {
            model,
            input,
            context_weight,
        } => cmd_bench(&model, &input, context_weight),
    }
}

fn cmd_train(input_path: &str, vocab_size: usize, output_path: &str, verbose: bool) {
    eprintln!("Loading corpus from: {}", input_path);
    let corpus = std::fs::read(input_path).expect("Failed to read input corpus");
    eprintln!(
        "Corpus size: {} bytes ({:.2} MB)",
        corpus.len(),
        corpus.len() as f64 / 1_048_576.0
    );

    let start = Instant::now();
    let mut trainer = Trainer::new(&corpus);
    trainer.train(vocab_size, verbose);

    let elapsed = start.elapsed();

    let (total_tokens, entropy, bigram_h) = trainer.corpus_stats();
    let vocab = trainer.into_vocab();

    eprintln!("Training completed in {:.2}s", elapsed.as_secs_f64());
    eprintln!("Final vocabulary size: {}", vocab.vocab_size);

    let original_len = corpus.len();
    let cr = stats::compression_ratio(original_len, total_tokens as usize);

    eprintln!("--- Corpus Statistics ---");
    eprintln!("Original size:       {} bytes", original_len);
    eprintln!("Token count:         {}", total_tokens);
    eprintln!("Compression ratio:   {:.2} bytes/token", cr);
    eprintln!("Unigram entropy:     {:.4} bits", entropy);
    eprintln!("Bigram entropy:      {:.4} bits", bigram_h);

    vocab.save(output_path).expect("Failed to save model");
    eprintln!("Model saved to: {}", output_path);
}

fn cmd_encode(
    model_path: &str,
    text: Option<String>,
    file: Option<String>,
    greedy: bool,
    context_weight: f64,
) {
    let vocab = Vocabulary::load(model_path).expect("Failed to load model");
    let encoder = Encoder::new(vocab, context_weight);

    let input_bytes = match (text, file) {
        (Some(t), _) => t.into_bytes(),
        (_, Some(f)) => std::fs::read(&f).expect("Failed to read input file"),
        _ => {
            eprintln!("Error: provide --text or --file");
            std::process::exit(1);
        }
    };

    let start = Instant::now();
    let tokens = if greedy {
        encoder.encode_greedy(&input_bytes)
    } else {
        encoder.encode(&input_bytes)
    };
    let elapsed = start.elapsed();

    let mode = if greedy { "greedy" } else { "dp-optimal" };
    eprintln!(
        "Encoded {} bytes -> {} tokens ({} mode, {:.2}ms)",
        input_bytes.len(),
        tokens.len(),
        mode,
        elapsed.as_secs_f64() * 1000.0
    );
    eprintln!(
        "Compression ratio: {:.2} bytes/token",
        stats::compression_ratio(input_bytes.len(), tokens.len())
    );

    let token_strs: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
    println!("{}", token_strs.join(","));

    eprintln!("\n--- Token Details ---");
    for &tid in &tokens {
        let bytes = encoder.vocab().get_bytes(tid);
        let display = String::from_utf8_lossy(bytes);
        eprintln!("  [{}] {:?} ({}B)", tid, display, bytes.len());
    }
}

fn cmd_decode(model_path: &str, tokens_str: &str) {
    let vocab = Vocabulary::load(model_path).expect("Failed to load model");
    let encoder = Encoder::new(vocab, 0.0);

    let token_ids: Vec<u32> = tokens_str
        .split(',')
        .map(|s| s.trim().parse::<u32>().expect("Invalid token ID"))
        .collect();

    let text = encoder.decode_to_string(&token_ids);
    println!("{}", text);
}

fn cmd_bench(model_path: &str, input_path: &str, context_weight: f64) {
    let vocab = Vocabulary::load(model_path).expect("Failed to load model");
    let encoder = Encoder::new(vocab, context_weight);

    let input = std::fs::read(input_path).expect("Failed to read input file");
    let input_len = input.len();
    eprintln!(
        "Benchmarking on {} bytes ({:.2} MB)",
        input_len,
        input_len as f64 / 1_048_576.0
    );

    let token_entropy = |tokens: &[u32]| -> f64 {
        let mut freq: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
        for &t in tokens {
            *freq.entry(t).or_insert(0) += 1;
        }
        stats::entropy_from_map(&freq)
    };

    // Dense (minimum token count) encoding
    let start = Instant::now();
    let dense_tokens = encoder.encode_dense(&input);
    let dense_elapsed = start.elapsed();
    let dense_decoded = encoder.decode(&dense_tokens);
    let dense_roundtrip = dense_decoded == input;
    let dense_entropy = token_entropy(&dense_tokens);

    // DP-Optimal (entropy-aware) encoding
    let start = Instant::now();
    let dp_tokens = encoder.encode(&input);
    let dp_elapsed = start.elapsed();
    let dp_decoded = encoder.decode(&dp_tokens);
    let dp_roundtrip = dp_decoded == input;
    let dp_entropy = token_entropy(&dp_tokens);

    // Greedy encoding
    let start = Instant::now();
    let greedy_tokens = encoder.encode_greedy(&input);
    let greedy_elapsed = start.elapsed();
    let greedy_decoded = encoder.decode(&greedy_tokens);
    let greedy_roundtrip = greedy_decoded == input;
    let greedy_entropy = token_entropy(&greedy_tokens);

    println!("=== Benchmark Results ===");
    println!("Input: {} ({} bytes)", input_path, input_len);

    println!();
    println!("--- Dense Encoding (min tokens) ---");
    println!("  Tokens:            {}", dense_tokens.len());
    println!(
        "  Compression ratio: {:.2} bytes/token",
        stats::compression_ratio(input_len, dense_tokens.len())
    );
    println!("  Entropy:           {:.4} bits", dense_entropy);
    println!(
        "  Speed:             {:.2} ms ({:.2} MB/s)",
        dense_elapsed.as_secs_f64() * 1000.0,
        input_len as f64 / 1_048_576.0 / dense_elapsed.as_secs_f64()
    );
    println!("  Roundtrip OK:      {}", dense_roundtrip);

    println!();
    println!("--- DP-Optimal Encoding ---");
    println!("  Tokens:            {}", dp_tokens.len());
    println!(
        "  Compression ratio: {:.2} bytes/token",
        stats::compression_ratio(input_len, dp_tokens.len())
    );
    println!("  Entropy:           {:.4} bits", dp_entropy);
    println!(
        "  Speed:             {:.2} ms ({:.2} MB/s)",
        dp_elapsed.as_secs_f64() * 1000.0,
        input_len as f64 / 1_048_576.0 / dp_elapsed.as_secs_f64()
    );
    println!("  Roundtrip OK:      {}", dp_roundtrip);

    println!();
    println!("--- Greedy Encoding ---");
    println!("  Tokens:            {}", greedy_tokens.len());
    println!(
        "  Compression ratio: {:.2} bytes/token",
        stats::compression_ratio(input_len, greedy_tokens.len())
    );
    println!("  Entropy:           {:.4} bits", greedy_entropy);
    println!(
        "  Speed:             {:.2} ms ({:.2} MB/s)",
        greedy_elapsed.as_secs_f64() * 1000.0,
        input_len as f64 / 1_048_576.0 / greedy_elapsed.as_secs_f64()
    );
    println!("  Roundtrip OK:      {}", greedy_roundtrip);

    println!();
    println!("--- Comparison ---");
    println!(
        "  Dense saves {} tokens ({:.2}% fewer) vs greedy",
        greedy_tokens.len() as i64 - dense_tokens.len() as i64,
        if greedy_tokens.is_empty() {
            0.0
        } else {
            (1.0 - dense_tokens.len() as f64 / greedy_tokens.len() as f64) * 100.0
        }
    );
    println!(
        "  DP saves {} tokens ({:.2}% fewer) vs greedy",
        greedy_tokens.len() as i64 - dp_tokens.len() as i64,
        if greedy_tokens.is_empty() {
            0.0
        } else {
            (1.0 - dp_tokens.len() as f64 / greedy_tokens.len() as f64) * 100.0
        }
    );
}
