//! Text filters that can be applied before/after tokenization.
//! No filters are enabled by default — users opt in.

/// A filter that transforms text before tokenization or token IDs after.
pub trait PreTokenFilter: Send + Sync {
    fn name(&self) -> &str;
    fn apply(&self, input: &[u8]) -> Vec<u8>;
}

/// Built-in filter: normalize Unicode whitespace to ASCII spaces
pub struct NormalizeWhitespace;

impl PreTokenFilter for NormalizeWhitespace {
    fn name(&self) -> &str {
        "normalize_whitespace"
    }
    fn apply(&self, input: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(input.len());
        let mut prev_space = false;
        for &b in input {
            if b == b' ' || b == b'\t' || b == b'\r' || b == b'\x0b' || b == b'\x0c' {
                if !prev_space {
                    out.push(b' ');
                    prev_space = true;
                }
            } else {
                out.push(b);
                prev_space = false;
            }
        }
        out
    }
}

/// Built-in filter: lowercase ASCII
pub struct LowercaseAscii;

impl PreTokenFilter for LowercaseAscii {
    fn name(&self) -> &str {
        "lowercase_ascii"
    }
    fn apply(&self, input: &[u8]) -> Vec<u8> {
        input.iter().map(|&b| {
            if b.is_ascii_uppercase() {
                b + 32
            } else {
                b
            }
        }).collect()
    }
}

/// Built-in filter: strip non-printable bytes (keep 0x20-0x7E, newline, tab)
pub struct StripNonPrintable;

impl PreTokenFilter for StripNonPrintable {
    fn name(&self) -> &str {
        "strip_non_printable"
    }
    fn apply(&self, input: &[u8]) -> Vec<u8> {
        input.iter().copied().filter(|&b| {
            b >= 0x20 || b == b'\n' || b == b'\t'
        }).collect()
    }
}

/// Filter pipeline: applies a chain of filters in order
pub struct FilterPipeline {
    filters: Vec<Box<dyn PreTokenFilter>>,
}

impl FilterPipeline {
    pub fn new() -> Self {
        FilterPipeline { filters: Vec::new() }
    }

    pub fn add(&mut self, filter: Box<dyn PreTokenFilter>) {
        self.filters.push(filter);
    }

    pub fn apply(&self, input: &[u8]) -> Vec<u8> {
        let mut data = input.to_vec();
        for f in &self.filters {
            data = f.apply(&data);
        }
        data
    }

    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    pub fn filter_names(&self) -> Vec<&str> {
        self.filters.iter().map(|f| f.name()).collect()
    }
}

impl Default for FilterPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_whitespace() {
        let f = NormalizeWhitespace;
        assert_eq!(f.apply(b"hello   world"), b"hello world");
        assert_eq!(f.apply(b"a\t\tb"), b"a b");
    }

    #[test]
    fn test_lowercase_ascii() {
        let f = LowercaseAscii;
        assert_eq!(f.apply(b"Hello WORLD"), b"hello world");
        assert_eq!(f.apply(b"123"), b"123");
    }

    #[test]
    fn test_strip_non_printable() {
        let f = StripNonPrintable;
        assert_eq!(f.apply(b"hello\x00world\x01!"), b"helloworld!");
        assert_eq!(f.apply(b"keep\nnewlines\ttabs"), b"keep\nnewlines\ttabs");
    }

    #[test]
    fn test_pipeline() {
        let mut pipeline = FilterPipeline::new();
        pipeline.add(Box::new(LowercaseAscii));
        pipeline.add(Box::new(NormalizeWhitespace));
        assert_eq!(pipeline.apply(b"Hello   WORLD"), b"hello world");
        assert_eq!(pipeline.filter_names(), vec!["lowercase_ascii", "normalize_whitespace"]);
    }

    #[test]
    fn test_empty_pipeline() {
        let pipeline = FilterPipeline::new();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.apply(b"unchanged"), b"unchanged");
    }
}
