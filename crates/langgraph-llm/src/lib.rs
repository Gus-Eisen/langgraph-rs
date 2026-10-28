//! LLM provider surface for langgraph-rs.
//!
//! `ChatModel`, the mock model, and the optional OpenAI feature are added by
//! a later milestone. This crate currently exposes only its version. See the
//! repository `ARCHITECTURE.md`.

#![forbid(unsafe_code)]

/// Version of `langgraph-llm`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_set() {
        assert!(!crate::VERSION.is_empty());
    }
}
