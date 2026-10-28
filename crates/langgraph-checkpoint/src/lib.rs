//! Checkpoint storage for langgraph-rs.
//!
//! The `Checkpointer` trait, in-memory saver, and SQLite saver are added by
//! later milestones. This crate currently exposes only its version. See the
//! repository `ARCHITECTURE.md`.

#![forbid(unsafe_code)]

/// Version of `langgraph-checkpoint`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_set() {
        assert!(!crate::VERSION.is_empty());
    }
}
