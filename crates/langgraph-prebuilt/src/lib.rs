//! Prebuilt graphs for langgraph-rs.
//!
//! The ReAct tool-calling loop is added by a later milestone. This crate
//! currently records the core and LLM crates it will build on. See the
//! repository `ARCHITECTURE.md`.

#![forbid(unsafe_code)]

/// Version of `langgraph-prebuilt`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Version of `langgraph-core` linked into this crate.
pub const CORE_VERSION: &str = langgraph_core::VERSION;

/// Version of `langgraph-llm` linked into this crate.
pub const LLM_VERSION: &str = langgraph_llm::VERSION;

#[cfg(test)]
mod tests {
    #[test]
    fn linked_crates_share_the_workspace_version() {
        assert_eq!(crate::VERSION, crate::CORE_VERSION);
        assert_eq!(crate::VERSION, crate::LLM_VERSION);
    }
}
