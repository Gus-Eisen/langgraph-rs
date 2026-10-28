//! LangGraph for Rust.
//!
//! This crate is the public facade. Milestone 0 links the workspace crates and
//! does not expose a graph API. Types for state, execution, checkpoints, and
//! the prebuilt agent land in later milestones. See the repository
//! `ROADMAP.md` and `ARCHITECTURE.md`.

#![forbid(unsafe_code)]

/// Version of this facade crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Workspace crates this facade is wired to.
///
/// The pairs are `(crate name, crate version)`. They record the scaffold's
/// dependency graph. They are not a graph-execution API.
pub const LINKED_CRATES: &[(&str, &str)] = &[
    ("langgraph-core", langgraph_core::VERSION),
    ("langgraph-checkpoint", langgraph_checkpoint::VERSION),
    ("langgraph-llm", langgraph_llm::VERSION),
    ("langgraph-prebuilt", langgraph_prebuilt::VERSION),
];

#[cfg(test)]
mod tests {
    #[test]
    fn facade_links_every_workspace_crate() {
        let names: Vec<&str> = crate::LINKED_CRATES.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            names,
            [
                "langgraph-core",
                "langgraph-checkpoint",
                "langgraph-llm",
                "langgraph-prebuilt",
            ]
        );
        assert!(crate::LINKED_CRATES
            .iter()
            .all(|(_, version)| !version.is_empty()));
    }
}
