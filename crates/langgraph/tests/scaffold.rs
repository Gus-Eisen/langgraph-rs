//! Smoke test for the workspace facade.
//!
//! The graph builder is covered in `builder.rs`. This checks that the facade
//! links the crates named in `ARCHITECTURE.md`.

#[test]
fn workspace_facade_links_planned_crates() {
    assert_eq!(langgraph::VERSION, env!("CARGO_PKG_VERSION"));
    assert_eq!(langgraph::LINKED_CRATES.len(), 4);
    for (name, version) in langgraph::LINKED_CRATES {
        assert!(!name.is_empty());
        assert_eq!(*version, langgraph::VERSION);
    }
}
