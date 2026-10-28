//! LangGraph for Rust.
//!
//! This crate is the public facade. It re-exports state channels, reducers,
//! `add_messages`, the graph builder, Pregel `invoke`, and the checkpointers.
//! `compile` checks a graph. `CompiledGraph::invoke` runs it. `InMemorySaver`
//! keeps a thread in memory. `SqliteSaver::open` keeps the same checkpoints
//! in a SQLite file. The prebuilt agent is a later milestone. See the
//! repository `ROADMAP.md` and `ARCHITECTURE.md`.

#![forbid(unsafe_code)]

pub use langgraph_checkpoint::{
    checkpoint_timestamp, new_checkpoint_id, BoxFuture, Checkpoint, CheckpointConfig,
    CheckpointMetadata, CheckpointSource, CheckpointTuple, Checkpointer, Error as CheckpointError,
    InMemorySaver, PendingWrite, SqliteSaver,
};
pub use langgraph_core::{
    add_messages, AsyncNodeMarker, ChannelMap, CompiledGraph, CustomReducer, EdgeSource, Error,
    Field, GraphBuilder, GraphState, IntoEdgeSource, IntoRoute, PendingTask, Reducer, Route,
    RunConfig, StateSnapshot, SyncNodeMarker, Write, DEFAULT_RECURSION_LIMIT, END,
    REMOVE_ALL_MESSAGES, RESERVED_NAMES, START,
};

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

    #[test]
    fn facade_reexports_the_message_reducer() {
        assert_eq!(crate::REMOVE_ALL_MESSAGES, "__remove_all__");
        let _add_messages = crate::add_messages;
        let _kind = crate::Reducer::AddMessages;
    }

    #[test]
    fn facade_reexports_the_builder_sentinels() {
        assert_eq!(crate::START, "__start__");
        assert_eq!(crate::END, "__end__");
        assert!(crate::RESERVED_NAMES.contains(&crate::START));
        assert!(crate::RESERVED_NAMES.contains(&crate::END));
        assert_eq!(crate::Route::end().destinations(), ["__end__".to_string()]);
    }

    #[test]
    fn facade_reexports_the_sqlite_saver() {
        fn assert_saver<T: crate::Checkpointer>() {}
        assert_saver::<crate::InMemorySaver>();
        assert_saver::<crate::SqliteSaver>();
    }

    #[test]
    fn facade_reexports_the_recursion_limit() {
        assert_eq!(crate::DEFAULT_RECURSION_LIMIT, 25);
        assert_eq!(crate::RunConfig::default(), crate::RunConfig::new());
        assert_ne!(
            crate::RunConfig::new().recursion_limit(1),
            crate::RunConfig::default()
        );
    }
}
