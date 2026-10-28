//! Graph state, reducers, the builder, and the Pregel runtime.
//!
//! Milestone 0 reserves this crate and links the tokio runtime the executor
//! will use. Channels, `GraphState`, validation, and superstep execution are
//! added by later milestones. See the repository `ARCHITECTURE.md`.

#![forbid(unsafe_code)]

/// Version of `langgraph-core`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Version of the checkpoint crate the runtime will call.
///
/// The checkpointer API is not re-exported here.
pub const CHECKPOINT_VERSION: &str = langgraph_checkpoint::VERSION;

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_set() {
        assert!(!crate::VERSION.is_empty());
        assert_eq!(crate::CHECKPOINT_VERSION, crate::VERSION);
    }

    #[tokio::test]
    async fn tokio_runtime_executes_a_task() {
        let value = tokio::spawn(async { 7 }).await.expect("join task");
        assert_eq!(value, 7);
    }
}
