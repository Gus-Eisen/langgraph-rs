//! Library errors.
//!
//! Channel folds return [`Error::InvalidUpdate`]. Schema checks and
//! [`crate::GraphBuilder::compile`] return [`Error::GraphValidation`].
//! [`crate::CompiledGraph::invoke`] adds [`Error::UnknownNode`],
//! [`Error::RecursionLimit`], [`Error::NoRuntime`], and [`Error::Node`].
//! A configured checkpointer adds [`Error::MissingThreadId`] and
//! [`Error::Checkpoint`]. [`Error::GraphInterrupt`] is how [`crate::interrupt`]
//! leaves a node; `invoke` does not return it.
//! [`Error::ParentGraph`] rejects [`crate::GraphTarget::Parent`] until a
//! subgraph can route to its parent.

use thiserror::Error;

/// Failure returned by state-channel operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Error {
    /// A step's writes could not be folded into a channel.
    ///
    /// Overwrite channels reject a second write in the same step. Append
    /// channels reject a value that is not a JSON array. `AddMessages`
    /// rejects a message list that cannot be merged, including a `remove`
    /// of an unknown id. A custom reducer may return this variant itself.
    #[error("invalid update: {0}")]
    InvalidUpdate(String),

    /// The graph or its channel schema is not usable.
    ///
    /// Empty names, duplicate names, and reserved names are rejected when a
    /// channel map is built. [`crate::GraphBuilder::compile`] rejects those
    /// names on nodes and fields, and rejects edges that do not connect
    /// [`crate::START`], registered nodes, and [`crate::END`]. Reading a map
    /// with a different [`crate::GraphState`] schema fails the same way.
    /// A recursion limit below 1 is the same variant.
    #[error("graph validation: {0}")]
    GraphValidation(String),

    /// A router returned a name that is not [`crate::END`] and not a node.
    ///
    /// With a static path map, the router's label is mapped first. A label
    /// the map does not contain is this variant. Without a path map, the
    /// router is expected to return node names directly. A [`crate::Send`]
    /// names its node directly. The path map does not rewrite it. The
    /// superstep that produced the route does not commit.
    #[error("unknown node: {0}")]
    UnknownNode(String),

    /// The run would start a superstep past the recursion limit.
    ///
    /// The value is the limit. The refused step is not executed, and the
    /// failed step's writes are not committed. Input writes are not a
    /// superstep and do not count against this limit.
    #[error("recursion limit of {0} supersteps reached")]
    RecursionLimit(u32),

    /// [`crate::CompiledGraph::invoke`] was polled outside a tokio runtime.
    #[error("no tokio runtime is available")]
    NoRuntime,

    /// A node returned [`Err`](Result::Err), or its task panicked and was
    /// cancelled.
    ///
    /// Sibling tasks in the superstep are joined first. None of that step's
    /// writes are committed.
    #[error("node '{node}' failed: {message}")]
    Node { node: String, message: String },

    /// A checkpointer is required and the run cannot use one.
    ///
    /// A configured checkpointer with no thread id is this error, checked
    /// before any node runs and before any checkpoint read. With no
    /// checkpointer, `thread_id` is ignored.
    ///
    /// [`crate::interrupt`] without a checkpointer is also this error. The
    /// superstep does not commit. `interrupt_before` / `interrupt_after`
    /// without a checkpointer are this error before any node runs.
    #[error("missing thread id")]
    MissingThreadId,

    /// The checkpointer failed, or the requested checkpoint does not exist.
    #[error("checkpoint: {0}")]
    Checkpoint(String),

    /// [`crate::interrupt`] paused the node that returned this error.
    ///
    /// Node functions propagate it with `?`. [`crate::CompiledGraph::invoke`]
    /// does not return it: a pause is
    /// [`Ok`](Result::Ok)`(`[`crate::RunOutput::Interrupted`]`)`.
    #[error("graph interrupt")]
    GraphInterrupt,

    /// [`crate::Command`] set `graph` to [`crate::GraphTarget::Parent`].
    ///
    /// Parent routing requires a subgraph. `invoke` returns this error and
    /// does not commit that command. A node return fails its superstep
    /// before the barrier: sibling tasks are joined, and that step's writes
    /// are not stored. An input command fails before any checkpoint write.
    #[error("parent routing requires a subgraph")]
    ParentGraph,
}
