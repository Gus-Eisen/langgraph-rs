//! Library errors.
//!
//! Channel folds return [`Error::InvalidUpdate`]. Schema checks and
//! [`crate::GraphBuilder::compile`] return [`Error::GraphValidation`].
//! [`crate::CompiledGraph::invoke`] adds [`Error::UnknownNode`],
//! [`Error::RecursionLimit`], [`Error::NoRuntime`], and [`Error::Node`].
//! A configured checkpointer adds [`Error::MissingThreadId`] and
//! [`Error::Checkpoint`].

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
    /// router is expected to return node names directly.
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

    /// A checkpointer is configured and the run has no thread id.
    ///
    /// Checked before any node runs and before any checkpoint read. With no
    /// checkpointer, `thread_id` is ignored.
    #[error("missing thread id")]
    MissingThreadId,

    /// The checkpointer failed, or the requested checkpoint does not exist.
    #[error("checkpoint: {0}")]
    Checkpoint(String),
}
