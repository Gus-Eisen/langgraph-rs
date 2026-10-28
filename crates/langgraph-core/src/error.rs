//! Library errors.
//!
//! Channel folds return [`Error::InvalidUpdate`]. Schema checks and
//! [`crate::GraphBuilder::compile`] return [`Error::GraphValidation`].
//! Later milestones add variants to this enum.

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
    #[error("graph validation: {0}")]
    GraphValidation(String),
}
