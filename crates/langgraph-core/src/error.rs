//! Library errors for state channels.
//!
//! Later milestones add variants to this enum. This milestone only returns
//! the ones the reducer and schema checks produce.

use thiserror::Error;

/// Failure returned by state-channel operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Error {
    /// A step's writes could not be folded into a channel.
    ///
    /// Overwrite channels reject a second write in the same step. Append
    /// channels reject a value that is not a JSON array. A custom reducer
    /// may return this variant itself.
    #[error("invalid update: {0}")]
    InvalidUpdate(String),

    /// The channel schema is not usable.
    ///
    /// Empty names, duplicate names, and reserved names are rejected when a
    /// channel map is built. Reading a map with a different [`crate::GraphState`]
    /// schema fails the same way.
    #[error("graph validation: {0}")]
    GraphValidation(String),
}
