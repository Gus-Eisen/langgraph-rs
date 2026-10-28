//! Reducers fixed on a channel when its schema is defined.

use serde_json::Value;

use crate::error::Error;

/// Fold used by [`Reducer::Custom`].
///
/// `current` is `None` when the channel has no committed value. A step folds
/// left to right: the value returned for one write is `current` for the next
/// write. Tasks do not see each other's writes; the fold starts from the
/// committed value.
pub type CustomReducer = fn(current: Option<&Value>, update: &Value) -> Result<Value, Error>;

/// How a channel combines writes at the end of a step.
///
/// The kind is fixed on the schema. It is not chosen per write.
///
/// This enum does not implement `PartialEq`. [`Reducer::Custom`] holds a
/// function pointer, and this toolchain rejects function-pointer equality.
#[derive(Debug, Clone, Copy)]
pub enum Reducer {
    /// At most one write per step. That write replaces the committed value.
    ///
    /// Zero writes leave the channel unchanged. Two or more writes in the
    /// same step are [`Error::InvalidUpdate`](Error::InvalidUpdate).
    Overwrite,
    /// Concatenate JSON arrays in task order.
    ///
    /// The committed value, when present, and every write must be JSON
    /// arrays. Each write is the elements to add, not a replacement list.
    /// A missing committed value is treated as an empty array. Any other
    /// JSON type is [`Error::InvalidUpdate`](Error::InvalidUpdate).
    Append,
    /// Caller-supplied left fold. See [`CustomReducer`].
    Custom(CustomReducer),
}

/// One state field: a channel name and its reducer.
///
/// `name` is a static string so a [`crate::GraphState`] schema can live in a
/// constant. Names are checked when the channel map is built.
#[derive(Debug, Clone, Copy)]
pub struct Field {
    /// Channel name. Compared exactly, including underscores.
    pub name: &'static str,
    /// Reducer used for every step that writes this channel.
    pub reducer: Reducer,
}
