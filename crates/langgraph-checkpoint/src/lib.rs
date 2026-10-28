//! Checkpoint storage for langgraph-rs.
//!
//! [`Checkpointer`] is the saver protocol. [`InMemorySaver`] keeps one mutex
//! map on the saver value. Channel values are JSON. This crate does not
//! depend on another workspace crate; the runtime converts state at the
//! boundary. SQLite is a later milestone.

#![forbid(unsafe_code)]

mod memory;
mod time;
mod types;

pub use memory::InMemorySaver;
pub use types::{
    checkpoint_timestamp, new_checkpoint_id, BoxFuture, Checkpoint, CheckpointConfig,
    CheckpointMetadata, CheckpointSource, CheckpointTuple, PendingWrite,
};

use std::collections::BTreeMap;

/// Failure from a [`Checkpointer`] implementation.
///
/// The runtime maps this to its own checkpoint error. Callers that use a
/// saver directly match this type.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// `config.checkpoint_id` is not in the thread and namespace.
    #[error("unknown checkpoint id '{0}'")]
    UnknownCheckpoint(String),

    /// The saver rejected the write or could not read its storage.
    #[error("{0}")]
    Storage(String),
}

/// Object-safe checkpoint store.
///
/// Futures are boxed so a compiled graph can hold `Arc<dyn Checkpointer>`.
/// There is no `async-trait` dependency. Implementations may complete the
/// future immediately, as [`InMemorySaver`] does.
///
/// `checkpoint_ns` is empty at the root. A namespace is a separate list on
/// the same thread. `get_tuple` with no `checkpoint_id` loads the latest
/// checkpoint in that thread and namespace. `list` returns newest first.
pub trait Checkpointer: Send + Sync {
    /// Load one checkpoint.
    ///
    /// `Ok(None)` means this thread and namespace have no checkpoints and
    /// `config.checkpoint_id` was not set. A `checkpoint_id` that is not in
    /// the namespace is [`Error::UnknownCheckpoint`].
    fn get_tuple<'a>(
        &'a self,
        config: &'a CheckpointConfig,
    ) -> BoxFuture<'a, Result<Option<CheckpointTuple>, Error>>;

    /// Checkpoints in `config`'s thread and namespace, newest first.
    ///
    /// `before` is a checkpoint id. The result is the checkpoints stored
    /// strictly earlier than that id. `limit` keeps the first entries of
    /// that newest-first list. `config.checkpoint_id` is not a filter.
    /// An unknown `before` id is [`Error::UnknownCheckpoint`].
    fn list<'a>(
        &'a self,
        config: &'a CheckpointConfig,
        before: Option<&'a str>,
        limit: Option<usize>,
    ) -> BoxFuture<'a, Result<Vec<CheckpointTuple>, Error>>;

    /// Store `checkpoint` and return a config that points at it.
    ///
    /// The parent is `config.checkpoint_id` when that is set, otherwise the
    /// latest checkpoint in the thread and namespace. The selected parent is
    /// not required to be the latest. Older checkpoints are left as they
    /// were. `metadata.parents` is replaced with the parent checkpoint's
    /// parent map plus this namespace mapped to the parent id.
    ///
    /// `new_versions` is the channel-version delta the runtime just wrote.
    /// Each entry must match `checkpoint.channel_versions`.
    fn put<'a>(
        &'a self,
        config: &'a CheckpointConfig,
        checkpoint: &'a Checkpoint,
        metadata: &'a CheckpointMetadata,
        new_versions: &'a BTreeMap<String, u64>,
    ) -> BoxFuture<'a, Result<CheckpointConfig, Error>>;

    /// Append writes for `task_id` onto the checkpoint `config` names.
    ///
    /// `config.checkpoint_id` is required. The writes are not folded into
    /// channel values. Each stored write records `task_id`.
    fn put_writes<'a>(
        &'a self,
        config: &'a CheckpointConfig,
        writes: &'a [PendingWrite],
        task_id: &'a str,
    ) -> BoxFuture<'a, Result<(), Error>>;

    /// Drop every namespace stored for `thread_id`.
    ///
    /// A thread that was never stored is success.
    fn delete_thread<'a>(&'a self, thread_id: &'a str) -> BoxFuture<'a, Result<(), Error>>;
}

/// Version of `langgraph-checkpoint`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod suite;

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_set() {
        assert!(!crate::VERSION.is_empty());
    }

    #[test]
    fn saver_is_send_and_sync() {
        fn assert_bounds<T: Send + Sync>() {}
        assert_bounds::<crate::InMemorySaver>();
        let saver = crate::InMemorySaver::new();
        let _object: &dyn crate::Checkpointer = &saver;
    }
}
