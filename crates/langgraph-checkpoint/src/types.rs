//! Checkpoint values stored by a [`crate::Checkpointer`].

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::time::format_rfc3339;

/// Boxed future returned by [`crate::Checkpointer`].
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Where a checkpoint was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointSource {
    /// Input writes applied before the first superstep of an invoke.
    Input,
    /// A committed superstep.
    Loop,
    /// [`crate::Checkpointer::put`] from an external state update.
    Update,
    /// A checkpoint whose parent is not the previous tip.
    ///
    /// The runtime expresses a fork by the parent id. This variant is stored
    /// when a caller sets it on [`CheckpointMetadata::source`].
    Fork,
}

/// Identity of one checkpoint list.
///
/// `checkpoint_ns` is empty at the root graph. `checkpoint_id` selects one
/// snapshot. Without it, reads use the latest checkpoint in the thread and
/// namespace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Thread this checkpoint belongs to.
    pub thread_id: String,
    /// Namespace inside the thread. Empty at the root.
    pub checkpoint_ns: String,
    /// One snapshot. `None` means the latest on read, and no parent on write
    /// unless a latest checkpoint already exists.
    pub checkpoint_id: Option<String>,
}

/// Saved channels and the runtime's scheduling map.
///
/// `versions_seen` is node name to control key to version. Control keys are
/// `done:{node}` (including `done:__start__`) and `branch:{edge_index}:{dest}`.
/// `done` and `branch` are those maps without the key prefix split: `done`
/// is keyed by node name, and `branch` is keyed by `branch:{edge_index}:{dest}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Format version. This milestone writes `1`.
    pub v: u32,
    /// Time-ordered UUID v7.
    pub id: String,
    /// UTC timestamp, RFC3339 with millisecond precision.
    pub ts: String,
    /// Committed channel values. Unseen channels are omitted.
    pub channel_values: BTreeMap<String, Value>,
    /// Version of every channel the runtime stored, including unseen `0`s.
    pub channel_versions: BTreeMap<String, u64>,
    /// Node to control key to the version that node has consumed.
    pub versions_seen: BTreeMap<String, BTreeMap<String, u64>>,
    /// Channels written by the step that produced this checkpoint.
    pub updated_channels: Vec<String>,
    /// Completion counts keyed by node name, including `__start__`.
    pub done: BTreeMap<String, u64>,
    /// Conditional selections keyed by `branch:{edge_index}:{dest}`.
    pub branch: BTreeMap<String, u64>,
}

impl Checkpoint {
    /// Empty format-1 checkpoint with a new id and timestamp.
    ///
    /// The runtime fills the maps before [`crate::Checkpointer::put`].
    pub fn new() -> Self {
        Self {
            v: 1,
            id: new_checkpoint_id(),
            ts: checkpoint_timestamp(),
            channel_values: BTreeMap::new(),
            channel_versions: BTreeMap::new(),
            versions_seen: BTreeMap::new(),
            updated_channels: Vec::new(),
            done: BTreeMap::new(),
            branch: BTreeMap::new(),
        }
    }
}

impl Default for Checkpoint {
    fn default() -> Self {
        Self::new()
    }
}

/// How a checkpoint relates to the run that wrote it.
///
/// `parents` maps a checkpoint namespace to the parent checkpoint id. The
/// saver writes this map in [`crate::Checkpointer::put`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Why this checkpoint was written.
    pub source: CheckpointSource,
    /// `-1` for the first input checkpoint. The first loop checkpoint is `0`.
    pub step: i64,
    /// Namespace to parent checkpoint id.
    pub parents: BTreeMap<String, String>,
}

/// One stored checkpoint plus the config that addresses it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointTuple {
    /// Config whose `checkpoint_id` is this checkpoint.
    pub config: CheckpointConfig,
    /// Saved channels and scheduling map.
    pub checkpoint: Checkpoint,
    /// Source, step, and parents.
    pub metadata: CheckpointMetadata,
    /// Config of the parent checkpoint in this namespace, if it has one.
    pub parent_config: Option<CheckpointConfig>,
    /// Writes stored with [`crate::Checkpointer::put_writes`] and not yet
    /// folded into `checkpoint.channel_values`.
    pub pending_writes: Vec<PendingWrite>,
}

/// A channel write saved beside a checkpoint.
///
/// [`crate::Checkpointer::put_writes`] sets [`PendingWrite::task_id`] from
/// its `task_id` argument.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingWrite {
    /// Task that produced the write.
    pub task_id: String,
    /// Channel the write targets.
    pub channel: String,
    /// JSON value. The saver does not fold it.
    pub value: Value,
}

/// Time-ordered UUID v7 text, hyphenated and lowercase.
pub fn new_checkpoint_id() -> String {
    Uuid::now_v7().to_string()
}

/// Current UTC time as RFC3339 with millisecond precision.
pub fn checkpoint_timestamp() -> String {
    format_rfc3339(std::time::SystemTime::now())
}
