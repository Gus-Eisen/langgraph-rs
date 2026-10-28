//! Streaming run output.
//!
//! [`crate::CompiledGraph::stream`] yields [`StreamEvent`]s while the run is
//! in progress. [`GraphStream::output`] is the same [`crate::RunOutput`]
//! `invoke` returns. Modes combine. A root run uses [`Namespace::root`].
//! There is no token stream and no user stream writer.

use std::ops::BitOr;

use langgraph_checkpoint::CheckpointConfig;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;

use crate::interrupt::{Interrupt, RunOutput};
use crate::Error;
use crate::GraphState;

/// Path of graph namespaces on a stream event.
///
/// A root run uses the empty path. A later nested run can extend it with
/// [`Namespace::join`]. Comparison is element by element.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Namespace {
    parts: Vec<String>,
}

impl Namespace {
    /// Empty path. This is the namespace of a root run.
    pub const fn root() -> Self {
        Self { parts: Vec::new() }
    }

    /// Path from these segments, in order.
    pub fn from_parts<I, T>(parts: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        Self {
            parts: parts.into_iter().map(Into::into).collect(),
        }
    }

    /// Segments of this path. Empty for [`Namespace::root`].
    pub fn parts(&self) -> &[String] {
        &self.parts
    }

    /// True when this is the root path.
    pub fn is_root(&self) -> bool {
        self.parts.is_empty()
    }

    /// This path plus `segment` at the end.
    pub fn join(&self, segment: impl Into<String>) -> Self {
        let mut parts = self.parts.clone();
        parts.push(segment.into());
        Self { parts }
    }
}

/// Which outputs [`crate::CompiledGraph::stream`] emits.
///
/// The three flags combine. [`StreamModes::values`], [`StreamModes::updates`],
/// and [`StreamModes::events`] each set one flag. [`BitOr`] unions them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamModes {
    values: bool,
    updates: bool,
    events: bool,
}

impl StreamModes {
    /// Full state after each committed superstep, and the state at an interrupt.
    pub const fn values() -> Self {
        Self {
            values: true,
            updates: false,
            events: false,
        }
    }

    /// One item per task that wrote, at the barrier, in task order.
    pub const fn updates() -> Self {
        Self {
            values: false,
            updates: true,
            events: false,
        }
    }

    /// Task, checkpoint, and interrupt events.
    pub const fn events() -> Self {
        Self {
            values: false,
            updates: false,
            events: true,
        }
    }

    /// Values, updates, and events together.
    pub const fn all() -> Self {
        Self {
            values: true,
            updates: true,
            events: true,
        }
    }

    /// Union of two mode sets.
    pub const fn combine(self, other: Self) -> Self {
        Self {
            values: self.values || other.values,
            updates: self.updates || other.updates,
            events: self.events || other.events,
        }
    }

    pub(crate) const fn wants_values(self) -> bool {
        self.values
    }

    pub(crate) const fn wants_updates(self) -> bool {
        self.updates
    }

    pub(crate) const fn wants_events(self) -> bool {
        self.events
    }
}

impl Default for StreamModes {
    /// No modes. The stream still runs and [`GraphStream::output`] matches `invoke`.
    fn default() -> Self {
        Self {
            values: false,
            updates: false,
            events: false,
        }
    }
}

impl BitOr for StreamModes {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        self.combine(rhs)
    }
}

/// One stream mode.
///
/// Convert with [`StreamModes::from`] or combine with [`BitOr`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// [`StreamModes::values`].
    Values,
    /// [`StreamModes::updates`].
    Updates,
    /// [`StreamModes::events`].
    Events,
}

impl From<StreamMode> for StreamModes {
    fn from(mode: StreamMode) -> Self {
        match mode {
            StreamMode::Values => StreamModes::values(),
            StreamMode::Updates => StreamModes::updates(),
            StreamMode::Events => StreamModes::events(),
        }
    }
}

impl BitOr for StreamMode {
    type Output = StreamModes;

    fn bitor(self, rhs: Self) -> StreamModes {
        StreamModes::from(self) | StreamModes::from(rhs)
    }
}

impl BitOr<StreamModes> for StreamMode {
    type Output = StreamModes;

    fn bitor(self, rhs: StreamModes) -> StreamModes {
        StreamModes::from(self) | rhs
    }
}

impl BitOr<StreamMode> for StreamModes {
    type Output = StreamModes;

    fn bitor(self, rhs: StreamMode) -> StreamModes {
        self | StreamModes::from(rhs)
    }
}

/// How a task finished, carried on [`StreamEvent::TaskFinished`].
///
/// `Update` is the node's update. `None` means the task wrote no channels.
/// `Error` is the task failure. A pause inside the task is
/// [`Error::GraphInterrupt`] here; the run then emits
/// [`StreamEvent::Interrupted`] and [`GraphStream::output`] is
/// [`crate::RunOutput::Interrupted`], not that error.
#[derive(Debug, Clone, PartialEq)]
pub enum TaskResult<U> {
    /// The task returned this update.
    Update(Option<U>),
    /// The task failed. That superstep does not commit.
    Error(Error),
}

/// One item from [`crate::CompiledGraph::stream`].
///
/// Every variant carries a [`Namespace`]. A root run uses the empty path.
/// `S` does not need [`Clone`](Clone). [`GraphState::Update`] does, because
/// a task's update can appear on both [`StreamEvent::TaskFinished`] and
/// [`StreamEvent::Updates`].
#[derive(Debug)]
pub enum StreamEvent<S: GraphState> {
    /// Full state after a committed superstep, or the state at an interrupt.
    Values {
        /// Namespace path. Empty at the root.
        namespace: Namespace,
        /// Committed state, or the last committed state when the run pauses.
        state: S,
    },
    /// One task that wrote, in `(task_path, task_id)` order at the barrier.
    Updates {
        /// Namespace path. Empty at the root.
        namespace: Namespace,
        /// Node that wrote. Two sends to one node share this name.
        node: String,
        /// That task's update.
        update: S::Update,
    },
    /// A task was scheduled and started.
    ///
    /// Emitted when the task starts, not at the barrier. Replayed tasks from
    /// a previous pause are not started again and do not emit this.
    TaskStarted {
        /// Namespace path. Empty at the root.
        namespace: Namespace,
        /// Node name.
        node: String,
    },
    /// A task returned or failed.
    ///
    /// Emitted when that task finishes, before the barrier. The superstep
    /// still joins its siblings.
    TaskFinished {
        /// Namespace path. Empty at the root.
        namespace: Namespace,
        /// Node name.
        node: String,
        /// The update, or the task error.
        result: TaskResult<S::Update>,
    },
    /// A checkpoint was stored.
    ///
    /// Emitted only when the run has a checkpointer, after `put` returns.
    /// This includes the input checkpoint and each loop checkpoint.
    Checkpoint {
        /// Namespace path. Empty at the root.
        namespace: Namespace,
        /// Config that loads this checkpoint.
        config: CheckpointConfig,
    },
    /// The run paused. This is the last event.
    ///
    /// Emitted for `interrupt_before`, `interrupt_after`, and [`crate::interrupt`].
    /// The stream produces nothing after it.
    Interrupted {
        /// Namespace path. Empty at the root.
        namespace: Namespace,
        /// Last committed state. An interrupted superstep has not applied its writes.
        state: S,
        /// Unanswered interrupts. Empty for a breakpoint.
        interrupts: Vec<Interrupt>,
    },
}

/// Events from one [`crate::CompiledGraph::stream`] call, then its result.
///
/// [`GraphStream::recv`] waits for the next event. [`GraphStream::output`]
/// is [`crate::RunOutput`], the same success or pause `invoke` returns, or
/// the same error. Dropping the stream cancels the run.
pub struct GraphStream<S: GraphState> {
    receiver: UnboundedReceiver<StreamEvent<S>>,
    driver: Option<JoinHandle<Result<RunOutput<S>, Error>>>,
}

impl<S: GraphState> GraphStream<S> {
    pub(crate) fn new(
        receiver: UnboundedReceiver<StreamEvent<S>>,
        driver: JoinHandle<Result<RunOutput<S>, Error>>,
    ) -> Self {
        Self {
            receiver,
            driver: Some(driver),
        }
    }

    /// Next event produced by the run.
    ///
    /// `None` means the run has finished emitting. The terminal result is
    /// still waiting on [`GraphStream::output`].
    pub async fn recv(&mut self) -> Option<StreamEvent<S>> {
        self.receiver.recv().await
    }

    /// Terminal result. This matches [`crate::CompiledGraph::invoke`].
    ///
    /// Unread events stay buffered until this returns, then they are dropped.
    /// A panic in the run resumes here.
    pub async fn output(mut self) -> Result<RunOutput<S>, Error> {
        let handle = self.driver.take().ok_or_else(|| Error::Node {
            node: "stream".to_string(),
            message: "stream output was already taken".to_string(),
        })?;
        match handle.await {
            Ok(result) => result,
            Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
            Err(_) => Err(Error::Node {
                node: "stream".to_string(),
                message: "stream task was cancelled".to_string(),
            }),
        }
    }
}

impl<S: GraphState> Drop for GraphStream<S> {
    fn drop(&mut self) {
        if let Some(handle) = self.driver.take() {
            handle.abort();
        }
    }
}

/// Borrowed task result passed to [`Emit::task_finished`].
pub(crate) enum TaskFinish<'a, S: GraphState> {
    Update(&'a Option<S::Update>),
    Error(&'a Error),
}

/// Sink for stream events. [`NoEmit`] is `invoke`.
pub(crate) trait Emit<S: GraphState> {
    fn wants_values(&self) -> bool {
        false
    }

    fn wants_updates(&self) -> bool {
        false
    }

    fn wants_events(&self) -> bool {
        false
    }

    fn task_started(&self, _node: &str) {}

    fn task_finished(&self, _node: &str, _result: TaskFinish<'_, S>) {}

    fn wrote(&self, _node: &str, _update: &S::Update) {}

    fn values(&self, _state: S) {}

    fn checkpoint(&self, _config: &CheckpointConfig) {}

    fn interrupted(&self, _state: S, _interrupts: Vec<Interrupt>) {}
}

/// `invoke` uses this. Every method is a no-op.
pub(crate) struct NoEmit;

impl<S: GraphState> Emit<S> for NoEmit {}

/// Sends the modes the caller asked for. A closed receiver is ignored.
pub(crate) struct ChannelEmit<S: GraphState> {
    modes: StreamModes,
    namespace: Namespace,
    tx: UnboundedSender<StreamEvent<S>>,
}

impl<S: GraphState> ChannelEmit<S> {
    pub(crate) fn new(modes: StreamModes, tx: UnboundedSender<StreamEvent<S>>) -> Self {
        Self {
            modes,
            namespace: Namespace::root(),
            tx,
        }
    }
}

impl<S> Emit<S> for ChannelEmit<S>
where
    S: GraphState,
    S::Update: Clone,
{
    fn wants_values(&self) -> bool {
        self.modes.wants_values()
    }

    fn wants_updates(&self) -> bool {
        self.modes.wants_updates()
    }

    fn wants_events(&self) -> bool {
        self.modes.wants_events()
    }

    fn task_started(&self, node: &str) {
        if !self.modes.wants_events() {
            return;
        }
        let _ = self.tx.send(StreamEvent::TaskStarted {
            namespace: self.namespace.clone(),
            node: node.to_string(),
        });
    }

    fn task_finished(&self, node: &str, result: TaskFinish<'_, S>) {
        if !self.modes.wants_events() {
            return;
        }
        let result = match result {
            TaskFinish::Update(update) => TaskResult::Update(update.clone()),
            TaskFinish::Error(error) => TaskResult::Error(error.clone()),
        };
        let _ = self.tx.send(StreamEvent::TaskFinished {
            namespace: self.namespace.clone(),
            node: node.to_string(),
            result,
        });
    }

    fn wrote(&self, node: &str, update: &S::Update) {
        if !self.modes.wants_updates() {
            return;
        }
        let _ = self.tx.send(StreamEvent::Updates {
            namespace: self.namespace.clone(),
            node: node.to_string(),
            update: update.clone(),
        });
    }

    fn values(&self, state: S) {
        if !self.modes.wants_values() {
            return;
        }
        let _ = self.tx.send(StreamEvent::Values {
            namespace: self.namespace.clone(),
            state,
        });
    }

    fn checkpoint(&self, config: &CheckpointConfig) {
        if !self.modes.wants_events() {
            return;
        }
        let _ = self.tx.send(StreamEvent::Checkpoint {
            namespace: self.namespace.clone(),
            config: config.clone(),
        });
    }

    fn interrupted(&self, state: S, interrupts: Vec<Interrupt>) {
        if !self.modes.wants_events() {
            return;
        }
        let _ = self.tx.send(StreamEvent::Interrupted {
            namespace: self.namespace.clone(),
            state,
            interrupts,
        });
    }
}
