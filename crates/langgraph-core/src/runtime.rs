//! Pregel superstep runtime.
//!
//! [`CompiledGraph::invoke`] applies the input as [`START`] writes, then
//! runs scheduled nodes until the frontier is empty or the recursion limit
//! is reached. Control channels are separate from user channels:
//! `done:{node}` counts completions of that node or of `START`, and
//! `branch:{edge}:{dest}` counts conditional selections of `dest`.
//! Duplicate plain names on one edge share that branch entry. A [`crate::Send`]
//! is not a branch entry: each packet is stored on the checkpoint and becomes
//! its own task. `versions_seen` maps a node to the control-channel versions
//! it has already consumed. A checkpointer persists that map with the channel
//! values so a later invoke can resume. [`crate::interrupt`] pauses a task.
//! Finished siblings keep their writes in `put_writes` until the superstep
//! commits.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use langgraph_checkpoint::{
    Checkpoint, CheckpointConfig, CheckpointMetadata, CheckpointSend, CheckpointSource,
    CheckpointTuple, Checkpointer, PendingWrite,
};
use serde_json::{json, Value};
use tokio::task::JoinHandle;

use crate::builder::{
    Command, ConditionalEdge, Edge, EdgeSource, GraphTarget, InterruptSet, InvokeNode, RouteTarget,
};
use crate::interrupt::{
    dynamic_interrupts, has_dynamic_interrupt, is_breakpoint_marker, is_pause, scope_task,
    stored_resume, task_done, task_interrupted, with_thread_frame, FrameHandle, Interrupt,
    InterruptFrame, Resume, RunOutput, TaskResume, BREAKPOINT_TASK, CHANNEL_INTERRUPT,
    CHANNEL_RESUME, CHANNEL_TASKS, INTERRUPT_SEEN,
};
use crate::state::{ChannelMap, Write};
use crate::{Error, GraphState, END, START};

/// Default superstep budget for [`RunConfig::new`].
///
/// A run may execute this many supersteps. The next one is refused with
/// [`Error::RecursionLimit`]. Input writes are not a superstep.
pub const DEFAULT_RECURSION_LIMIT: u32 = 25;

/// Options for [`CompiledGraph::invoke`].
///
/// [`RunConfig::new`] and [`Default`] set the recursion limit to
/// [`DEFAULT_RECURSION_LIMIT`] and leave the checkpointer unset. `invoke`
/// rejects a limit below 1. Add a saver with [`RunConfig::checkpointer`] and
/// a thread with [`RunConfig::thread_id`]. Those fields are optional, so
/// [`CompiledGraph::invoke`] keeps the same parameters.
///
/// With no checkpointer, `thread_id` and `checkpoint_id` are ignored and
/// state lives only for that call. A checkpointer without a thread id is
/// [`Error::MissingThreadId`] before any node runs.
/// [`RunConfig::resume`] continues a pause when the input is `&S`.
/// When it is set on a loaded checkpoint, that `&S` input is not applied
/// as channel writes. A [`crate::Command`] input carries its own resume,
/// which is used instead of this field when both are set.
#[derive(Clone)]
pub struct RunConfig {
    recursion_limit: u32,
    thread_id: Option<String>,
    checkpoint_ns: String,
    checkpoint_id: Option<String>,
    checkpointer: Option<Arc<dyn Checkpointer>>,
    resume: Option<Resume>,
}

impl RunConfig {
    /// Config with [`DEFAULT_RECURSION_LIMIT`] and no checkpointer.
    pub const fn new() -> Self {
        Self {
            recursion_limit: DEFAULT_RECURSION_LIMIT,
            thread_id: None,
            checkpoint_ns: String::new(),
            checkpoint_id: None,
            checkpointer: None,
            resume: None,
        }
    }

    /// Superstep budget. `invoke` accepts a minimum of 1.
    pub const fn recursion_limit(mut self, limit: u32) -> Self {
        self.recursion_limit = limit;
        self
    }

    /// Thread to load and store. Ignored when no checkpointer is set.
    ///
    /// An empty string is treated as a missing id.
    pub fn thread_id(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// Checkpoint namespace. The root graph uses an empty string.
    pub fn checkpoint_ns(mut self, checkpoint_ns: impl Into<String>) -> Self {
        self.checkpoint_ns = checkpoint_ns.into();
        self
    }

    /// Snapshot to load. The next write forks a child of this id.
    pub fn checkpoint_id(mut self, checkpoint_id: impl Into<String>) -> Self {
        self.checkpoint_id = Some(checkpoint_id.into());
        self
    }

    /// Saver for this run. The map belongs to `checkpointer`, not to the
    /// process. Clone the `Arc` to share one thread across calls.
    pub fn checkpointer(mut self, checkpointer: Arc<dyn Checkpointer>) -> Self {
        self.checkpointer = Some(checkpointer);
        self
    }

    /// Value that fills the next unanswered [`crate::interrupt`] call.
    ///
    /// [`Resume::Value`] fills one call. [`Resume::Map`] fills the call with
    /// that id. A loaded checkpoint does not also apply an `&S` input when
    /// this is set. A resume value with no checkpointer is
    /// [`Error::MissingThreadId`].
    ///
    /// [`crate::Command::resume`] on the `invoke` input is the public
    /// carrier. When that field is set it is used instead of this one.
    /// This method still continues a pause for an `&S` input.
    pub fn resume(mut self, resume: Resume) -> Self {
        self.resume = Some(resume);
        self
    }

    /// Superstep budget stored on this config.
    pub const fn get_recursion_limit(&self) -> u32 {
        self.recursion_limit
    }

    /// Thread id, if one was set and it is not empty.
    pub fn get_thread_id(&self) -> Option<&str> {
        match self.thread_id.as_deref() {
            Some(id) if !id.is_empty() => Some(id),
            _ => None,
        }
    }

    /// Namespace stored on this config. Empty at the root.
    pub fn get_checkpoint_ns(&self) -> &str {
        &self.checkpoint_ns
    }

    /// Snapshot id stored on this config.
    pub fn get_checkpoint_id(&self) -> Option<&str> {
        self.checkpoint_id.as_deref()
    }

    /// Saver stored on this config.
    pub fn get_checkpointer(&self) -> Option<&Arc<dyn Checkpointer>> {
        self.checkpointer.as_ref()
    }

    /// Resume value stored on this config.
    pub fn get_resume(&self) -> Option<&Resume> {
        self.resume.as_ref()
    }

    /// Thread, namespace, and checkpoint id for a saver call.
    ///
    /// [`Error::MissingThreadId`] when [`RunConfig::get_thread_id`] is
    /// `None`. The checkpointer is not part of the returned config.
    pub fn checkpoint_config(&self) -> Result<CheckpointConfig, Error> {
        let thread_id = self.get_thread_id().ok_or(Error::MissingThreadId)?;
        Ok(CheckpointConfig {
            thread_id: thread_id.to_string(),
            checkpoint_ns: self.checkpoint_ns.clone(),
            checkpoint_id: self.checkpoint_id.clone(),
        })
    }

    fn at_checkpoint(&self, checkpoint_id: Option<String>) -> Self {
        let mut config = self.clone();
        config.checkpoint_id = checkpoint_id;
        config
    }
}

impl std::fmt::Debug for RunConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RunConfig")
            .field("recursion_limit", &self.recursion_limit)
            .field("thread_id", &self.thread_id)
            .field("checkpoint_ns", &self.checkpoint_ns)
            .field("checkpoint_id", &self.checkpoint_id)
            .field("checkpointer", &self.checkpointer.as_ref().map(|_| "saver"))
            .field("resume", &self.resume)
            .finish()
    }
}

impl PartialEq for RunConfig {
    fn eq(&self, other: &Self) -> bool {
        self.recursion_limit == other.recursion_limit
            && self.thread_id == other.thread_id
            && self.checkpoint_ns == other.checkpoint_ns
            && self.checkpoint_id == other.checkpoint_id
            && self.resume == other.resume
            && match (&self.checkpointer, &other.checkpointer) {
                (None, None) => true,
                (Some(left), Some(right)) => Arc::ptr_eq(left, right),
                _ => false,
            }
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> crate::CompiledGraph<S>
where
    S: GraphState + Send + Sync + 'static,
    S::Update: Send + 'static,
{
    /// Run the graph until it finishes or pauses.
    ///
    /// `input` is `&S` or a [`crate::Command`]. Channel updates are attributed
    /// to [`START`] and are not a superstep. With no checkpointer this is the
    /// whole state of a fresh run: the map starts empty, then the input is
    /// folded in. A loaded checkpoint receives an `&S` input unless
    /// [`RunConfig::resume`] is set or the checkpoint is already a pause. In
    /// those cases the `&S` input is not applied. A command `update` is still
    /// applied. A command `resume` is the public resume carrier and is used
    /// instead of [`RunConfig::resume`] when both are set.
    ///
    /// Scheduling uses control channels, not the user-channel versions. A
    /// normal edge fires after its source completes. [`START`] fires the
    /// entry nodes. A waiting edge fires after every listed source has a
    /// completion newer than the one the target last consumed. Those sources
    /// need not have finished in the same superstep.
    ///
    /// A superstep runs every scheduled task at once. A plain trigger reads
    /// the committed snapshot from the start of the step. A [`crate::Send`]
    /// reads that snapshot with the packet's JSON object merged on top, and
    /// each packet is its own task, including two sends to the same node.
    /// Sync nodes are called on `spawn_blocking`. Async nodes are awaited on
    /// this runtime. After a node returns a plain update, each of its
    /// conditional routers runs inside that task. A router sees
    /// `reduce(snapshot, this_node_update)` and does not see other tasks from
    /// this step. A static path map is applied to plain labels before names
    /// are checked. A [`crate::Send`] names its node directly. An unknown name
    /// is [`Error::UnknownNode`] and that step does not commit.
    ///
    /// A node may return [`crate::Command`] instead of `S::Update`. `update`
    /// is merged like a normal node return. `goto`, when set, replaces that
    /// node's static outgoing edges for this step and skips its conditional
    /// routers. Plain names in `goto` still dedupe. Each [`crate::Send`] is
    /// its own task. `resume` on a node return is ignored.
    /// [`GraphTarget::Parent`] is [`Error::ParentGraph`] and that superstep
    /// does not commit. Parent routing requires a subgraph.
    ///
    /// A command `goto` on the input routes from [`START`] for this run and
    /// does not also fire `START`'s static edges or `START` routers.
    /// [`GraphTarget::Parent`] on the input is [`Error::ParentGraph`] before
    /// any checkpoint write.
    ///
    /// The barrier folds the step's writes in `(task_path, task_id, write_index)`
    /// order. `task_path` is empty at the root. `task_id` is the node name for
    /// a plain trigger. Each [`crate::Send`] gets its own `task_id`, ordered
    /// by send index, so two sends to one node do not fold in completion order.
    /// A task [`Err`](Result::Err) fails the superstep: the other tasks
    /// are joined, and the step commits nothing. Two overwrite writes in one
    /// step are [`Error::InvalidUpdate`] and also commit nothing.
    /// [`crate::interrupt`] without a checkpointer is [`Error::MissingThreadId`]
    /// and that step does not commit either.
    ///
    /// `interrupt_before` pauses before a matching node runs.
    /// `interrupt_after` pauses after the superstep that ran a matching node
    /// has committed. Neither breakpoint carries a resume value. A graph
    /// compiled with either breakpoint and invoked with no checkpointer
    /// returns [`Error::MissingThreadId`] before any node runs.
    ///
    /// When a task calls [`crate::interrupt`] and that call has no resume
    /// value yet, sibling tasks still finish. Their writes are stored with
    /// `put_writes` and are not applied yet. On resume, finished tasks are
    /// not executed again. Their pending writes are replayed. The interrupted
    /// task runs from the start. Channel values commit only when every task
    /// in the superstep has finished without a new interrupt. The result is
    /// [`RunOutput::Interrupted`], not an error.
    ///
    /// The recursion limit counts committed supersteps. The run executes at
    /// most `config`'s limit and returns [`Error::RecursionLimit`] before
    /// starting the next one. Routers are not a separate superstep. A
    /// breakpoint pause does not count as a superstep.
    ///
    /// Polling this future outside tokio returns [`Error::NoRuntime`]. There
    /// is no `stream` method. A conditional router or a command `goto` may
    /// return [`crate::Send`].
    ///
    /// With a checkpointer, the call loads `config`'s checkpoint first
    /// (`checkpoint_id` when set, otherwise the latest in the thread and
    /// namespace). If the loaded checkpoint still has nodes scheduled, those
    /// nodes run. The call does not start again from [`START`]. If nothing is
    /// scheduled, or there is no checkpoint yet, `done:__start__` increments
    /// and the entry nodes run on the merged state. A command `goto` skips
    /// that increment and schedules its own targets instead. An input checkpoint
    /// (`source: Input`, step `-1` on a fresh thread) is stored after that
    /// preparation and before the first superstep, except when this call is
    /// continuing a pause. Each committed superstep stores a `Loop`
    /// checkpoint before the next superstep starts.
    ///
    /// A `checkpoint_id` loads that snapshot. The following `put` creates a
    /// new checkpoint whose parent is the selected id. Older checkpoints,
    /// including the previous tip, stay as they were.
    pub async fn invoke<I>(&self, input: I, config: &RunConfig) -> Result<RunOutput<S>, Error>
    where
        I: InvokeInput<S>,
    {
        if tokio::runtime::Handle::try_current().is_err() {
            return Err(Error::NoRuntime);
        }
        let run_input = input.into_run_input()?;
        if run_input.graph == GraphTarget::Parent {
            return Err(Error::ParentGraph);
        }
        let RunInput {
            writes,
            resume: command_resume,
            goto,
            graph: _,
            honor_on_resume,
        } = run_input;
        let resume = command_resume.or_else(|| config.resume.clone());
        if config.checkpointer.is_some() && config.get_thread_id().is_none() {
            return Err(Error::MissingThreadId);
        }
        let breakpoints = !self.interrupt_before.is_empty() || !self.interrupt_after.is_empty();
        if breakpoints && config.checkpointer.is_none() {
            return Err(Error::MissingThreadId);
        }
        if resume.is_some() && config.checkpointer.is_none() {
            return Err(Error::MissingThreadId);
        }
        if config.recursion_limit < 1 {
            return Err(Error::GraphValidation(
                "recursion limit must be at least 1".into(),
            ));
        }

        let nodes = node_map(&self.nodes);
        let known = Arc::new(nodes.keys().cloned().collect::<BTreeSet<_>>());
        let conditionals = Arc::new(self.conditional_edges.clone());
        let saver = config.checkpointer.clone();

        let mut channels = ChannelMap::new(S::schema())?;
        let mut control = Control::new();
        let mut step = -1i64;
        let mut parent_id: Option<String> = None;
        let mut loaded = false;
        let mut carried_writes: Vec<PendingWrite> = Vec::new();
        if let Some(saver) = saver.as_ref() {
            let lookup = config.checkpoint_config()?;
            if let Some(tuple) = saver.get_tuple(&lookup).await.map_err(checkpoint_error)? {
                let restored = restore_run::<S>(&tuple.checkpoint)?;
                channels = restored.0;
                control = restored.1;
                step = tuple.metadata.step;
                parent_id = Some(tuple.checkpoint.id);
                carried_writes = tuple.pending_writes;
                loaded = true;
            }
        }

        let continuing_pause = is_pause(&carried_writes);
        let skip_input = loaded && (continuing_pause || resume.is_some());
        if skip_input {
            carried_writes.retain(|write| !is_breakpoint_marker(write));
            if honor_on_resume && !writes.is_empty() {
                channels.apply_writes(&writes)?;
            }
        } else {
            let input_channels = channel_names(&writes);
            channels.apply_writes(&writes)?;
            let pending = schedule(&self.edges, &nodes, &control);
            if !loaded || pending.is_empty() {
                if let Some(goto) = &goto {
                    let routes = targets_to_routes::<S>(goto, COMMAND_EDGE, known.as_ref())?;
                    control.apply_routes(&routes);
                } else {
                    control.bump_done(START);
                    if conditionals.iter().any(|edge| edge.from == START) {
                        let state = channels.to_state::<S>()?;
                        let routes =
                            route_from(conditionals.as_slice(), START, &state, known.as_ref())?;
                        control.apply_routes(&routes);
                    }
                }
            }
            if let Some(saver) = saver.as_ref() {
                parent_id = Some(
                    save_checkpoint(
                        saver.as_ref(),
                        config,
                        parent_id.as_deref(),
                        SavedStep {
                            channels: &channels,
                            control: &control,
                            updated: &input_channels,
                            source: CheckpointSource::Input,
                            step,
                        },
                    )
                    .await?
                    .checkpoint_id
                    .ok_or_else(|| {
                        Error::Checkpoint("saver did not return a checkpoint id".into())
                    })?,
                );
            }
        }

        let mut completed_supersteps = 0u32;
        loop {
            let scheduled = schedule(&self.edges, &nodes, &control);
            if scheduled.is_empty() {
                return Ok(RunOutput::Finished {
                    state: channels.to_state()?,
                });
            }
            if completed_supersteps == config.recursion_limit {
                return Err(Error::RecursionLimit(config.recursion_limit));
            }
            let names = scheduled
                .iter()
                .map(|task| task.name.clone())
                .collect::<Vec<_>>();
            if !has_dynamic_interrupt(&carried_writes)
                && interrupts_match(&self.interrupt_before, &names, &channels, &control)
            {
                mark_interrupt_seen(&mut control, &channels);
                if let Some(saver) = saver.as_ref() {
                    step += 1;
                    let saved = save_checkpoint(
                        saver.as_ref(),
                        config,
                        parent_id.as_deref(),
                        SavedStep {
                            channels: &channels,
                            control: &control,
                            updated: &[],
                            source: CheckpointSource::Loop,
                            step,
                        },
                    )
                    .await?;
                    put_pending(saver.as_ref(), &saved, &breakpoint_writes()).await?;
                }
                return Ok(RunOutput::Interrupted {
                    state: channels.to_state()?,
                    interrupts: Vec::new(),
                });
            }
            match run_superstep(
                &mut channels,
                &mut control,
                scheduled,
                SuperstepInput {
                    nodes: &nodes,
                    conditionals: &conditionals,
                    known: &known,
                    pending: &carried_writes,
                    config,
                    resume: resume.as_ref(),
                },
            )
            .await?
            {
                StepOutcome::Done(done) => {
                    carried_writes.clear();
                    completed_supersteps += 1;
                    let pause_after = interrupts_match(
                        &self.interrupt_after,
                        &done.finished,
                        &channels,
                        &control,
                    );
                    if pause_after {
                        mark_interrupt_seen(&mut control, &channels);
                    }
                    if let Some(saver) = saver.as_ref() {
                        step += 1;
                        let saved = save_checkpoint(
                            saver.as_ref(),
                            config,
                            parent_id.as_deref(),
                            SavedStep {
                                channels: &channels,
                                control: &control,
                                updated: &done.updated,
                                source: CheckpointSource::Loop,
                                step,
                            },
                        )
                        .await?;
                        parent_id = Some(saved.checkpoint_id.clone().ok_or_else(|| {
                            Error::Checkpoint("saver did not return a checkpoint id".into())
                        })?);
                        if pause_after {
                            put_pending(saver.as_ref(), &saved, &breakpoint_writes()).await?;
                        }
                    }
                    if pause_after {
                        return Ok(RunOutput::Interrupted {
                            state: channels.to_state()?,
                            interrupts: Vec::new(),
                        });
                    }
                }
                StepOutcome::Paused {
                    interrupts,
                    pending,
                } => {
                    if let Some(saver) = saver.as_ref() {
                        step += 1;
                        let saved = save_checkpoint(
                            saver.as_ref(),
                            config,
                            parent_id.as_deref(),
                            SavedStep {
                                channels: &channels,
                                control: &control,
                                updated: &[],
                                source: CheckpointSource::Loop,
                                step,
                            },
                        )
                        .await?;
                        put_pending(saver.as_ref(), &saved, &pending).await?;
                    }
                    return Ok(RunOutput::Interrupted {
                        state: channels.to_state()?,
                        interrupts,
                    });
                }
            }
        }
    }

    /// Load the checkpoint selected by `config`.
    ///
    /// The snapshot carries the channel values as `S`, the node names the
    /// next superstep would still run, the config for this checkpoint, the
    /// parent config, and those pending nodes as tasks. `interrupts` comes
    /// from unanswered [`crate::interrupt`] calls stored on the checkpoint.
    /// A finished sibling is not listed in `next`.
    ///
    /// No checkpointer is [`Error::Checkpoint`]. A checkpointer without a
    /// thread id is [`Error::MissingThreadId`]. A thread with no checkpoint
    /// is [`Error::Checkpoint`].
    pub async fn get_state(&self, config: &RunConfig) -> Result<StateSnapshot<S>, Error> {
        require_runtime()?;
        let saver = require_saver(config)?;
        let lookup = config.checkpoint_config()?;
        let tuple = saver
            .get_tuple(&lookup)
            .await
            .map_err(checkpoint_error)?
            .ok_or_else(|| Error::Checkpoint("no checkpoint for this thread".into()))?;
        self.snapshot(config, &tuple)
    }

    /// Checkpoints for `config`'s thread and namespace, newest first.
    ///
    /// `before` and `limit` are forwarded to [`Checkpointer::list`]. This
    /// walks the stored list. It does not follow only the parent chain, so a
    /// fork leaves the previous tip in the result.
    pub async fn get_state_history(
        &self,
        config: &RunConfig,
        before: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StateSnapshot<S>>, Error> {
        require_runtime()?;
        let saver = require_saver(config)?;
        let lookup = config.checkpoint_config()?;
        let tuples = saver
            .list(&lookup, before, limit)
            .await
            .map_err(checkpoint_error)?;
        tuples
            .iter()
            .map(|tuple| self.snapshot(config, tuple))
            .collect()
    }

    /// Write `values` as if `as_node` finished, and store an `Update` checkpoint.
    ///
    /// No node function runs. `as_node`'s triggers are marked consumed, its
    /// completion count increments, and its conditional routers run on the
    /// merged state. The next superstep is that node's successors. `START`
    /// is a legal `as_node` and schedules the entry nodes.
    ///
    /// The returned config points at the new checkpoint and keeps the same
    /// saver, thread, namespace, and recursion limit. A `checkpoint_id` on
    /// `config` selects the parent snapshot. The previous tip stays in
    /// history.
    pub async fn update_state(
        &self,
        config: &RunConfig,
        values: &S::Update,
        as_node: &str,
    ) -> Result<RunConfig, Error> {
        require_runtime()?;
        let saver = require_saver(config)?;
        if as_node != START && !self.nodes.iter().any(|(name, _)| name == as_node) {
            return Err(Error::UnknownNode(as_node.to_string()));
        }

        let lookup = config.checkpoint_config()?;
        let loaded = saver.get_tuple(&lookup).await.map_err(checkpoint_error)?;
        let (mut channels, mut control, step, parent_id) = match &loaded {
            Some(tuple) => {
                let (channels, control) = restore_run::<S>(&tuple.checkpoint)?;
                (
                    channels,
                    control,
                    tuple.metadata.step.saturating_add(1),
                    Some(tuple.checkpoint.id.clone()),
                )
            }
            None => (ChannelMap::new(S::schema())?, Control::new(), -1, None),
        };

        let writes = S::update_to_writes(values, &[], as_node)?;
        let updated = channel_names(&writes);
        channels.apply_writes(&writes)?;
        if as_node == START {
            control.bump_done(START);
        } else {
            let consumed = incoming_triggers(&self.edges, as_node, &control);
            control.mark_consumed(as_node, &consumed);
            control.bump_done(as_node);
        }
        if self
            .conditional_edges
            .iter()
            .any(|edge| edge.from == as_node)
        {
            let known = self
                .nodes
                .iter()
                .map(|(name, _)| name.clone())
                .collect::<BTreeSet<_>>();
            let state = channels.to_state::<S>()?;
            let routes = route_from(&self.conditional_edges, as_node, &state, &known)?;
            control.apply_routes(&routes);
        }

        let saved = save_checkpoint(
            saver.as_ref(),
            config,
            parent_id.as_deref(),
            SavedStep {
                channels: &channels,
                control: &control,
                updated: &updated,
                source: CheckpointSource::Update,
                step,
            },
        )
        .await?;
        Ok(config.at_checkpoint(saved.checkpoint_id))
    }

    fn snapshot(
        &self,
        config: &RunConfig,
        tuple: &CheckpointTuple,
    ) -> Result<StateSnapshot<S>, Error> {
        let (channels, control) = restore_run::<S>(&tuple.checkpoint)?;
        let nodes = node_map(&self.nodes);
        let next = schedule(&self.edges, &nodes, &control)
            .into_iter()
            .filter(|task| !task_done(&tuple.pending_writes, &task.task_id))
            .map(|task| task.name)
            .collect::<Vec<_>>();
        let tasks = next
            .iter()
            .map(|name| PendingTask { name: name.clone() })
            .collect();
        let parent_config = tuple
            .parent_config
            .as_ref()
            .map(|parent| config.at_checkpoint(parent.checkpoint_id.clone()));
        Ok(StateSnapshot {
            values: channels.to_state()?,
            next,
            config: config.at_checkpoint(Some(tuple.checkpoint.id.clone())),
            parent_config,
            tasks,
            interrupts: dynamic_interrupts(&tuple.pending_writes)?,
            metadata: tuple.metadata.clone(),
        })
    }
}

/// Channel values and scheduling position of one checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct StateSnapshot<S> {
    /// State reconstructed from the checkpoint's channel values.
    pub values: S,
    /// Node names the next superstep would run, in name order.
    ///
    /// One plain trigger contributes one name. Each stored fan-out packet
    /// contributes another, so one node may appear more than once.
    pub next: Vec<String>,
    /// Config that loads this checkpoint, including the same saver.
    pub config: RunConfig,
    /// Config of the parent checkpoint in this namespace.
    pub parent_config: Option<RunConfig>,
    /// One entry per task in [`StateSnapshot::next`], in the same order.
    pub tasks: Vec<PendingTask>,
    /// Unanswered [`crate::interrupt`] calls stored on this checkpoint.
    ///
    /// Empty when the checkpoint is not paused inside a node. Breakpoints do
    /// not add an entry.
    pub interrupts: Vec<Interrupt>,
    /// Source, step, and parent ids stored with the checkpoint.
    pub metadata: CheckpointMetadata,
}

/// A node that would run on the next superstep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingTask {
    /// Node name. Two fan-out packets to one node are two tasks with the
    /// same name.
    pub name: String,
}

struct Control {
    done: BTreeMap<String, u64>,
    branch: BTreeMap<String, u64>,
    seen: BTreeMap<String, BTreeMap<String, u64>>,
    sends: Vec<CheckpointSend>,
}

impl Control {
    fn new() -> Self {
        Self {
            done: BTreeMap::new(),
            branch: BTreeMap::new(),
            seen: BTreeMap::new(),
            sends: Vec::new(),
        }
    }

    fn version(&self, key: &str) -> u64 {
        if let Some(name) = key.strip_prefix("done:") {
            self.done.get(name).copied().unwrap_or(0)
        } else {
            self.branch.get(key).copied().unwrap_or(0)
        }
    }

    fn seen_version(&self, node: &str, key: &str) -> u64 {
        self.seen
            .get(node)
            .and_then(|versions| versions.get(key))
            .copied()
            .unwrap_or(0)
    }

    fn apply_routes(&mut self, routes: &[RoutePush]) {
        let mut bumped = BTreeSet::new();
        for route in routes {
            if route.destination == END {
                continue;
            }
            if let Some(value) = &route.send {
                self.sends.push(CheckpointSend {
                    edge_index: route.edge_index,
                    node: route.destination.clone(),
                    value: value.clone(),
                });
                continue;
            }
            if !bumped.insert((route.edge_index, route.destination.clone())) {
                continue;
            }
            let key = branch_key(route.edge_index, &route.destination);
            *self.branch.entry(key).or_insert(0) += 1;
        }
    }

    fn mark_consumed(&mut self, node: &str, consumed: &[(String, u64)]) {
        let seen = self.seen.entry(node.to_string()).or_default();
        for (key, version) in consumed {
            seen.insert(key.clone(), *version);
        }
    }

    fn bump_done(&mut self, node: &str) {
        *self.done.entry(node.to_string()).or_insert(0) += 1;
    }
}

struct RoutePush {
    edge_index: usize,
    destination: String,
    /// Set for a [`crate::Send`]. Plain names are `None` and are deduped.
    send: Option<Value>,
}

struct Scheduled {
    name: String,
    task_id: String,
    consumed: Vec<(String, u64)>,
    /// Index into [`Control::sends`] when this task is a fan-out packet.
    send_index: Option<usize>,
    send_value: Option<Value>,
}

struct TaskOutput {
    task_id: String,
    writes: Vec<Write>,
    routed: Vec<RoutePush>,
    pause: Option<TaskPause>,
    /// `Command.goto` replaced this task's static outgoing edges.
    skip_static: bool,
}

struct StepShared<S: GraphState> {
    base: ChannelMap,
    conditionals: Arc<Vec<ConditionalEdge<S>>>,
    known: Arc<BTreeSet<String>>,
}

fn input_writes<S: GraphState>(input: &S) -> Result<Vec<Write>, Error> {
    let values = input.to_channel_values()?;
    for key in values.keys() {
        if !S::schema().iter().any(|field| field.name == key) {
            return Err(Error::InvalidUpdate(format!(
                "input writes unknown channel '{key}'"
            )));
        }
    }
    let mut writes = Vec::new();
    for field in S::schema() {
        if let Some(value) = values.get(field.name) {
            writes.push(Write {
                channel: field.name.to_string(),
                value: value.clone(),
                task_path: Vec::new(),
                task_id: START.to_string(),
                write_index: u64::try_from(writes.len()).map_err(|_| {
                    Error::InvalidUpdate(format!("channel '{}' write index overflow", field.name))
                })?,
            });
        }
    }
    Ok(writes)
}

/// Prepared `invoke` input.
///
/// [`InvokeInput::into_run_input`] builds this. `&S` fills `writes` from the
/// whole state. [`Command`] fills them from `update`.
#[derive(Debug)]
pub struct RunInput {
    /// Channel writes attributed to [`START`].
    pub writes: Vec<Write>,
    /// Resume payload for this call, if the input carried one.
    pub resume: Option<Resume>,
    /// `Some` replaces the static edges and routers from [`START`].
    pub goto: Option<Vec<RouteTarget>>,
    /// [`GraphTarget::Parent`] is rejected before a checkpoint write.
    pub graph: GraphTarget,
    /// When resuming, apply `writes`. `&S` inputs leave this false.
    pub(crate) honor_on_resume: bool,
}

mod invoke_sealed {
    use super::{Command, GraphState};

    pub trait Sealed<S: GraphState> {}

    impl<S: GraphState> Sealed<S> for &S {}

    impl<S: GraphState> Sealed<S> for Command<S::Update> {}
}

/// Value accepted by [`crate::CompiledGraph::invoke`].
///
/// `&S` is a full state write. [`Command`] may carry `update`, `resume`,
/// and `goto`. The trait is sealed.
#[allow(private_bounds)]
pub trait InvokeInput<S: GraphState>: invoke_sealed::Sealed<S> {
    /// Channel writes and control for this call.
    fn into_run_input(self) -> Result<RunInput, Error>;
}

impl<S> InvokeInput<S> for &S
where
    S: GraphState,
{
    fn into_run_input(self) -> Result<RunInput, Error> {
        Ok(RunInput {
            writes: input_writes(self)?,
            resume: None,
            goto: None,
            graph: GraphTarget::Current,
            honor_on_resume: false,
        })
    }
}

impl<S> InvokeInput<S> for Command<S::Update>
where
    S: GraphState,
{
    fn into_run_input(self) -> Result<RunInput, Error> {
        if self.graph == GraphTarget::Parent {
            return Err(Error::ParentGraph);
        }
        let writes = match &self.update {
            Some(update) => S::update_to_writes(update, &[], START)?,
            None => Vec::new(),
        };
        Ok(RunInput {
            writes,
            resume: self.resume,
            goto: self.goto,
            graph: self.graph,
            honor_on_resume: true,
        })
    }
}

fn node_map<S>(
    nodes: &[(String, Arc<dyn InvokeNode<S>>)],
) -> BTreeMap<String, Arc<dyn InvokeNode<S>>>
where
    S: GraphState,
{
    nodes
        .iter()
        .map(|(name, node)| (name.clone(), Arc::clone(node)))
        .collect()
}

fn done_key(name: &str) -> String {
    format!("done:{name}")
}

fn branch_key(edge_index: usize, destination: &str) -> String {
    format!("branch:{edge_index}:{destination}")
}

fn branch_destination(key: &str) -> Option<&str> {
    let rest = key.strip_prefix("branch:")?;
    let (_index, destination) = rest.split_once(':')?;
    Some(destination)
}

fn schedule<S>(
    edges: &[Edge],
    nodes: &BTreeMap<String, Arc<dyn InvokeNode<S>>>,
    control: &Control,
) -> Vec<Scheduled>
where
    S: GraphState,
{
    let mut scheduled = Vec::new();
    for name in nodes.keys() {
        if let Some(consumed) = ready_triggers(edges, name, control) {
            scheduled.push(Scheduled {
                name: name.clone(),
                task_id: name.clone(),
                consumed,
                send_index: None,
                send_value: None,
            });
        }
    }
    for (index, packet) in control.sends.iter().enumerate() {
        scheduled.push(Scheduled {
            name: packet.node.clone(),
            task_id: send_task_id(&packet.node, index),
            consumed: Vec::new(),
            send_index: Some(index),
            send_value: Some(packet.value.clone()),
        });
    }
    scheduled.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.task_id.cmp(&right.task_id))
    });
    scheduled
}

fn send_task_id(node: &str, index: usize) -> String {
    format!("{node}:{index:08}")
}

fn ready_triggers(edges: &[Edge], node: &str, control: &Control) -> Option<Vec<(String, u64)>> {
    let mut consumed = BTreeMap::<String, u64>::new();
    let mut ready = false;

    for edge in edges {
        if edge.target != node {
            continue;
        }
        let group = match &edge.source {
            EdgeSource::One(source) => vec![done_key(source)],
            EdgeSource::Wait(sources) => sources.iter().map(|source| done_key(source)).collect(),
        };
        if group.is_empty() {
            continue;
        }
        let versions: Vec<(String, u64)> = group
            .into_iter()
            .map(|key| {
                let version = control.version(&key);
                (key, version)
            })
            .collect();
        let satisfied = versions
            .iter()
            .all(|(key, version)| *version > control.seen_version(node, key));
        if satisfied {
            ready = true;
            for (key, version) in versions {
                consumed.insert(key, version);
            }
        }
    }

    for (key, version) in &control.branch {
        if branch_destination(key) != Some(node) {
            continue;
        }
        if *version > control.seen_version(node, key) {
            ready = true;
            consumed.insert(key.clone(), *version);
        }
    }

    if ready {
        Some(consumed.into_iter().collect())
    } else {
        None
    }
}

/// Branch index for [`crate::Command::goto`]. It is not a conditional edge.
const COMMAND_EDGE: usize = usize::MAX;

fn push_send<S: GraphState>(
    node: &str,
    value: &Value,
    edge_index: usize,
    known: &BTreeSet<String>,
) -> Result<Option<RoutePush>, Error> {
    if node == END {
        return Ok(None);
    }
    if !known.contains(node) {
        return Err(Error::UnknownNode(node.to_string()));
    }
    if !value.is_object() {
        return Err(Error::InvalidUpdate(
            "Send value must be a JSON object".into(),
        ));
    }
    if let Some(object) = value.as_object() {
        for key in object.keys() {
            if !S::schema().iter().any(|field| field.name == key) {
                return Err(Error::InvalidUpdate(format!(
                    "Send value writes unknown channel '{key}'"
                )));
            }
        }
    }
    Ok(Some(RoutePush {
        edge_index,
        destination: node.to_string(),
        send: Some(value.clone()),
    }))
}

fn targets_to_routes<S: GraphState>(
    targets: &[RouteTarget],
    edge_index: usize,
    known: &BTreeSet<String>,
) -> Result<Vec<RoutePush>, Error> {
    let mut routed = Vec::new();
    for target in targets {
        let route = match target {
            RouteTarget::Node(destination) => {
                if destination == END {
                    None
                } else if !known.contains(destination) {
                    return Err(Error::UnknownNode(destination.clone()));
                } else {
                    Some(RoutePush {
                        edge_index,
                        destination: destination.clone(),
                        send: None,
                    })
                }
            }
            RouteTarget::Send(packet) => {
                push_send::<S>(&packet.node, &packet.value, edge_index, known)?
            }
        };
        if let Some(route) = route {
            routed.push(route);
        }
    }
    Ok(routed)
}

fn route_from<S: GraphState>(
    conditionals: &[ConditionalEdge<S>],
    from: &str,
    state: &S,
    known: &BTreeSet<String>,
) -> Result<Vec<RoutePush>, Error> {
    let mut routed = Vec::new();
    for (edge_index, edge) in conditionals.iter().enumerate() {
        if edge.from != from {
            continue;
        }
        let targets = edge.router.route(state)?;
        for target in targets {
            match target {
                RouteTarget::Node(label) => {
                    let destination = match &edge.path_map {
                        Some(path_map) => path_map.get(&label).cloned().ok_or_else(|| {
                            Error::UnknownNode(format!("path map label '{label}'"))
                        })?,
                        None => label,
                    };
                    if destination != END && !known.contains(&destination) {
                        return Err(Error::UnknownNode(destination));
                    }
                    if destination != END {
                        routed.push(RoutePush {
                            edge_index,
                            destination,
                            send: None,
                        });
                    }
                }
                RouteTarget::Send(packet) => {
                    if let Some(route) =
                        push_send::<S>(&packet.node, &packet.value, edge_index, known)?
                    {
                        routed.push(route);
                    }
                }
            }
        }
    }
    Ok(routed)
}

fn preview_state<S: GraphState>(base: &ChannelMap, writes: &[Write]) -> Result<S, Error> {
    if writes.is_empty() {
        return base.to_state();
    }
    let mut preview = base.clone();
    preview.apply_writes(writes)?;
    preview.to_state()
}

fn after_command<S: GraphState>(
    name: String,
    task_id: String,
    command: Command<S::Update>,
    shared: &StepShared<S>,
) -> Result<TaskOutput, Error> {
    if command.graph == GraphTarget::Parent {
        return Err(Error::ParentGraph);
    }
    let writes = match &command.update {
        Some(update) => S::update_to_writes(update, &[], &task_id)?,
        None => Vec::new(),
    };
    let (routed, skip_static) = if let Some(goto) = &command.goto {
        (
            targets_to_routes::<S>(goto, COMMAND_EDGE, shared.known.as_ref())?,
            true,
        )
    } else if shared.conditionals.iter().any(|edge| edge.from == name) {
        let preview = preview_state::<S>(&shared.base, &writes)?;
        (
            route_from(
                shared.conditionals.as_slice(),
                &name,
                &preview,
                shared.known.as_ref(),
            )?,
            false,
        )
    } else {
        (Vec::new(), false)
    };
    Ok(TaskOutput {
        task_id,
        writes,
        routed,
        pause: None,
        skip_static,
    })
}

fn merge_send_view<S: GraphState>(base: &ChannelMap, value: &Value) -> Result<S, Error> {
    let object = value
        .as_object()
        .ok_or_else(|| Error::InvalidUpdate("Send value must be a JSON object".into()))?;
    let mut values = base.values();
    for (key, field_value) in object {
        if !S::schema().iter().any(|field| field.name == key) {
            return Err(Error::InvalidUpdate(format!(
                "Send value writes unknown channel '{key}'"
            )));
        }
        values.insert(key.clone(), field_value.clone());
    }
    S::from_channel_values(&values)
}

fn node_failure(name: &str, error: Error) -> Error {
    Error::Node {
        node: name.to_string(),
        message: error.to_string(),
    }
}

fn drive_sync<S: GraphState>(
    node: &dyn InvokeNode<S>,
    state: &S,
) -> Result<Command<S::Update>, Error> {
    let mut future = node.call(state);
    let waker = std::task::Waker::noop();
    let mut context = std::task::Context::from_waker(waker);
    match future.as_mut().poll(&mut context) {
        std::task::Poll::Ready(result) => result,
        std::task::Poll::Pending => {
            panic!("sync node returned a future that is not ready");
        }
    }
}

struct TaskPause {
    interrupt: Interrupt,
    resume_values: Vec<Value>,
}

struct CommittedStep {
    updated: Vec<String>,
    finished: Vec<String>,
}

enum StepOutcome {
    Done(CommittedStep),
    Paused {
        interrupts: Vec<Interrupt>,
        pending: BTreeMap<String, Vec<PendingWrite>>,
    },
}

fn spawn_task<S>(
    node: Arc<dyn InvokeNode<S>>,
    snapshot: Arc<S>,
    shared: Arc<StepShared<S>>,
    name: String,
    task_id: String,
    frame: FrameHandle,
) -> JoinHandle<Result<TaskOutput, Error>>
where
    S: GraphState + Send + Sync + 'static,
    S::Update: Send + 'static,
{
    if node.is_sync() {
        tokio::task::spawn_blocking(move || {
            with_thread_frame(Arc::clone(&frame), || {
                let command = drive_sync(node.as_ref(), snapshot.as_ref());
                finish_task(name, task_id, command, frame.as_ref(), shared.as_ref())
            })
        })
    } else {
        tokio::spawn(scope_task(Arc::clone(&frame), async move {
            let command = node.call(snapshot.as_ref()).await;
            finish_task(name, task_id, command, frame.as_ref(), shared.as_ref())
        }))
    }
}

fn finish_task<S: GraphState>(
    name: String,
    task_id: String,
    command: Result<Command<S::Update>, Error>,
    frame: &InterruptFrame,
    shared: &StepShared<S>,
) -> Result<TaskOutput, Error> {
    if let Some(interrupt) = frame.take_raised() {
        return Ok(TaskOutput {
            task_id,
            writes: Vec::new(),
            routed: Vec::new(),
            pause: Some(TaskPause {
                interrupt,
                resume_values: frame.resume_values(),
            }),
            skip_static: false,
        });
    }
    match command {
        Ok(command) => after_command(name, task_id, command, shared),
        Err(Error::MissingThreadId) => Err(Error::MissingThreadId),
        Err(Error::ParentGraph) => Err(Error::ParentGraph),
        Err(error) => Err(node_failure(&name, error)),
    }
}

struct SuperstepInput<'a, S: GraphState> {
    nodes: &'a BTreeMap<String, Arc<dyn InvokeNode<S>>>,
    conditionals: &'a Arc<Vec<ConditionalEdge<S>>>,
    known: &'a Arc<BTreeSet<String>>,
    pending: &'a [PendingWrite],
    config: &'a RunConfig,
    resume: Option<&'a Resume>,
}

async fn run_superstep<S>(
    channels: &mut ChannelMap,
    control: &mut Control,
    scheduled: Vec<Scheduled>,
    input: SuperstepInput<'_, S>,
) -> Result<StepOutcome, Error>
where
    S: GraphState + Send + Sync + 'static,
    S::Update: Send + 'static,
{
    let SuperstepInput {
        nodes,
        conditionals,
        known,
        pending,
        config,
        resume,
    } = input;
    for task in &scheduled {
        if !nodes.contains_key(&task.name) {
            return Err(Error::UnknownNode(task.name.clone()));
        }
    }
    let snapshot = Arc::new(channels.to_state::<S>()?);
    let mut views = Vec::with_capacity(scheduled.len());
    for task in &scheduled {
        if task_done(pending, &task.task_id) {
            views.push(None);
            continue;
        }
        let view = match &task.send_value {
            Some(value) => Arc::new(merge_send_view::<S>(channels, value)?),
            None => Arc::clone(&snapshot),
        };
        views.push(Some(view));
    }
    let shared = Arc::new(StepShared {
        base: channels.clone(),
        conditionals: Arc::clone(conditionals),
        known: Arc::clone(known),
    });
    let (null_value, resume_map) = resume_parts(resume);
    let null_owner =
        null_resume_task(&scheduled, pending, null_value.is_some()).map(str::to_string);

    let mut results = Vec::new();
    let mut handles = Vec::new();
    let mut handle_names = Vec::new();
    for (task, view) in scheduled.iter().zip(views) {
        if task_done(pending, &task.task_id) {
            let replay = decode_replay(pending, &task.task_id)?;
            results.push(TaskOutput {
                task_id: task.task_id.clone(),
                writes: replay.writes,
                routed: replay.routed,
                pause: None,
                skip_static: replay.skip_static,
            });
            continue;
        }
        let node = Arc::clone(nodes.get(&task.name).expect("scheduled node is registered"));
        let answers = stored_resume(pending, &task.task_id)?;
        let frame = InterruptFrame::new(TaskResume {
            checkpoint_ns: config.get_checkpoint_ns().to_string(),
            task_path: vec![task.name.clone()],
            has_checkpointer: config.get_checkpointer().is_some(),
            answers,
            null_resume: if Some(task.task_id.as_str()) == null_owner.as_deref() {
                null_value.clone()
            } else {
                None
            },
            resume_map: resume_map.clone(),
        });
        handle_names.push(task.name.clone());
        handles.push(spawn_task(
            node,
            view.expect("running task has an input view"),
            Arc::clone(&shared),
            task.name.clone(),
            task.task_id.clone(),
            frame,
        ));
    }

    let mut failures = Vec::new();
    for (name, handle) in handle_names.into_iter().zip(handles) {
        match handle.await {
            Ok(Ok(output)) => results.push(output),
            Ok(Err(error)) => failures.push((name, error)),
            Err(join_error) if join_error.is_panic() => {
                std::panic::resume_unwind(join_error.into_panic());
            }
            Err(_) => failures.push((
                name.clone(),
                Error::Node {
                    node: name,
                    message: "task was cancelled".into(),
                },
            )),
        }
    }
    if failures
        .iter()
        .any(|(_, error)| matches!(error, Error::ParentGraph))
    {
        return Err(Error::ParentGraph);
    }
    if failures
        .iter()
        .any(|(_, error)| matches!(error, Error::MissingThreadId))
    {
        return Err(Error::MissingThreadId);
    }
    if !failures.is_empty() {
        failures.sort_by(|left, right| left.0.cmp(&right.0));
        return Err(failures.remove(0).1);
    }

    if results.iter().any(|result| result.pause.is_some()) {
        let mut groups = BTreeMap::new();
        let mut interrupts = Vec::new();
        for result in &results {
            if let Some(pause) = &result.pause {
                interrupts.push(pause.interrupt.clone());
                groups.insert(
                    result.task_id.clone(),
                    encode_interrupted(&result.task_id, &pause.interrupt, &pause.resume_values),
                );
            } else {
                groups.insert(
                    result.task_id.clone(),
                    encode_finished(
                        &result.task_id,
                        &result.writes,
                        &result.routed,
                        result.skip_static,
                    ),
                );
            }
        }
        interrupts.sort_by(|left, right| left.id.cmp(&right.id));
        return Ok(StepOutcome::Paused {
            interrupts,
            pending: groups,
        });
    }

    results.sort_by(|left, right| left.task_id.cmp(&right.task_id));
    let mut writes = Vec::new();
    for result in &mut results {
        writes.append(&mut result.writes);
    }
    channels.apply_writes(&writes)?;

    for task in &scheduled {
        if !task.consumed.is_empty() {
            control.mark_consumed(&task.name, &task.consumed);
        }
        let skip_static = results
            .iter()
            .find(|result| result.task_id == task.task_id)
            .is_some_and(|result| result.skip_static);
        if !skip_static {
            control.bump_done(&task.name);
        }
    }
    drop_executed_sends(control, &scheduled);
    for result in &results {
        control.apply_routes(&result.routed);
    }
    Ok(StepOutcome::Done(CommittedStep {
        updated: channel_names(&writes),
        finished: scheduled.into_iter().map(|task| task.name).collect(),
    }))
}

fn resume_parts(resume: Option<&Resume>) -> (Option<Value>, BTreeMap<String, Value>) {
    match resume {
        Some(Resume::Value(value)) => (Some(value.clone()), BTreeMap::new()),
        Some(Resume::Map(map)) => (None, map.clone()),
        None => (None, BTreeMap::new()),
    }
}

fn null_resume_task<'a>(
    tasks: &'a [Scheduled],
    pending: &[PendingWrite],
    has_value: bool,
) -> Option<&'a str> {
    if !has_value {
        return None;
    }
    let running: Vec<&Scheduled> = tasks
        .iter()
        .filter(|task| !task_done(pending, &task.task_id))
        .collect();
    let mut chosen: Vec<&Scheduled> = running
        .iter()
        .copied()
        .filter(|task| task_interrupted(pending, &task.task_id))
        .collect();
    if chosen.is_empty() {
        chosen = running;
    }
    chosen.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.task_id.cmp(&right.task_id))
    });
    chosen.first().map(|task| task.task_id.as_str())
}

fn drop_executed_sends(control: &mut Control, scheduled: &[Scheduled]) {
    let executed = scheduled
        .iter()
        .filter_map(|task| task.send_index)
        .collect::<BTreeSet<_>>();
    if executed.is_empty() {
        return;
    }
    let mut index = 0;
    control.sends.retain(|_| {
        let drop_this = executed.contains(&index);
        index += 1;
        !drop_this
    });
}

fn interrupts_match(
    set: &InterruptSet,
    names: &[String],
    channels: &ChannelMap,
    control: &Control,
) -> bool {
    if !names.iter().any(|name| set.contains(name)) {
        return false;
    }
    channels
        .versions()
        .into_iter()
        .any(|(name, version)| version > control.seen_version(INTERRUPT_SEEN, &name))
}

fn mark_interrupt_seen(control: &mut Control, channels: &ChannelMap) {
    control
        .seen
        .insert(INTERRUPT_SEEN.to_string(), channels.versions());
}

fn breakpoint_writes() -> BTreeMap<String, Vec<PendingWrite>> {
    BTreeMap::from([(
        BREAKPOINT_TASK.to_string(),
        vec![PendingWrite {
            task_id: BREAKPOINT_TASK.to_string(),
            channel: CHANNEL_TASKS.to_string(),
            value: json!({"status": "breakpoint"}),
        }],
    )])
}

async fn put_pending(
    saver: &dyn Checkpointer,
    saved: &CheckpointConfig,
    groups: &BTreeMap<String, Vec<PendingWrite>>,
) -> Result<(), Error> {
    for (task_id, writes) in groups {
        if writes.is_empty() {
            continue;
        }
        saver
            .put_writes(saved, writes, task_id)
            .await
            .map_err(checkpoint_error)?;
    }
    Ok(())
}

fn encode_finished(
    task_id: &str,
    writes: &[Write],
    routes: &[RoutePush],
    skip_static: bool,
) -> Vec<PendingWrite> {
    let mut pending = Vec::new();
    for write in writes {
        pending.push(PendingWrite {
            task_id: task_id.to_string(),
            channel: write.channel.clone(),
            value: write.value.clone(),
        });
    }
    pending.push(PendingWrite {
        task_id: task_id.to_string(),
        channel: CHANNEL_TASKS.to_string(),
        value: json!({
            "status": "done",
            "skip_static": skip_static,
            "routes": routes
                .iter()
                .map(|route| {
                    let mut encoded = json!({
                        "edge_index": route.edge_index,
                        "destination": route.destination,
                    });
                    if let Some(value) = &route.send {
                        encoded["send"] = value.clone();
                    }
                    encoded
                })
                .collect::<Vec<_>>(),
        }),
    });
    pending
}

fn encode_interrupted(
    task_id: &str,
    interrupt: &Interrupt,
    resume_values: &[Value],
) -> Vec<PendingWrite> {
    let mut pending = Vec::new();
    if !resume_values.is_empty() {
        pending.push(PendingWrite {
            task_id: task_id.to_string(),
            channel: CHANNEL_RESUME.to_string(),
            value: Value::Array(resume_values.to_vec()),
        });
    }
    pending.push(PendingWrite {
        task_id: task_id.to_string(),
        channel: CHANNEL_INTERRUPT.to_string(),
        value: json!({
            "id": interrupt.id,
            "value": interrupt.value,
        }),
    });
    pending
}

struct ReplayedTask {
    writes: Vec<Write>,
    routed: Vec<RoutePush>,
    skip_static: bool,
}

fn decode_replay(pending: &[PendingWrite], task_id: &str) -> Result<ReplayedTask, Error> {
    let mut writes = Vec::new();
    let mut routes = Vec::new();
    let mut skip_static = false;
    let mut saw_done = false;
    for write in pending {
        if write.task_id != task_id {
            continue;
        }
        if write.channel == CHANNEL_TASKS {
            saw_done = true;
            routes = parse_routes(&write.value)?;
            skip_static = write
                .value
                .get("skip_static")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            continue;
        }
        if write.channel == CHANNEL_RESUME || write.channel == CHANNEL_INTERRUPT {
            continue;
        }
        let write_index = u64::try_from(writes.len())
            .map_err(|_| Error::InvalidUpdate(format!("task '{task_id}' write index overflow")))?;
        writes.push(Write {
            channel: write.channel.clone(),
            value: write.value.clone(),
            task_path: Vec::new(),
            task_id: task_id.to_string(),
            write_index,
        });
    }
    if !saw_done {
        return Err(Error::Checkpoint(format!(
            "task '{task_id}' is missing its pending result"
        )));
    }
    Ok(ReplayedTask {
        writes,
        routed: routes,
        skip_static,
    })
}

fn parse_routes(value: &Value) -> Result<Vec<RoutePush>, Error> {
    let routes = value
        .get("routes")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Checkpoint("pending task result is missing routes".into()))?;
    let mut parsed = Vec::with_capacity(routes.len());
    for route in routes {
        let edge_index = route
            .get("edge_index")
            .and_then(Value::as_u64)
            .ok_or_else(|| Error::Checkpoint("pending route is missing edge_index".into()))?;
        let destination = route
            .get("destination")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Checkpoint("pending route is missing destination".into()))?;
        let edge_index = usize::try_from(edge_index)
            .map_err(|_| Error::Checkpoint("pending route edge_index does not fit".into()))?;
        let send = match route.get("send") {
            None | Some(Value::Null) => None,
            Some(value) => Some(value.clone()),
        };
        parsed.push(RoutePush {
            edge_index,
            destination: destination.to_string(),
            send,
        });
    }
    Ok(parsed)
}

fn require_runtime() -> Result<(), Error> {
    if tokio::runtime::Handle::try_current().is_err() {
        Err(Error::NoRuntime)
    } else {
        Ok(())
    }
}

fn require_saver(config: &RunConfig) -> Result<&Arc<dyn Checkpointer>, Error> {
    config
        .get_checkpointer()
        .ok_or_else(|| Error::Checkpoint("no checkpointer configured".into()))
}

fn checkpoint_error(error: langgraph_checkpoint::Error) -> Error {
    Error::Checkpoint(error.to_string())
}

fn restore_run<S: GraphState>(checkpoint: &Checkpoint) -> Result<(ChannelMap, Control), Error> {
    let mut channels = ChannelMap::new(S::schema())?;
    channels.restore(&checkpoint.channel_values, &checkpoint.channel_versions)?;
    let control = Control {
        done: checkpoint.done.clone(),
        branch: checkpoint.branch.clone(),
        seen: checkpoint.versions_seen.clone(),
        sends: checkpoint.sends.clone(),
    };
    Ok((channels, control))
}

fn channel_names(writes: &[Write]) -> Vec<String> {
    writes
        .iter()
        .map(|write| write.channel.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn new_versions(channels: &ChannelMap, updated: &[String]) -> Result<BTreeMap<String, u64>, Error> {
    let mut versions = BTreeMap::new();
    for name in updated {
        let version = channels.version(name).ok_or_else(|| {
            Error::Checkpoint(format!("updated channel '{name}' is not in the schema"))
        })?;
        versions.insert(name.clone(), version);
    }
    Ok(versions)
}

struct SavedStep<'a> {
    channels: &'a ChannelMap,
    control: &'a Control,
    updated: &'a [String],
    source: CheckpointSource,
    step: i64,
}

async fn save_checkpoint(
    saver: &dyn Checkpointer,
    config: &RunConfig,
    parent_id: Option<&str>,
    step: SavedStep<'_>,
) -> Result<CheckpointConfig, Error> {
    let mut stored = config.checkpoint_config()?;
    stored.checkpoint_id = parent_id.map(str::to_string);
    let checkpoint = Checkpoint {
        v: 1,
        id: langgraph_checkpoint::new_checkpoint_id(),
        ts: langgraph_checkpoint::checkpoint_timestamp(),
        channel_values: step.channels.values(),
        channel_versions: step.channels.versions(),
        versions_seen: step.control.seen.clone(),
        updated_channels: step.updated.to_vec(),
        done: step.control.done.clone(),
        branch: step.control.branch.clone(),
        sends: step.control.sends.clone(),
    };
    let metadata = CheckpointMetadata {
        source: step.source,
        step: step.step,
        parents: BTreeMap::new(),
    };
    let versions = new_versions(step.channels, step.updated)?;
    saver
        .put(&stored, &checkpoint, &metadata, &versions)
        .await
        .map_err(checkpoint_error)
}

fn incoming_triggers(edges: &[Edge], node: &str, control: &Control) -> Vec<(String, u64)> {
    let mut consumed = BTreeMap::<String, u64>::new();
    for edge in edges {
        if edge.target != node {
            continue;
        }
        let keys = match &edge.source {
            EdgeSource::One(source) => vec![done_key(source)],
            EdgeSource::Wait(sources) => sources.iter().map(|source| done_key(source)).collect(),
        };
        for key in keys {
            let version = control.version(&key);
            consumed.insert(key, version);
        }
    }
    for key in control.branch.keys() {
        if branch_destination(key) == Some(node) {
            consumed.insert(key.clone(), control.version(key));
        }
    }
    consumed.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use serde_json::{json, Value};

    use super::*;
    use crate::{Field, GraphBuilder, Reducer};

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Sample {
        count: i64,
        tag: String,
        items: Vec<String>,
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct SampleUpdate {
        count: Option<i64>,
        tag: Option<String>,
        items: Option<Vec<String>>,
    }

    impl GraphState for Sample {
        type Update = SampleUpdate;

        fn schema() -> &'static [Field] {
            static SCHEMA: &[Field] = &[
                Field {
                    name: "count",
                    reducer: Reducer::Overwrite,
                },
                Field {
                    name: "tag",
                    reducer: Reducer::Overwrite,
                },
                Field {
                    name: "items",
                    reducer: Reducer::Append,
                },
            ];
            SCHEMA
        }

        fn to_channel_values(&self) -> Result<BTreeMap<String, Value>, Error> {
            Ok(BTreeMap::from([
                ("count".to_string(), json!(self.count)),
                ("tag".to_string(), Value::String(self.tag.clone())),
                ("items".to_string(), string_array(&self.items)),
            ]))
        }

        fn update_to_channel_values(
            update: &Self::Update,
        ) -> Result<BTreeMap<String, Value>, Error> {
            let mut values = BTreeMap::new();
            if let Some(count) = update.count {
                values.insert("count".to_string(), json!(count));
            }
            if let Some(tag) = &update.tag {
                values.insert("tag".to_string(), Value::String(tag.clone()));
            }
            if let Some(items) = &update.items {
                values.insert("items".to_string(), string_array(items));
            }
            Ok(values)
        }

        fn from_channel_values(values: &BTreeMap<String, Value>) -> Result<Self, Error> {
            Ok(Sample {
                count: read_i64(values, "count")?,
                tag: read_string(values, "tag")?,
                items: read_items(values)?,
            })
        }
    }

    fn string_array(items: &[String]) -> Value {
        Value::Array(items.iter().cloned().map(Value::String).collect())
    }

    fn read_i64(values: &BTreeMap<String, Value>, key: &str) -> Result<i64, Error> {
        values
            .get(key)
            .and_then(Value::as_i64)
            .ok_or_else(|| Error::InvalidUpdate(format!("channel '{key}' is not an integer")))
    }

    fn read_string(values: &BTreeMap<String, Value>, key: &str) -> Result<String, Error> {
        values
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| Error::InvalidUpdate(format!("channel '{key}' is not a string")))
    }

    fn read_items(values: &BTreeMap<String, Value>) -> Result<Vec<String>, Error> {
        let array = values
            .get("items")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::InvalidUpdate("channel 'items' is not an array".into()))?;
        array
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| Error::InvalidUpdate("items entry is not a string".into()))
            })
            .collect()
    }

    fn sample(count: i64) -> Sample {
        Sample {
            count,
            tag: String::new(),
            items: Vec::new(),
        }
    }

    fn set_count(count: i64) -> SampleUpdate {
        SampleUpdate {
            count: Some(count),
            ..SampleUpdate::default()
        }
    }

    fn set_tag(tag: impl Into<String>) -> SampleUpdate {
        SampleUpdate {
            tag: Some(tag.into()),
            ..SampleUpdate::default()
        }
    }

    fn push_item(item: impl Into<String>) -> SampleUpdate {
        SampleUpdate {
            items: Some(vec![item.into()]),
            ..SampleUpdate::default()
        }
    }

    fn compile(build: impl FnOnce(&mut GraphBuilder<Sample>)) -> crate::CompiledGraph<Sample> {
        let mut builder = GraphBuilder::new();
        build(&mut builder);
        builder.compile().expect("compile")
    }

    fn take_finished(output: crate::RunOutput<Sample>) -> Sample {
        match output {
            crate::RunOutput::Finished { state } => state,
            crate::RunOutput::Interrupted { .. } => panic!("expected Finished"),
        }
    }

    fn assert_send<T: Send>(_: &T) {}

    #[test]
    fn default_run_config_uses_25_supersteps() {
        assert_eq!(DEFAULT_RECURSION_LIMIT, 25);
        assert_eq!(RunConfig::default(), RunConfig::new());
        assert_ne!(RunConfig::new().recursion_limit(1), RunConfig::new());
        assert_eq!(RunConfig::new().get_recursion_limit(), 25);
        assert_eq!(RunConfig::new().get_thread_id(), None);
        assert_eq!(RunConfig::new().get_checkpoint_ns(), "");
        assert_eq!(RunConfig::new().get_checkpoint_id(), None);
        assert!(RunConfig::new().get_checkpointer().is_none());
    }

    #[test]
    fn invoke_future_is_send() {
        let graph = compile(|builder| {
            builder
                .add_node("once", |state: &Sample| Ok(set_count(state.count + 1)))
                .add_edge(START, "once")
                .add_edge("once", END);
        });
        let state = sample(0);
        let config = RunConfig::default();
        let future = graph.invoke(&state, &config);
        assert_send(&future);
    }

    #[test]
    fn invoke_outside_tokio_returns_no_runtime() {
        let graph = compile(|builder| {
            builder
                .add_node("once", |_state: &Sample| Ok(set_count(1)))
                .add_edge(START, "once")
                .add_edge("once", END);
        });
        let state = sample(0);
        let config = RunConfig::default();
        let mut future = std::pin::pin!(graph.invoke(&state, &config));
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(Err(Error::NoRuntime)) => {}
            other => panic!("expected NoRuntime, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn recursion_limit_below_one_is_rejected_before_nodes_run() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_node = Arc::clone(&calls);
        let graph = compile(|builder| {
            builder
                .add_node("once", move |_state: &Sample| {
                    calls_node.fetch_add(1, Ordering::SeqCst);
                    Ok(set_count(1))
                })
                .add_edge(START, "once")
                .add_edge("once", END);
        });
        let error = graph
            .invoke(&sample(0), &RunConfig::new().recursion_limit(0))
            .await
            .expect_err("limit 0");
        assert!(
            matches!(error, Error::GraphValidation(ref message) if message.contains("at least 1")),
            "{error:?}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn input_is_not_a_superstep() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_node = Arc::clone(&calls);
        let graph = compile(|builder| {
            builder
                .add_node("once", move |state: &Sample| {
                    calls_node.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(state.count, 4);
                    assert_eq!(state.tag, "kept");
                    Ok(set_count(state.count + 1))
                })
                .add_edge(START, "once")
                .add_edge("once", END);
        });
        let mut input = sample(4);
        input.tag = "kept".to_string();
        let result = take_finished(
            graph
                .invoke(&input, &RunConfig::new().recursion_limit(1))
                .await
                .expect("one superstep"),
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.count, 5);
        assert_eq!(result.tag, "kept");
    }

    #[tokio::test]
    async fn conditional_branch_sees_the_source_update() {
        let low_ran = Arc::new(AtomicUsize::new(0));
        let low_flag = Arc::clone(&low_ran);
        let graph = compile(|builder| {
            builder
                .add_node("pick", |state: &Sample| {
                    let tag = if state.count >= 0 {
                        "non-negative"
                    } else {
                        "negative"
                    };
                    Ok(set_tag(tag))
                })
                .add_node("high", |_state: &Sample| Ok(set_tag("took-high")))
                .add_node("low", move |_state: &Sample| {
                    low_flag.fetch_add(1, Ordering::SeqCst);
                    Ok(set_tag("took-low"))
                })
                .add_edge(START, "pick")
                .add_conditional_edges("pick", |state: &Sample| {
                    if state.tag == "non-negative" {
                        Ok("high")
                    } else {
                        Ok("low")
                    }
                })
                .add_edge("high", END)
                .add_edge("low", END);
        });
        let result = take_finished(
            graph
                .invoke(&sample(2), &RunConfig::default())
                .await
                .expect("branch"),
        );
        assert_eq!(result.tag, "took-high");
        assert_eq!(low_ran.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn path_map_is_applied_before_the_unknown_node_check() {
        let graph = compile(|builder| {
            builder
                .add_node("pick", |_state: &Sample| Ok(SampleUpdate::default()))
                .add_node("high", |_state: &Sample| Ok(set_tag("mapped")))
                .add_edge(START, "pick")
                .add_conditional_edges_with_path_map(
                    "pick",
                    |_state: &Sample| Ok("left"),
                    [("left", "high"), ("stop", END)],
                )
                .add_edge("high", END);
        });
        let result = take_finished(
            graph
                .invoke(&sample(0), &RunConfig::default())
                .await
                .expect("path map"),
        );
        assert_eq!(result.tag, "mapped");
    }

    #[tokio::test]
    async fn unknown_router_name_fails_at_runtime() {
        let graph = compile(|builder| {
            builder
                .add_node("pick", |_state: &Sample| Ok(SampleUpdate::default()))
                .add_edge(START, "pick")
                .add_conditional_edges("pick", |_state: &Sample| Ok("ghost"));
        });
        let error = graph
            .invoke(&sample(0), &RunConfig::default())
            .await
            .expect_err("unknown node");
        match error {
            Error::UnknownNode(message) => assert!(message.contains("ghost"), "{message}"),
            other => panic!("expected UnknownNode, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn unknown_path_map_label_fails_at_runtime() {
        let graph = compile(|builder| {
            builder
                .add_node("pick", |_state: &Sample| Ok(SampleUpdate::default()))
                .add_node("high", |_state: &Sample| Ok(SampleUpdate::default()))
                .add_edge(START, "pick")
                .add_conditional_edges_with_path_map(
                    "pick",
                    |_state: &Sample| Ok("missing"),
                    [("left", "high")],
                );
        });
        let error = graph
            .invoke(&sample(0), &RunConfig::default())
            .await
            .expect_err("missing label");
        match error {
            Error::UnknownNode(message) => assert!(message.contains("missing"), "{message}"),
            other => panic!("expected UnknownNode, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn router_can_select_several_nodes() {
        let graph = compile(|builder| {
            builder
                .add_node("pick", |_state: &Sample| Ok(SampleUpdate::default()))
                .add_node("left", |_state: &Sample| Ok(push_item("left")))
                .add_node("right", |_state: &Sample| Ok(push_item("right")))
                .add_edge(START, "pick")
                .add_conditional_edges("pick", |_state: &Sample| Ok(vec!["left", "right"]))
                .add_edge("left", END)
                .add_edge("right", END);
        });
        let result = take_finished(
            graph
                .invoke(&sample(0), &RunConfig::default())
                .await
                .expect("fan out"),
        );
        assert_eq!(result.items, ["left".to_string(), "right".to_string()]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn parallel_append_follows_task_order_not_completion_order() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let b_done = Arc::new(AtomicBool::new(false));
        let runtime_thread = std::thread::current().id();

        let order_a = Arc::clone(&order);
        let b_done_a = Arc::clone(&b_done);
        let order_b = Arc::clone(&order);
        let b_done_b = Arc::clone(&b_done);

        let graph = compile(|builder| {
            builder
                .add_node("a", move |_state: &Sample| {
                    assert_ne!(std::thread::current().id(), runtime_thread);
                    let started = Instant::now();
                    while !b_done_a.load(Ordering::SeqCst) {
                        assert!(
                            started.elapsed() < Duration::from_secs(2),
                            "sync node made no progress while waiting for the async node"
                        );
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    order_a.lock().expect("lock").push("a");
                    Ok(push_item("a"))
                })
                .add_node("b", move |_state: &Sample| {
                    let order_b = Arc::clone(&order_b);
                    let b_done_b = Arc::clone(&b_done_b);
                    async move {
                        tokio::task::yield_now().await;
                        order_b.lock().expect("lock").push("b");
                        b_done_b.store(true, Ordering::SeqCst);
                        Ok(push_item("b"))
                    }
                })
                .add_edge(START, "a")
                .add_edge(START, "b")
                .add_edge("a", END)
                .add_edge("b", END);
        });

        let mut input = sample(0);
        input.items = vec!["seed".to_string()];
        let result = take_finished(
            graph
                .invoke(&input, &RunConfig::default())
                .await
                .expect("parallel append"),
        );
        assert_eq!(order.lock().expect("lock").as_slice(), ["b", "a"]);
        assert_eq!(
            result.items,
            ["seed".to_string(), "a".to_string(), "b".to_string()]
        );
    }

    #[tokio::test]
    async fn routers_in_one_step_do_not_see_each_other() {
        let seen = Arc::new(Mutex::new(BTreeMap::new()));
        let seen_a = Arc::clone(&seen);
        let seen_b = Arc::clone(&seen);
        let graph = compile(|builder| {
            builder
                .add_node("a", |_state: &Sample| Ok(push_item("A")))
                .add_node("b", |_state: &Sample| Ok(push_item("B")))
                .add_edge(START, "a")
                .add_edge(START, "b")
                .add_conditional_edges("a", move |state: &Sample| {
                    seen_a
                        .lock()
                        .expect("lock")
                        .insert("a".to_string(), state.items.clone());
                    Ok(END)
                })
                .add_conditional_edges("b", move |state: &Sample| {
                    seen_b
                        .lock()
                        .expect("lock")
                        .insert("b".to_string(), state.items.clone());
                    Ok(END)
                });
        });
        let mut input = sample(0);
        input.items = vec!["seed".to_string()];
        let result = take_finished(
            graph
                .invoke(&input, &RunConfig::default())
                .await
                .expect("isolated routers"),
        );
        let seen = seen.lock().expect("lock");
        assert_eq!(
            seen.get("a").map(Vec::as_slice),
            Some(["seed".to_string(), "A".to_string()].as_slice())
        );
        assert_eq!(
            seen.get("b").map(Vec::as_slice),
            Some(["seed".to_string(), "B".to_string()].as_slice())
        );
        assert_eq!(
            result.items,
            ["seed".to_string(), "A".to_string(), "B".to_string()]
        );
    }

    #[tokio::test]
    async fn cycle_reaches_end() {
        let graph = compile(|builder| {
            builder
                .add_node("tick", |state: &Sample| Ok(set_count(state.count + 1)))
                .add_edge(START, "tick")
                .add_conditional_edges("tick", |state: &Sample| {
                    if state.count >= 3 {
                        Ok(END)
                    } else {
                        Ok("tick")
                    }
                });
        });
        let mut input = sample(0);
        input.tag = "kept".to_string();
        let result = take_finished(
            graph
                .invoke(&input, &RunConfig::default())
                .await
                .expect("cycle ends"),
        );
        assert_eq!(result.count, 3);
        assert_eq!(result.tag, "kept");

        let with_thread = take_finished(
            graph
                .invoke(
                    &input,
                    &RunConfig::new()
                        .thread_id("ignored")
                        .checkpoint_id("also-ignored"),
                )
                .await
                .expect("thread id without a saver"),
        );
        assert_eq!(with_thread.count, 3);
        assert_eq!(with_thread.tag, "kept");
    }

    #[tokio::test]
    async fn cycle_stops_at_the_recursion_limit() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_node = Arc::clone(&calls);
        let graph = compile(|builder| {
            builder
                .add_node("tick", move |state: &Sample| {
                    calls_node.fetch_add(1, Ordering::SeqCst);
                    Ok(set_count(state.count + 1))
                })
                .add_edge(START, "tick")
                .add_conditional_edges("tick", |_state: &Sample| Ok("tick"));
        });
        let error = graph
            .invoke(&sample(0), &RunConfig::new().recursion_limit(2))
            .await
            .expect_err("limit");
        assert_eq!(error, Error::RecursionLimit(2));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn default_limit_allows_25_supersteps_and_refuses_the_next() {
        let stopping = compile(|builder| {
            builder
                .add_node("tick", |state: &Sample| Ok(set_count(state.count + 1)))
                .add_edge(START, "tick")
                .add_conditional_edges("tick", |state: &Sample| {
                    if state.count >= i64::from(DEFAULT_RECURSION_LIMIT) {
                        Ok(END)
                    } else {
                        Ok("tick")
                    }
                });
        });
        let finished = take_finished(
            stopping
                .invoke(&sample(0), &RunConfig::default())
                .await
                .expect("25 steps reach the limit value"),
        );
        assert_eq!(finished.count, i64::from(DEFAULT_RECURSION_LIMIT));

        let calls = Arc::new(AtomicUsize::new(0));
        let calls_node = Arc::clone(&calls);
        let infinite = compile(|builder| {
            builder
                .add_node("tick", move |state: &Sample| {
                    calls_node.fetch_add(1, Ordering::SeqCst);
                    Ok(set_count(state.count + 1))
                })
                .add_edge(START, "tick")
                .add_conditional_edges("tick", |_state: &Sample| Ok("tick"));
        });
        let error = infinite
            .invoke(&sample(0), &RunConfig::default())
            .await
            .expect_err("26th superstep");
        assert_eq!(error, Error::RecursionLimit(DEFAULT_RECURSION_LIMIT));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            usize::try_from(DEFAULT_RECURSION_LIMIT).expect("limit fits")
        );
    }

    #[tokio::test]
    async fn recursion_limit_refuses_the_next_node_before_it_runs() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let calls_a = Arc::clone(&calls);
        let calls_b = Arc::clone(&calls);
        let graph = compile(|builder| {
            builder
                .add_node("a", move |_state: &Sample| {
                    calls_a.lock().expect("lock").push("a");
                    Ok(SampleUpdate::default())
                })
                .add_node("b", move |_state: &Sample| {
                    calls_b.lock().expect("lock").push("b");
                    Ok(SampleUpdate::default())
                })
                .add_edge(START, "a")
                .add_edge("a", "b")
                .add_edge("b", END);
        });
        let error = graph
            .invoke(&sample(0), &RunConfig::new().recursion_limit(1))
            .await
            .expect_err("second superstep");
        assert_eq!(error, Error::RecursionLimit(1));
        assert_eq!(calls.lock().expect("lock").as_slice(), ["a"]);
    }

    #[tokio::test]
    async fn two_overwrite_writers_commit_nothing() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_a = Arc::clone(&seen);
        let seen_b = Arc::clone(&seen);
        let after = Arc::new(AtomicUsize::new(0));
        let after_node = Arc::clone(&after);
        let graph = compile(|builder| {
            builder
                .add_node("a", move |state: &Sample| {
                    seen_a.lock().expect("lock").push(state.count);
                    Ok(set_count(2))
                })
                .add_node("b", move |state: &Sample| {
                    seen_b.lock().expect("lock").push(state.count);
                    Ok(set_count(3))
                })
                .add_node("after", move |_state: &Sample| {
                    after_node.fetch_add(1, Ordering::SeqCst);
                    Ok(set_tag("committed"))
                })
                .add_edge(START, "a")
                .add_edge(START, "b")
                .add_edge("a", "after")
                .add_edge("b", "after")
                .add_edge("after", END);
        });
        let error = graph
            .invoke(&sample(7), &RunConfig::default())
            .await
            .expect_err("two overwrites");
        match error {
            Error::InvalidUpdate(message) => {
                assert!(message.contains("count"), "{message}");
                assert!(message.contains("overwrite"), "{message}");
            }
            other => panic!("expected InvalidUpdate, got {other:?}"),
        }
        let mut observed = seen.lock().expect("lock").clone();
        observed.sort();
        assert_eq!(observed, [7, 7]);
        assert_eq!(after.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn node_error_waits_for_siblings_and_drops_the_step() {
        let a_ran = Arc::new(AtomicUsize::new(0));
        let a_flag = Arc::clone(&a_ran);
        let after = Arc::new(AtomicUsize::new(0));
        let after_node = Arc::clone(&after);
        let graph = compile(|builder| {
            builder
                .add_node("a", move |_state: &Sample| {
                    a_flag.fetch_add(1, Ordering::SeqCst);
                    Ok(set_count(5))
                })
                .add_node("b", |_state: &Sample| -> Result<SampleUpdate, Error> {
                    Err(Error::InvalidUpdate("boom".into()))
                })
                .add_node("after", move |_state: &Sample| {
                    after_node.fetch_add(1, Ordering::SeqCst);
                    Ok(SampleUpdate::default())
                })
                .add_edge(START, "a")
                .add_edge(START, "b")
                .add_edge("a", "after")
                .add_edge("b", "after");
        });
        let error = graph
            .invoke(&sample(1), &RunConfig::default())
            .await
            .expect_err("node failure");
        assert_eq!(
            error,
            Error::Node {
                node: "b".to_string(),
                message: "invalid update: boom".into(),
            }
        );
        assert_eq!(a_ran.load(Ordering::SeqCst), 1);
        assert_eq!(after.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn waiting_edge_runs_after_both_predecessors_from_earlier_steps() {
        let snapshots = Arc::new(Mutex::new(Vec::new()));
        let snapshots_join = Arc::clone(&snapshots);
        let graph = compile(|builder| {
            builder
                .add_node("a", |_state: &Sample| Ok(push_item("a")))
                .add_node("b", |_state: &Sample| Ok(push_item("b")))
                .add_node("join", move |state: &Sample| {
                    snapshots_join
                        .lock()
                        .expect("lock")
                        .push(state.items.clone());
                    Ok(set_tag(state.items.join(",")))
                })
                .add_edge(START, "a")
                .add_edge("a", "b")
                .add_edge(["a", "b"], "join")
                .add_edge("join", END);
        });
        let result = take_finished(
            graph
                .invoke(&sample(0), &RunConfig::default())
                .await
                .expect("waiting edge"),
        );
        assert_eq!(
            snapshots.lock().expect("lock").as_slice(),
            [vec!["a".to_string(), "b".to_string()]]
        );
        assert_eq!(result.items, ["a".to_string(), "b".to_string()]);
        assert_eq!(result.tag, "a,b");
    }

    #[tokio::test]
    async fn waiting_edge_does_not_run_in_the_predecessors_superstep() {
        let snapshots = Arc::new(Mutex::new(Vec::new()));
        let snapshots_join = Arc::clone(&snapshots);
        let graph = compile(|builder| {
            builder
                .add_node("a", |_state: &Sample| Ok(push_item("a")))
                .add_node("b", |_state: &Sample| Ok(push_item("b")))
                .add_node("join", move |state: &Sample| {
                    snapshots_join
                        .lock()
                        .expect("lock")
                        .push(state.items.clone());
                    Ok(set_tag(state.items.join(",")))
                })
                .add_edge(START, "a")
                .add_edge(START, "b")
                .add_edge(["a", "b"], "join")
                .add_edge("join", END);
        });
        let result = take_finished(
            graph
                .invoke(&sample(0), &RunConfig::default())
                .await
                .expect("parallel wait"),
        );
        assert_eq!(
            snapshots.lock().expect("lock").as_slice(),
            [vec!["a".to_string(), "b".to_string()]]
        );
        assert_eq!(result.tag, "a,b");
    }
}
