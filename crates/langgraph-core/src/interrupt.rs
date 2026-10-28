//! Interrupts and resume values.
//!
//! [`interrupt`] is called from inside a node. The first unanswered call
//! records an [`Interrupt`] and pauses that task. [`CompiledGraph::invoke`]
//! turns the pause into [`RunOutput::Interrupted`]. On resume the node
//! function runs again from the start. Each [`interrupt`] call returns the
//! next stored resume value for that task.
//!
//! The id is a UUID v5 of `(checkpoint_ns, task_path, call_index)`. At the
//! root, `task_path` is the node name. `call_index` counts [`interrupt`]
//! calls in that task from zero.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use langgraph_checkpoint::PendingWrite;
use serde_json::Value;
use uuid::Uuid;

use crate::Error;

/// `versions_seen` key that records channel versions at the last breakpoint.
pub(crate) const INTERRUPT_SEEN: &str = "__interrupt__";

/// Pending-write channel for an unanswered [`Interrupt`].
pub(crate) const CHANNEL_INTERRUPT: &str = "__interrupt__";

/// Pending-write channel for resume values already consumed by a task.
pub(crate) const CHANNEL_RESUME: &str = "__resume__";

/// Pending-write channel for a finished task or a breakpoint marker.
pub(crate) const CHANNEL_TASKS: &str = "__pregel_tasks";

/// `task_id` of a breakpoint marker. It is not a node name.
pub(crate) const BREAKPOINT_TASK: &str = "__interrupt__";

/// A pause recorded by [`interrupt`].
///
/// `id` does not change when the same task pauses again on the same call.
/// [`Resume::Map`] addresses that id.
#[derive(Debug, Clone, PartialEq)]
pub struct Interrupt {
    /// Stable id for `(checkpoint_ns, task_path, call_index)`.
    pub id: String,
    /// JSON value passed to [`interrupt`].
    pub value: Value,
}

/// Value that fills an unanswered [`interrupt`] call.
///
/// [`Resume::Value`] fills the next unanswered call. [`Resume::Map`] fills
/// the call whose id is the key. When a run config carries a resume value,
/// [`crate::CompiledGraph::invoke`] does not also apply `input` as channel
/// writes.
#[derive(Debug, Clone, PartialEq)]
pub enum Resume {
    /// One value for the next unanswered interrupt.
    Value(Value),
    /// Interrupt id to the value that call returns.
    Map(BTreeMap<String, Value>),
}

/// Successful result of [`crate::CompiledGraph::invoke`].
///
/// A pause is [`RunOutput::Interrupted`], not an [`Err`](Result::Err).
#[derive(Debug, Clone, PartialEq)]
pub enum RunOutput<S> {
    /// Every scheduled task finished and channel writes were committed.
    Finished {
        /// Committed state.
        state: S,
    },
    /// The run paused. Channel writes from an interrupted superstep are not
    /// in `state`.
    Interrupted {
        /// Last committed state.
        state: S,
        /// Unanswered interrupts. Empty for a breakpoint.
        interrupts: Vec<Interrupt>,
    },
}

/// UUID v5 of `(checkpoint_ns, task_path, call_index)`.
///
/// `task_path` at the root is one segment, the node name. The same inputs
/// always return the same id.
pub fn interrupt_id(checkpoint_ns: &str, task_path: &[String], call_index: u32) -> String {
    let path = task_path.join("\u{1f}");
    let name = format!("langgraph-rs-interrupt\0{checkpoint_ns}\0{path}\0{call_index}");
    Uuid::new_v5(&Uuid::NAMESPACE_OID, name.as_bytes()).to_string()
}

/// Pause the current node until a resume value is available for this call.
///
/// Call this from a node and propagate [`Err`] with `?`. The first call that
/// has no stored answer records an [`Interrupt`] and pauses the task. Sibling
/// tasks in the superstep are still joined. Their writes are kept with
/// `put_writes` and are not applied until every task in the superstep
/// finishes without a new interrupt.
///
/// On resume the node function runs from the start. Calls are matched to
/// resume values in order. A further unanswered call pauses again.
///
/// There is no checkpointer when this returns [`Error::MissingThreadId`].
/// That failing superstep does not commit. Calling this outside a node
/// returns the same error.
pub fn interrupt(value: impl Into<Value>) -> Result<Value, Error> {
    let value = value.into();
    match current_frame() {
        Some(frame) => frame.on_interrupt(value),
        None => Err(Error::MissingThreadId),
    }
}

/// True when `pending` is a pause checkpoint.
pub(crate) fn is_pause(pending: &[PendingWrite]) -> bool {
    pending.iter().any(is_breakpoint_marker) || has_dynamic_interrupt(pending)
}

/// True for the marker written by a breakpoint pause.
pub(crate) fn is_breakpoint_marker(write: &PendingWrite) -> bool {
    write.task_id == BREAKPOINT_TASK && write.channel == CHANNEL_TASKS
}

/// True when a task paused inside [`interrupt`].
pub(crate) fn has_dynamic_interrupt(pending: &[PendingWrite]) -> bool {
    pending
        .iter()
        .any(|write| write.channel == CHANNEL_INTERRUPT)
}

/// Unanswered interrupts stored on a checkpoint, sorted by id.
pub(crate) fn dynamic_interrupts(pending: &[PendingWrite]) -> Result<Vec<Interrupt>, Error> {
    let mut interrupts = Vec::new();
    for write in pending {
        if write.channel != CHANNEL_INTERRUPT {
            continue;
        }
        interrupts.push(parse_interrupt(&write.value)?);
    }
    interrupts.sort_by(|left, right| left.id.cmp(&right.id));
    interrupts.dedup_by(|left, right| left.id == right.id);
    Ok(interrupts)
}

/// True when `task_id` finished and is waiting for the superstep to commit.
pub(crate) fn task_done(pending: &[PendingWrite], task_id: &str) -> bool {
    !task_interrupted(pending, task_id)
        && pending
            .iter()
            .any(|write| write.task_id == task_id && write.channel == CHANNEL_TASKS)
}

/// True when `task_id` has an unanswered [`interrupt`] call.
pub(crate) fn task_interrupted(pending: &[PendingWrite], task_id: &str) -> bool {
    pending
        .iter()
        .any(|write| write.task_id == task_id && write.channel == CHANNEL_INTERRUPT)
}

/// Resume values already consumed by `task_id`, in call order.
pub(crate) fn stored_resume(pending: &[PendingWrite], task_id: &str) -> Result<Vec<Value>, Error> {
    let mut found = None;
    for write in pending {
        if write.task_id == task_id && write.channel == CHANNEL_RESUME {
            let values = write.value.as_array().ok_or_else(|| {
                Error::Checkpoint(format!(
                    "resume values for task '{task_id}' are not an array"
                ))
            })?;
            found = Some(values.clone());
        }
    }
    Ok(found.unwrap_or_default())
}

pub(crate) type FrameHandle = Arc<InterruptFrame>;

/// Resume inputs for one execution of a task.
pub(crate) struct TaskResume {
    pub checkpoint_ns: String,
    pub task_path: Vec<String>,
    pub has_checkpointer: bool,
    pub answers: Vec<Value>,
    pub null_resume: Option<Value>,
    pub resume_map: BTreeMap<String, Value>,
}

pub(crate) struct InterruptFrame {
    checkpoint_ns: String,
    task_path: Vec<String>,
    has_checkpointer: bool,
    answers: Mutex<Vec<Value>>,
    null_resume: Mutex<Option<Value>>,
    resume_map: BTreeMap<String, Value>,
    next_index: AtomicU32,
    raised: Mutex<Option<Interrupt>>,
}

impl InterruptFrame {
    pub(crate) fn new(task: TaskResume) -> FrameHandle {
        Arc::new(Self {
            checkpoint_ns: task.checkpoint_ns,
            task_path: task.task_path,
            has_checkpointer: task.has_checkpointer,
            answers: Mutex::new(task.answers),
            null_resume: Mutex::new(task.null_resume),
            resume_map: task.resume_map,
            next_index: AtomicU32::new(0),
            raised: Mutex::new(None),
        })
    }

    fn on_interrupt(&self, value: Value) -> Result<Value, Error> {
        if !self.has_checkpointer {
            return Err(Error::MissingThreadId);
        }
        let index = self.next_index.fetch_add(1, Ordering::SeqCst);
        if let Some(answer) = self.answers().get(index as usize).cloned() {
            return Ok(answer);
        }
        let id = interrupt_id(&self.checkpoint_ns, &self.task_path, index);
        if let Some(answer) = self.resume_map.get(&id).cloned() {
            self.push_answer(answer.clone());
            return Ok(answer);
        }
        if index as usize == self.answers().len() {
            if let Some(answer) = self.null_resume.lock().expect("null resume").take() {
                self.push_answer(answer.clone());
                return Ok(answer);
            }
        }
        *self.raised.lock().expect("interrupt") = Some(Interrupt { id, value });
        Err(Error::GraphInterrupt)
    }

    fn answers(&self) -> std::sync::MutexGuard<'_, Vec<Value>> {
        self.answers.lock().expect("interrupt answers")
    }

    fn push_answer(&self, answer: Value) {
        self.answers().push(answer);
    }

    pub(crate) fn take_raised(&self) -> Option<Interrupt> {
        self.raised.lock().expect("interrupt").take()
    }

    pub(crate) fn resume_values(&self) -> Vec<Value> {
        self.answers().clone()
    }
}

tokio::task_local! {
    static INTERRUPT_FRAME: FrameHandle;
}

thread_local! {
    static THREAD_FRAME: RefCell<Option<FrameHandle>> = const { RefCell::new(None) };
}

fn current_frame() -> Option<FrameHandle> {
    if let Ok(handle) = INTERRUPT_FRAME.try_with(Clone::clone) {
        return Some(handle);
    }
    THREAD_FRAME.with(|slot| slot.borrow().clone())
}

/// Run `future` with `frame` visible to [`interrupt`].
pub(crate) fn scope_task<F>(
    frame: FrameHandle,
    future: F,
) -> impl std::future::Future<Output = F::Output> + Send + 'static
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    INTERRUPT_FRAME.scope(frame, future)
}

struct ClearThreadFrame;

impl Drop for ClearThreadFrame {
    fn drop(&mut self) {
        THREAD_FRAME.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}

/// Run `body` on this thread with `frame` visible to [`interrupt`].
///
/// Sync nodes call this inside `spawn_blocking`. The guard clears the thread
/// local when `body` returns or panics.
pub(crate) fn with_thread_frame<R>(frame: FrameHandle, body: impl FnOnce() -> R) -> R {
    THREAD_FRAME.with(|slot| {
        *slot.borrow_mut() = Some(frame);
    });
    let _guard = ClearThreadFrame;
    body()
}

fn parse_interrupt(value: &Value) -> Result<Interrupt, Error> {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Checkpoint("pending interrupt is missing id".into()))?
        .to_string();
    let payload = value
        .get("value")
        .cloned()
        .ok_or_else(|| Error::Checkpoint("pending interrupt is missing value".into()))?;
    Ok(Interrupt { id, value: payload })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interrupt_id_is_stable_for_the_same_inputs() {
        let path = vec!["ask".to_string()];
        let first = interrupt_id("", &path, 0);
        let again = interrupt_id("", &path, 0);
        assert_eq!(first, again);
        assert_ne!(first, interrupt_id("", &path, 1));
        assert_ne!(first, interrupt_id("child", &path, 0));
        assert_ne!(first, interrupt_id("", &["other".to_string()], 0));
        assert!(Uuid::parse_str(&first).is_ok());
    }

    #[test]
    fn interrupt_outside_a_node_is_missing_thread_id() {
        let error = interrupt(Value::Null).expect_err("no frame");
        assert_eq!(error, Error::MissingThreadId);
    }
}
