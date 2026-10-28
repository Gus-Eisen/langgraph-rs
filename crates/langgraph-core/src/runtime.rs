//! Pregel superstep runtime.
//!
//! [`CompiledGraph::invoke`] applies the input as [`START`] writes, then
//! runs scheduled nodes until the frontier is empty or the recursion limit
//! is reached. Control channels are separate from user channels:
//! `done:{node}` counts completions of that node or of `START`, and
//! `branch:{edge}:{dest}` counts conditional selections of `dest`.
//! `versions_seen` maps a node to the control-channel versions it has
//! already consumed. A later checkpointer can persist that map as
//! node to channel to version. This milestone keeps it only for the call.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use tokio::task::JoinHandle;

use crate::builder::{ConditionalEdge, Edge, EdgeSource, InvokeNode};
use crate::state::{ChannelMap, Write};
use crate::{Error, GraphState, END, START};

/// Default superstep budget for [`RunConfig::new`].
///
/// A run may execute this many supersteps. The next one is refused with
/// [`Error::RecursionLimit`]. Input writes are not a superstep.
pub const DEFAULT_RECURSION_LIMIT: u32 = 25;

/// Options for [`CompiledGraph::invoke`].
///
/// Build a value with [`RunConfig::new`] or [`Default`], then
/// [`RunConfig::recursion_limit`] when the default budget is wrong. `invoke`
/// rejects a limit below 1.
///
/// A later checkpointer milestone should add thread and checkpoint identity
/// here, defaulted in [`RunConfig::new`], and should keep
/// [`CompiledGraph::invoke`]'s parameters. New fields with defaults do not
/// break callers that already build a [`RunConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunConfig {
    recursion_limit: u32,
}

impl RunConfig {
    /// Config with [`DEFAULT_RECURSION_LIMIT`].
    pub const fn new() -> Self {
        Self {
            recursion_limit: DEFAULT_RECURSION_LIMIT,
        }
    }

    /// Superstep budget. `invoke` accepts a minimum of 1.
    pub const fn recursion_limit(mut self, limit: u32) -> Self {
        self.recursion_limit = limit;
        self
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
    /// Run the graph to completion on the current tokio runtime.
    ///
    /// `input` is written as channel updates attributed to [`START`], through
    /// the normal reducers. That write is not a superstep. With no
    /// checkpointer this is the whole state of a fresh run: the map starts
    /// empty, then the input is folded in. A later saver should load the
    /// checkpoint's channels first and then apply these same writes.
    ///
    /// Scheduling uses control channels, not the user-channel versions. A
    /// normal edge fires after its source completes. [`START`] fires the
    /// entry nodes. A waiting edge fires after every listed source has a
    /// completion newer than the one the target last consumed. Those sources
    /// need not have finished in the same superstep.
    ///
    /// A superstep runs every scheduled node at once. Each node reads the
    /// committed snapshot from the start of the step. Sync nodes are called
    /// on `spawn_blocking`. Async nodes are awaited on this runtime. After a
    /// node returns, each of its conditional routers runs inside that task.
    /// A router sees `reduce(snapshot, this_node_update)` and does not see
    /// other tasks from this step. A static path map is applied to the
    /// router's labels before names are checked. An unknown name is
    /// [`Error::UnknownNode`].
    ///
    /// The barrier folds the step's writes in `(task_path, task_id, write_index)`
    /// order. `task_path` is empty at the root. `task_id` is the node name.
    /// A task [`Err`](Result::Err) becomes [`Error::Node`]: the other tasks
    /// are joined, and the step commits nothing. Two overwrite writes in one
    /// step are [`Error::InvalidUpdate`] and also commit nothing.
    ///
    /// The recursion limit counts supersteps. The run executes at most
    /// `config`'s limit and returns [`Error::RecursionLimit`] before starting
    /// the next one. Routers are not a separate superstep.
    ///
    /// Polling this future outside tokio returns [`Error::NoRuntime`]. A
    /// finished run returns the committed state. There is no `stream` method,
    /// checkpoint, interrupt, `Send`, or `Command` on this call.
    ///
    /// Checkpoint configuration extends [`RunConfig`]. Do not add a required
    /// parameter to this method for it.
    pub async fn invoke(&self, input: &S, config: &RunConfig) -> Result<S, Error> {
        if tokio::runtime::Handle::try_current().is_err() {
            return Err(Error::NoRuntime);
        }
        if config.recursion_limit < 1 {
            return Err(Error::GraphValidation(
                "recursion limit must be at least 1".into(),
            ));
        }

        let mut channels = ChannelMap::new(S::schema())?;
        channels.apply_writes(&input_writes(input)?)?;

        let nodes = node_map(&self.nodes);
        let known = Arc::new(nodes.keys().cloned().collect::<BTreeSet<_>>());
        let conditionals = Arc::new(self.conditional_edges.clone());
        let mut control = Control::new();
        // Input counts as one START completion. Entry edges read this version.
        control.done.insert(START.to_string(), 1);
        if conditionals.iter().any(|edge| edge.from == START) {
            let state = channels.to_state::<S>()?;
            let routes = route_from(conditionals.as_slice(), START, &state, known.as_ref())?;
            control.apply_routes(&routes);
        }

        let mut completed_supersteps = 0u32;
        loop {
            let scheduled = schedule(&self.edges, &nodes, &control);
            if scheduled.is_empty() {
                return channels.to_state();
            }
            if completed_supersteps == config.recursion_limit {
                return Err(Error::RecursionLimit(config.recursion_limit));
            }
            run_superstep(
                &mut channels,
                &mut control,
                &nodes,
                &conditionals,
                &known,
                scheduled,
            )
            .await?;
            completed_supersteps += 1;
        }
    }
}

struct Control {
    done: BTreeMap<String, u64>,
    branch: BTreeMap<String, u64>,
    seen: BTreeMap<String, BTreeMap<String, u64>>,
}

impl Control {
    fn new() -> Self {
        Self {
            done: BTreeMap::new(),
            branch: BTreeMap::new(),
            seen: BTreeMap::new(),
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
}

struct Scheduled {
    name: String,
    consumed: Vec<(String, u64)>,
}

struct TaskOutput {
    name: String,
    writes: Vec<Write>,
    routed: Vec<RoutePush>,
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
                consumed,
            });
        }
    }
    scheduled
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
        let labels = edge.router.route(state)?;
        for label in labels {
            let destination = match &edge.path_map {
                Some(path_map) => path_map
                    .get(&label)
                    .cloned()
                    .ok_or_else(|| Error::UnknownNode(format!("path map label '{label}'")))?,
                None => label,
            };
            if destination != END && !known.contains(&destination) {
                return Err(Error::UnknownNode(destination));
            }
            if destination != END {
                routed.push(RoutePush {
                    edge_index,
                    destination,
                });
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

fn after_update<S: GraphState>(
    name: String,
    update: S::Update,
    shared: &StepShared<S>,
) -> Result<TaskOutput, Error> {
    let writes = S::update_to_writes(&update, &[], &name)?;
    let routed = if shared.conditionals.iter().any(|edge| edge.from == name) {
        let preview = preview_state::<S>(&shared.base, &writes)?;
        route_from(
            shared.conditionals.as_slice(),
            &name,
            &preview,
            shared.known.as_ref(),
        )?
    } else {
        Vec::new()
    };
    Ok(TaskOutput {
        name,
        writes,
        routed,
    })
}

fn node_failure(name: &str, error: Error) -> Error {
    Error::Node {
        node: name.to_string(),
        message: error.to_string(),
    }
}

fn drive_sync<S: GraphState>(node: &dyn InvokeNode<S>, state: &S) -> Result<S::Update, Error> {
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

fn spawn_task<S>(
    node: Arc<dyn InvokeNode<S>>,
    snapshot: Arc<S>,
    shared: Arc<StepShared<S>>,
    name: String,
) -> JoinHandle<Result<TaskOutput, Error>>
where
    S: GraphState + Send + Sync + 'static,
    S::Update: Send + 'static,
{
    if node.is_sync() {
        tokio::task::spawn_blocking(move || {
            let update = match drive_sync(node.as_ref(), snapshot.as_ref()) {
                Ok(update) => update,
                Err(error) => return Err(node_failure(&name, error)),
            };
            after_update(name, update, shared.as_ref())
        })
    } else {
        tokio::spawn(async move {
            let update = match node.call(snapshot.as_ref()).await {
                Ok(update) => update,
                Err(error) => return Err(node_failure(&name, error)),
            };
            after_update(name, update, shared.as_ref())
        })
    }
}

async fn run_superstep<S>(
    channels: &mut ChannelMap,
    control: &mut Control,
    nodes: &BTreeMap<String, Arc<dyn InvokeNode<S>>>,
    conditionals: &Arc<Vec<ConditionalEdge<S>>>,
    known: &Arc<BTreeSet<String>>,
    scheduled: Vec<Scheduled>,
) -> Result<(), Error>
where
    S: GraphState + Send + Sync + 'static,
    S::Update: Send + 'static,
{
    let snapshot = Arc::new(channels.to_state::<S>()?);
    let shared = Arc::new(StepShared {
        base: channels.clone(),
        conditionals: Arc::clone(conditionals),
        known: Arc::clone(known),
    });

    let mut handles = Vec::with_capacity(scheduled.len());
    for task in &scheduled {
        let node = Arc::clone(nodes.get(&task.name).expect("scheduled node is registered"));
        handles.push(spawn_task(
            node,
            Arc::clone(&snapshot),
            Arc::clone(&shared),
            task.name.clone(),
        ));
    }

    let mut results = Vec::with_capacity(scheduled.len());
    let mut failures = Vec::new();
    for (task, handle) in scheduled.iter().zip(handles) {
        match handle.await {
            Ok(Ok(output)) => results.push(output),
            Ok(Err(error)) => failures.push((task.name.clone(), error)),
            Err(join_error) if join_error.is_panic() => {
                std::panic::resume_unwind(join_error.into_panic());
            }
            Err(_) => failures.push((
                task.name.clone(),
                Error::Node {
                    node: task.name.clone(),
                    message: "task was cancelled".into(),
                },
            )),
        }
    }
    if !failures.is_empty() {
        failures.sort_by(|left, right| left.0.cmp(&right.0));
        return Err(failures.remove(0).1);
    }

    results.sort_by(|left, right| left.name.cmp(&right.name));
    let mut writes = Vec::new();
    for result in &mut results {
        writes.append(&mut result.writes);
    }
    channels.apply_writes(&writes)?;

    for task in &scheduled {
        control.mark_consumed(&task.name, &task.consumed);
        control.bump_done(&task.name);
    }
    for result in &results {
        control.apply_routes(&result.routed);
    }
    Ok(())
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

    fn assert_send<T: Send>(_: &T) {}

    #[test]
    fn default_run_config_uses_25_supersteps() {
        assert_eq!(DEFAULT_RECURSION_LIMIT, 25);
        assert_eq!(RunConfig::default(), RunConfig::new());
        assert_ne!(RunConfig::new().recursion_limit(1), RunConfig::new());
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
        let result = graph
            .invoke(&input, &RunConfig::new().recursion_limit(1))
            .await
            .expect("one superstep");
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
        let result = graph
            .invoke(&sample(2), &RunConfig::default())
            .await
            .expect("branch");
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
        let result = graph
            .invoke(&sample(0), &RunConfig::default())
            .await
            .expect("path map");
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
        let result = graph
            .invoke(&sample(0), &RunConfig::default())
            .await
            .expect("fan out");
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
        let result = graph
            .invoke(&input, &RunConfig::default())
            .await
            .expect("parallel append");
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
        let result = graph
            .invoke(&input, &RunConfig::default())
            .await
            .expect("isolated routers");
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
        let result = graph
            .invoke(&input, &RunConfig::default())
            .await
            .expect("cycle ends");
        assert_eq!(result.count, 3);
        assert_eq!(result.tag, "kept");
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
        let finished = stopping
            .invoke(&sample(0), &RunConfig::default())
            .await
            .expect("25 steps reach the limit value");
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
                .add_node("b", |_state: &Sample| {
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
        let result = graph
            .invoke(&sample(0), &RunConfig::default())
            .await
            .expect("waiting edge");
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
        let result = graph
            .invoke(&sample(0), &RunConfig::default())
            .await
            .expect("parallel wait");
        assert_eq!(
            snapshots.lock().expect("lock").as_slice(),
            [vec!["a".to_string(), "b".to_string()]]
        );
        assert_eq!(result.tag, "a,b");
    }
}
