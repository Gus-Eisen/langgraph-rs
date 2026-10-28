//! Breakpoints, `interrupt`, and `Resume`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use langgraph::{
    interrupt, interrupt_id, CheckpointConfig, Checkpointer, Error, Field, GraphBuilder,
    GraphState, InMemorySaver, InterruptNodes, Reducer, Resume, RunConfig, RunOutput, SqliteSaver,
    END, START,
};
use serde_json::{json, Value};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Gate {
    n: i64,
    note: String,
    items: Vec<String>,
}

#[derive(Clone, Debug, Default)]
struct GateUpdate {
    n: Option<i64>,
    note: Option<String>,
    items: Option<Vec<String>>,
}

impl GraphState for Gate {
    type Update = GateUpdate;

    fn schema() -> &'static [Field] {
        static SCHEMA: &[Field] = &[
            Field {
                name: "n",
                reducer: Reducer::Overwrite,
            },
            Field {
                name: "note",
                reducer: Reducer::Overwrite,
            },
            Field {
                name: "items",
                reducer: Reducer::Append,
            },
        ];
        SCHEMA
    }

    fn to_channel_values(&self) -> Result<std::collections::BTreeMap<String, Value>, Error> {
        Ok(std::collections::BTreeMap::from([
            ("n".to_string(), json!(self.n)),
            ("note".to_string(), Value::String(self.note.clone())),
            ("items".to_string(), string_array(&self.items)),
        ]))
    }

    fn update_to_channel_values(
        update: &Self::Update,
    ) -> Result<std::collections::BTreeMap<String, Value>, Error> {
        let mut values = std::collections::BTreeMap::new();
        if let Some(n) = update.n {
            values.insert("n".to_string(), json!(n));
        }
        if let Some(note) = &update.note {
            values.insert("note".to_string(), Value::String(note.clone()));
        }
        if let Some(items) = &update.items {
            values.insert("items".to_string(), string_array(items));
        }
        Ok(values)
    }

    fn from_channel_values(
        values: &std::collections::BTreeMap<String, Value>,
    ) -> Result<Self, Error> {
        let n = values
            .get("n")
            .and_then(Value::as_i64)
            .ok_or_else(|| Error::InvalidUpdate("channel 'n' is not an integer".into()))?;
        let note = values
            .get("note")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let items = values
            .get("items")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self { n, note, items })
    }
}

fn string_array(items: &[String]) -> Value {
    Value::Array(items.iter().cloned().map(Value::String).collect())
}

fn gate(n: i64) -> Gate {
    Gate {
        n,
        note: String::new(),
        items: Vec::new(),
    }
}

fn memory_config(thread_id: &str) -> (Arc<dyn Checkpointer>, RunConfig) {
    let saver: Arc<dyn Checkpointer> = Arc::new(InMemorySaver::new());
    let config = RunConfig::new()
        .checkpointer(Arc::clone(&saver))
        .thread_id(thread_id);
    (saver, config)
}

fn finished(output: RunOutput<Gate>) -> Gate {
    match output {
        RunOutput::Finished { state } => state,
        RunOutput::Interrupted { .. } => panic!("expected Finished"),
    }
}

fn interrupted(output: RunOutput<Gate>) -> (Gate, Vec<langgraph::Interrupt>) {
    match output {
        RunOutput::Interrupted { state, interrupts } => (state, interrupts),
        RunOutput::Finished { .. } => panic!("expected Interrupted"),
    }
}

#[tokio::test]
async fn breakpoint_before_pauses_and_resumes_with_no_value() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_node = Arc::clone(&calls);
    let mut builder = GraphBuilder::<Gate>::new();
    builder
        .add_node("gate", move |state: &Gate| {
            calls_node.fetch_add(1, Ordering::SeqCst);
            Ok(GateUpdate {
                n: Some(state.n + 1),
                ..GateUpdate::default()
            })
        })
        .add_node("after", |_state: &Gate| {
            Ok(GateUpdate {
                note: Some("after".to_string()),
                ..GateUpdate::default()
            })
        })
        .add_edge(START, "gate")
        .add_edge("gate", "after")
        .add_edge("after", END);
    let graph = builder
        .compile_with(["gate"], InterruptNodes::none())
        .expect("compile");
    let (_saver, config) = memory_config("before");
    let (state, interrupts) = interrupted(
        graph
            .invoke(&gate(4), &config)
            .await
            .expect("pause before the node"),
    );
    assert!(interrupts.is_empty());
    assert_eq!(state.n, 4);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let snapshot = graph.get_state(&config).await.expect("state");
    assert!(snapshot.interrupts.is_empty());
    assert_eq!(snapshot.next, ["gate".to_string()]);

    let resumed = finished(graph.invoke(&gate(99), &config).await.expect("resume"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(resumed.n, 5);
    assert_eq!(resumed.note, "after");
}

#[tokio::test]
async fn breakpoint_before_pauses_again_when_the_node_is_scheduled_again() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_node = Arc::clone(&calls);
    let mut builder = GraphBuilder::<Gate>::new();
    builder
        .add_node("tick", move |state: &Gate| {
            calls_node.fetch_add(1, Ordering::SeqCst);
            Ok(GateUpdate {
                n: Some(state.n + 1),
                ..GateUpdate::default()
            })
        })
        .add_edge(START, "tick")
        .add_conditional_edges(
            "tick",
            |state: &Gate| {
                if state.n >= 2 {
                    Ok(END)
                } else {
                    Ok("tick")
                }
            },
        );
    let graph = builder
        .compile_with(["tick"], InterruptNodes::none())
        .expect("compile");
    let (_saver, config) = memory_config("before-loop");

    let (state, interrupts) = interrupted(graph.invoke(&gate(0), &config).await.expect("first"));
    assert!(interrupts.is_empty());
    assert_eq!(state.n, 0);
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let (state, _) = interrupted(graph.invoke(&gate(8), &config).await.expect("second"));
    assert_eq!(state.n, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let resumed = finished(graph.invoke(&gate(8), &config).await.expect("third"));
    assert_eq!(resumed.n, 2);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn async_interrupt_uses_the_task_resume_value() {
    let mut builder = GraphBuilder::<Gate>::new();
    builder
        .add_node("ask", |_state: &Gate| async {
            let answer = interrupt(json!("async"))?;
            Ok(GateUpdate {
                note: Some(answer.as_str().unwrap_or("").to_string()),
                ..GateUpdate::default()
            })
        })
        .add_edge(START, "ask")
        .add_edge("ask", END);
    let graph = builder.compile().expect("compile");
    let (_saver, config) = memory_config("async");
    let (_state, interrupts) = interrupted(graph.invoke(&gate(0), &config).await.expect("pause"));
    assert_eq!(interrupts[0].value, json!("async"));
    let resumed = finished(
        graph
            .invoke(&gate(4), &config.resume(Resume::Value(json!("yes"))))
            .await
            .expect("resume"),
    );
    assert_eq!(resumed.note, "yes");
    assert_eq!(resumed.n, 0);
}

#[tokio::test]
async fn breakpoint_after_pauses_once_the_node_has_committed() {
    let gate_calls = Arc::new(AtomicUsize::new(0));
    let after_calls = Arc::new(AtomicUsize::new(0));
    let gate_node = Arc::clone(&gate_calls);
    let after_node = Arc::clone(&after_calls);
    let mut builder = GraphBuilder::<Gate>::new();
    builder
        .add_node("gate", move |state: &Gate| {
            gate_node.fetch_add(1, Ordering::SeqCst);
            Ok(GateUpdate {
                n: Some(state.n + 1),
                ..GateUpdate::default()
            })
        })
        .add_node("after", move |_state: &Gate| {
            after_node.fetch_add(1, Ordering::SeqCst);
            Ok(GateUpdate {
                note: Some("after".to_string()),
                ..GateUpdate::default()
            })
        })
        .add_edge(START, "gate")
        .add_edge("gate", "after")
        .add_edge("after", END);
    let graph = builder
        .compile_with(InterruptNodes::none(), ["gate"])
        .expect("compile");
    let (_saver, config) = memory_config("after");
    let (state, interrupts) = interrupted(graph.invoke(&gate(1), &config).await.expect("pause"));
    assert!(interrupts.is_empty());
    assert_eq!(state.n, 2);
    assert_eq!(gate_calls.load(Ordering::SeqCst), 1);
    assert_eq!(after_calls.load(Ordering::SeqCst), 0);

    let resumed = finished(graph.invoke(&gate(50), &config).await.expect("resume"));
    assert_eq!(gate_calls.load(Ordering::SeqCst), 1);
    assert_eq!(after_calls.load(Ordering::SeqCst), 1);
    assert_eq!(resumed.n, 2);
    assert_eq!(resumed.note, "after");
}

#[tokio::test]
async fn breakpoint_without_a_checkpointer_is_missing_thread_id() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_node = Arc::clone(&calls);
    let mut builder = GraphBuilder::<Gate>::new();
    builder
        .add_node("gate", move |_state: &Gate| {
            calls_node.fetch_add(1, Ordering::SeqCst);
            Ok(GateUpdate::default())
        })
        .add_edge(START, "gate")
        .add_edge("gate", END);
    let before = builder
        .compile_with(InterruptNodes::All, InterruptNodes::none())
        .expect("before");
    let error = before
        .invoke(&gate(0), &RunConfig::default())
        .await
        .expect_err("no saver");
    assert_eq!(error, Error::MissingThreadId);
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let calls_after = Arc::clone(&calls);
    let mut builder = GraphBuilder::<Gate>::new();
    builder
        .add_node("gate", move |_state: &Gate| {
            calls_after.fetch_add(1, Ordering::SeqCst);
            Ok(GateUpdate::default())
        })
        .add_edge(START, "gate")
        .add_edge("gate", END);
    let after = builder
        .compile_with(InterruptNodes::none(), InterruptNodes::All)
        .expect("after");
    let error = after
        .invoke(&gate(0), &RunConfig::new().thread_id("ignored"))
        .await
        .expect_err("no saver after");
    assert_eq!(error, Error::MissingThreadId);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn one_interrupt_pauses_and_resumes_with_a_value() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_node = Arc::clone(&calls);
    let mut builder = GraphBuilder::<Gate>::new();
    builder
        .add_node("ask", move |state: &Gate| {
            calls_node.fetch_add(1, Ordering::SeqCst);
            let answer = interrupt(json!("what?"))?;
            Ok(GateUpdate {
                note: Some(format!("{}:{}", state.n, answer.as_str().unwrap_or(""))),
                ..GateUpdate::default()
            })
        })
        .add_edge(START, "ask")
        .add_edge("ask", END);
    let graph = builder.compile().expect("compile");
    let (saver, config) = memory_config("one");
    let (state, interrupts) = interrupted(graph.invoke(&gate(1), &config).await.expect("pause"));
    assert_eq!(state.n, 1);
    assert_eq!(state.note, "");
    assert_eq!(interrupts.len(), 1);
    assert_eq!(interrupts[0].value, json!("what?"));
    assert_eq!(interrupts[0].id, interrupt_id("", &["ask".to_string()], 0));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let snapshot = graph.get_state(&config).await.expect("state");
    assert_eq!(snapshot.interrupts, interrupts);
    assert_eq!(snapshot.values.n, 1);

    let again = interrupted(
        graph
            .invoke(&gate(99), &config)
            .await
            .expect("same pause without a resume value"),
    );
    assert_eq!(again.1[0].id, interrupts[0].id);
    assert_eq!(again.0.n, 1);

    let resumed = finished(
        graph
            .invoke(
                &gate(99),
                &config.clone().resume(Resume::Value(json!("human"))),
            )
            .await
            .expect("resume"),
    );
    assert_eq!(resumed.n, 1);
    assert_eq!(resumed.note, "1:human");
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    let done = graph.get_state(&config).await.expect("done");
    assert!(done.interrupts.is_empty());
    assert_eq!(done.values.note, "1:human");
    let _ = saver;
}

#[tokio::test]
async fn map_resume_fills_the_interrupt_with_that_id() {
    let mut builder = GraphBuilder::<Gate>::new();
    builder
        .add_node("ask", |_state: &Gate| {
            let answer = interrupt(json!("q"))?;
            Ok(GateUpdate {
                note: Some(answer.as_str().unwrap_or("").to_string()),
                ..GateUpdate::default()
            })
        })
        .add_edge(START, "ask")
        .add_edge("ask", END);
    let graph = builder.compile().expect("compile");
    let (_saver, config) = memory_config("map");
    let (_state, interrupts) = interrupted(graph.invoke(&gate(0), &config).await.expect("pause"));
    let mut map = std::collections::BTreeMap::new();
    map.insert(interrupts[0].id.clone(), json!("mapped"));
    let resumed = finished(
        graph
            .invoke(&gate(7), &config.resume(Resume::Map(map)))
            .await
            .expect("map"),
    );
    assert_eq!(resumed.note, "mapped");
    assert_eq!(resumed.n, 0);
}

#[tokio::test]
async fn two_interrupt_calls_take_two_values_in_order() {
    let mut builder = GraphBuilder::<Gate>::new();
    builder
        .add_node("ask", |_state: &Gate| {
            let first = interrupt(json!("q1"))?;
            let second = interrupt(json!("q2"))?;
            Ok(GateUpdate {
                note: Some(format!(
                    "{}:{}",
                    first.as_str().unwrap_or(""),
                    second.as_str().unwrap_or("")
                )),
                ..GateUpdate::default()
            })
        })
        .add_edge(START, "ask")
        .add_edge("ask", END);
    let graph = builder.compile().expect("compile");
    let (_saver, config) = memory_config("two");
    let (_state, first_pause) =
        interrupted(graph.invoke(&gate(0), &config).await.expect("first pause"));
    assert_eq!(first_pause.len(), 1);
    assert_eq!(first_pause[0].value, json!("q1"));
    assert_eq!(first_pause[0].id, interrupt_id("", &["ask".to_string()], 0));

    let (_state, second_pause) = interrupted(
        graph
            .invoke(&gate(0), &config.clone().resume(Resume::Value(json!("a"))))
            .await
            .expect("second pause"),
    );
    assert_eq!(second_pause.len(), 1);
    assert_eq!(second_pause[0].value, json!("q2"));
    assert_eq!(
        second_pause[0].id,
        interrupt_id("", &["ask".to_string()], 1)
    );
    assert_ne!(second_pause[0].id, first_pause[0].id);

    let resumed = finished(
        graph
            .invoke(&gate(0), &config.resume(Resume::Value(json!("b"))))
            .await
            .expect("done"),
    );
    assert_eq!(resumed.note, "a:b");
}

#[tokio::test]
async fn map_resume_fills_two_interrupt_calls_in_order() {
    let mut builder = GraphBuilder::<Gate>::new();
    builder
        .add_node("ask", |_state: &Gate| {
            let first = interrupt(json!("q1"))?;
            let second = interrupt(json!("q2"))?;
            Ok(GateUpdate {
                note: Some(format!(
                    "{}:{}",
                    first.as_str().unwrap_or(""),
                    second.as_str().unwrap_or("")
                )),
                ..GateUpdate::default()
            })
        })
        .add_edge(START, "ask")
        .add_edge("ask", END);
    let graph = builder.compile().expect("compile");
    let (_saver, config) = memory_config("two-map");
    let (_state, pause) = interrupted(graph.invoke(&gate(0), &config).await.expect("pause"));
    let mut map = std::collections::BTreeMap::new();
    map.insert(pause[0].id.clone(), json!("a"));
    map.insert(interrupt_id("", &["ask".to_string()], 1), json!("b"));
    let resumed = finished(
        graph
            .invoke(&gate(3), &config.resume(Resume::Map(map)))
            .await
            .expect("both"),
    );
    assert_eq!(resumed.note, "a:b");
    assert_eq!(resumed.n, 0);
}

#[tokio::test]
async fn finished_sibling_is_not_rerun_and_its_write_commits_with_the_step() {
    let left_calls = Arc::new(AtomicUsize::new(0));
    let right_calls = Arc::new(AtomicUsize::new(0));
    let left_node = Arc::clone(&left_calls);
    let right_node = Arc::clone(&right_calls);
    let mut builder = GraphBuilder::<Gate>::new();
    builder
        .add_node("left", move |_state: &Gate| {
            left_node.fetch_add(1, Ordering::SeqCst);
            Ok(GateUpdate {
                items: Some(vec!["left".to_string()]),
                ..GateUpdate::default()
            })
        })
        .add_node("right", move |_state: &Gate| {
            right_node.fetch_add(1, Ordering::SeqCst);
            let answer = interrupt(json!("need"))?;
            Ok(GateUpdate {
                items: Some(vec![format!("right-{}", answer.as_str().unwrap_or(""))]),
                ..GateUpdate::default()
            })
        })
        .add_edge(START, "left")
        .add_edge(START, "right")
        .add_edge("left", END)
        .add_edge("right", END);
    let graph = builder.compile().expect("compile");
    let (saver, config) = memory_config("sibling");
    let (state, interrupts) = interrupted(graph.invoke(&gate(0), &config).await.expect("pause"));
    assert_eq!(interrupts.len(), 1);
    assert_eq!(interrupts[0].value, json!("need"));
    assert!(state.items.is_empty());
    assert_eq!(left_calls.load(Ordering::SeqCst), 1);
    assert_eq!(right_calls.load(Ordering::SeqCst), 1);
    let pending = saver
        .get_tuple(&CheckpointConfig {
            thread_id: "sibling".to_string(),
            checkpoint_ns: String::new(),
            checkpoint_id: None,
        })
        .await
        .expect("tuple")
        .expect("checkpoint")
        .pending_writes;
    assert!(pending.iter().any(|write| {
        write.task_id == "left" && write.channel == "items" && write.value == json!(["left"])
    }));
    let snapshot = graph.get_state(&config).await.expect("state");
    assert_eq!(snapshot.next, ["right".to_string()]);
    assert_eq!(snapshot.interrupts.len(), 1);

    let resumed = finished(
        graph
            .invoke(&gate(0), &config.resume(Resume::Value(json!("go"))))
            .await
            .expect("resume"),
    );
    assert_eq!(left_calls.load(Ordering::SeqCst), 1);
    assert_eq!(right_calls.load(Ordering::SeqCst), 2);
    assert_eq!(resumed.items, ["left".to_string(), "right-go".to_string()]);
}

#[tokio::test]
async fn interrupt_without_a_checkpointer_does_not_commit_the_step() {
    let left_calls = Arc::new(AtomicUsize::new(0));
    let right_calls = Arc::new(AtomicUsize::new(0));
    let after_calls = Arc::new(AtomicUsize::new(0));
    let left_node = Arc::clone(&left_calls);
    let right_node = Arc::clone(&right_calls);
    let after_node = Arc::clone(&after_calls);
    let mut builder = GraphBuilder::<Gate>::new();
    builder
        .add_node("left", move |_state: &Gate| {
            left_node.fetch_add(1, Ordering::SeqCst);
            Ok(GateUpdate {
                items: Some(vec!["left".to_string()]),
                ..GateUpdate::default()
            })
        })
        .add_node("right", move |_state: &Gate| {
            right_node.fetch_add(1, Ordering::SeqCst);
            let _answer = interrupt(json!("x"))?;
            Ok(GateUpdate::default())
        })
        .add_node("after", move |_state: &Gate| {
            after_node.fetch_add(1, Ordering::SeqCst);
            Ok(GateUpdate::default())
        })
        .add_edge(START, "left")
        .add_edge(START, "right")
        .add_edge(["left", "right"], "after")
        .add_edge("after", END);
    let graph = builder.compile().expect("compile");
    let error = graph
        .invoke(&gate(0), &RunConfig::default())
        .await
        .expect_err("no saver");
    assert_eq!(error, Error::MissingThreadId);
    assert_eq!(left_calls.load(Ordering::SeqCst), 1);
    assert_eq!(right_calls.load(Ordering::SeqCst), 1);
    assert_eq!(after_calls.load(Ordering::SeqCst), 0);
}

struct TempDb {
    path: PathBuf,
}

impl TempDb {
    fn new() -> Self {
        let name = format!(
            "langgraph-interrupt-{}-{}.db",
            std::process::id(),
            langgraph::new_checkpoint_id()
        );
        Self {
            path: std::env::temp_dir().join(name),
        }
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        let path = self.path.clone();
        let _ = std::fs::remove_file(&path);
        let mut wal = path.clone().into_os_string();
        wal.push("-wal");
        let _ = std::fs::remove_file(wal);
        let mut shm = path.into_os_string();
        shm.push("-shm");
        let _ = std::fs::remove_file(shm);
    }
}

#[tokio::test]
async fn sqlite_sibling_pending_write_commits_on_resume() {
    let file = TempDb::new();
    let left_calls = Arc::new(AtomicUsize::new(0));
    let right_calls = Arc::new(AtomicUsize::new(0));
    let left_node = Arc::clone(&left_calls);
    let right_node = Arc::clone(&right_calls);
    let mut builder = GraphBuilder::<Gate>::new();
    builder
        .add_node("left", move |_state: &Gate| {
            left_node.fetch_add(1, Ordering::SeqCst);
            Ok(GateUpdate {
                items: Some(vec!["left".to_string()]),
                ..GateUpdate::default()
            })
        })
        .add_node("right", move |_state: &Gate| {
            right_node.fetch_add(1, Ordering::SeqCst);
            let answer = interrupt(json!("need"))?;
            Ok(GateUpdate {
                items: Some(vec![format!("right-{}", answer.as_str().unwrap_or(""))]),
                ..GateUpdate::default()
            })
        })
        .add_edge(START, "left")
        .add_edge(START, "right")
        .add_edge("left", END)
        .add_edge("right", END);
    let graph = builder.compile().expect("compile");
    let saver = Arc::new(SqliteSaver::open(&file.path).expect("open"));
    let config = RunConfig::new()
        .checkpointer(Arc::clone(&saver) as Arc<dyn Checkpointer>)
        .thread_id("sqlite-sibling");
    let (state, interrupts) = interrupted(graph.invoke(&gate(0), &config).await.expect("pause"));
    assert!(state.items.is_empty());
    assert_eq!(interrupts[0].value, json!("need"));
    let resumed = finished(
        graph
            .invoke(&gate(0), &config.resume(Resume::Value(json!("go"))))
            .await
            .expect("resume"),
    );
    assert_eq!(left_calls.load(Ordering::SeqCst), 1);
    assert_eq!(right_calls.load(Ordering::SeqCst), 2);
    assert_eq!(resumed.items, ["left".to_string(), "right-go".to_string()]);
}
